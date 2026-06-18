# train.py
# ------------------------------------------------------------
# 最小版 FL + ResNet18 + Switch-MoE 训练脚本
#
# 功能：
# 1. 读取 config.yaml
# 2. 加载 CIFAR10
# 3. 用 Dirichlet 把训练集划分给多个客户端
# 4. 每个客户端本地训练
# 5. 服务端聚合参数
#    - expert 参数使用 expert_agg
#    - non-expert 参数使用 non_expert_agg
# 6. 每轮测试全局模型准确率
# 7. 自动保存训练日志到 dataset.data_root/logs/train.log
#
# 当前只支持两种聚合方式：
#   - uniform：每个客户端权重相同
#   - sample_weighted：按客户端样本数加权
# ------------------------------------------------------------

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 从 model.py 中导入模型和专家参数判断函数
from model import ResNet18SwitchMoE, is_expert_param, print_trainable_param_stats


# ------------------------------------------------------------
# 0. 日志工具：同时打印到终端和保存到文件
# ------------------------------------------------------------
class TeeLogger:
    """
    一个简单的日志分流器。

    作用：
        把 print() 的内容同时输出到：
        1. 终端
        2. 日志文件

    这样你不用再手动写：
        python train.py | tee train.log

    代码里所有 print() 都会自动保存。
    """

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        """
        print() 输出时会自动调用 write()。
        """
        self.terminal.write(message)
        self.log_file.write(message)

        # 及时刷新，防止训练中断时日志没写进去
        self.terminal.flush()
        self.log_file.flush()

    def flush(self):
        """
        兼容 Python 的输出刷新机制。
        """
        self.terminal.flush()
        self.log_file.flush()


def get_log_path(cfg):
    """
    根据 dataset.data_root 自动生成日志保存路径。

    例如：
        data_root: ./data

    日志路径就是：
        ./data/logs/train.log
    """

    data_root = cfg["dataset"].get("data_root", "./data")
    log_dir = os.path.join(data_root, "logs")

    os.makedirs(log_dir, exist_ok=True)

    return os.path.join(log_dir, "train.log")


def setup_logging(cfg):
    """
    开启自动日志保存。

    所有 print() 内容都会同时输出到终端和日志文件。
    """

    log_path = get_log_path(cfg)

    # 追加模式：不会覆盖之前的日志
    log_file = open(log_path, "a", encoding="utf-8")

    # 保存原始 stdout/stderr，并把它们替换成 TeeLogger
    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)

    print("\n" + "=" * 80)
    print(f"日志开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志保存路径: {log_path}")
    print("=" * 80)

    return log_path


# ------------------------------------------------------------
# 1. 读取配置文件
# ------------------------------------------------------------
def load_config(config_path):
    """
    读取 config.yaml。

    config.yaml 里面放所有实验参数，
    这样后面改实验不用频繁改代码。
    """

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ------------------------------------------------------------
# 2. 固定随机种子
# ------------------------------------------------------------
def set_seed(seed):
    """
    固定随机种子，方便复现实验结果。

    注意：
        这里是最小版设置。
        后面如果你要做强确定性复现，可以再加 cudnn、TF32 等设置。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# 3. 选择运行设备
# ------------------------------------------------------------
def get_device(cfg):
    """
    根据配置选择 cuda 或 cpu。

    如果 config 里写 cuda，但机器没有 GPU，
    就自动退回 cpu。
    """

    device_name = cfg.get("device", "cuda")

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ------------------------------------------------------------
# 4. 构建模型
# ------------------------------------------------------------
def build_model(cfg):
    """
    根据 config.yaml 创建 ResNet18SwitchMoE 模型。

    模型结构在 model.py 里定义。
    train.py 不关心模型内部细节。
    """

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    model = ResNet18SwitchMoE(
        num_classes=dataset_cfg["num_classes"],
        num_experts=model_cfg["num_experts"],
        expert_hidden_dim=model_cfg["expert_hidden_dim"],
    )

    return model


# ------------------------------------------------------------
# 5. 加载 CIFAR10 数据集
# ------------------------------------------------------------
def build_datasets(cfg):
    """
    加载 CIFAR10 训练集和测试集。

    这里为了保持最简单，只用了 ToTensor + Normalize。
    后面如果想提升准确率，可以再加 RandomCrop / RandomHorizontalFlip。
    """

    dataset_cfg = cfg["dataset"]
    data_root = dataset_cfg.get("data_root", "./data")

    # CIFAR10 常用均值和方差
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_set, test_set


# ------------------------------------------------------------
# 6. Dirichlet non-IID 客户端划分
# ------------------------------------------------------------
def dirichlet_partition(labels, num_clients, alpha, seed, min_size=10):
    """
    用 Dirichlet 分布划分 non-IID 客户端数据。

    参数：
        labels:
            整个训练集的标签列表。

        num_clients:
            客户端数量。

        alpha:
            Dirichlet 分布参数。
            alpha 越小，non-IID 越强。
            例如 alpha=0.1 表示每个客户端类别分布很偏。

        min_size:
            每个客户端至少要有多少样本。
            如果某次划分导致某个客户端样本太少，就重新划分。

    返回：
        client_indices:
            一个 list。
            client_indices[i] 是第 i 个客户端拥有的训练样本索引。
    """

    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    num_classes = int(labels.max()) + 1

    # 最多尝试 100 次，避免极端情况下某些客户端没有样本
    for _ in range(100):
        client_indices = [[] for _ in range(num_clients)]

        # 按类别逐个划分
        for class_id in range(num_classes):
            # 找到当前类别的所有样本索引
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            # 给每个客户端分配当前类别的比例
            proportions = rng.dirichlet(
                alpha * np.ones(num_clients)
            )

            # 根据比例切分当前类别样本
            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)

            class_splits = np.split(class_indices, split_points)

            # 把当前类别的样本分给各个客户端
            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        # 检查每个客户端样本数是否足够
        client_sizes = [len(indices) for indices in client_indices]

        if min(client_sizes) >= min_size:
            break

    # 打乱每个客户端内部样本顺序
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


# ------------------------------------------------------------
# 7. 构建每个客户端的 DataLoader
# ------------------------------------------------------------
def build_client_loaders(train_set, client_indices, cfg, device):
    """
    根据客户端样本索引，构建每个客户端自己的 DataLoader。
    """

    train_cfg = cfg["train"]

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 2)
    pin_memory = device.type == "cuda"

    client_loaders = []

    for indices in client_indices:
        client_dataset = Subset(train_set, indices)

        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        client_loaders.append(loader)

    return client_loaders


# ------------------------------------------------------------
# 8. 构建测试集 DataLoader
# ------------------------------------------------------------
def build_test_loader(test_set, cfg, device):
    """
    构建测试集 DataLoader。
    """

    train_cfg = cfg["train"]

    batch_size = train_cfg.get("test_batch_size", 256)
    num_workers = train_cfg.get("num_workers", 2)
    pin_memory = device.type == "cuda"

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader


# ------------------------------------------------------------
# 9. 本地训练
# ------------------------------------------------------------
def local_train(global_state_dict, train_loader, cfg, device):
    """
    单个客户端本地训练。

    输入：
        global_state_dict:
            当前服务端全局模型参数。

        train_loader:
            当前客户端的数据。

    输出：
        local_state_dict:
            当前客户端训练后的模型参数。

        num_samples:
            当前客户端样本数量。

        avg_loss:
            当前客户端本地训练平均 loss。
    """

    train_cfg = cfg["train"]

    # 每个客户端都从当前 global model 开始训练
    model = build_model(cfg)
    model.load_state_dict(global_state_dict)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
    )

    total_loss = 0.0
    total_samples = 0

    local_epochs = train_cfg["local_epochs"]

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)

    # 把本地模型参数搬回 CPU，方便服务端聚合
    local_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    num_samples = len(train_loader.dataset)

    # 删除本地模型，减少显存占用
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return local_state_dict, num_samples, avg_loss


# ------------------------------------------------------------
# 10. 测试全局模型
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, test_loader, device):
    """
    在测试集上评估全局模型。

    返回：
        acc:
            测试准确率，百分制。

        avg_loss:
            测试平均 loss。
    """

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size

    acc = correct / total * 100.0
    avg_loss = total_loss / total

    return acc, avg_loss


# ------------------------------------------------------------
# 11. 计算聚合权重
# ------------------------------------------------------------
def get_aggregation_weights(method, client_num_samples):
    """
    根据聚合方式计算客户端权重。

    当前支持：
        uniform:
            每个客户端权重一样。

        sample_weighted:
            按客户端样本数加权。
    """

    num_clients = len(client_num_samples)

    if method == "uniform":
        weights = np.ones(num_clients, dtype=np.float64) / num_clients

    elif method == "sample_weighted":
        client_num_samples = np.array(client_num_samples, dtype=np.float64)
        weights = client_num_samples / client_num_samples.sum()

    else:
        raise ValueError(f"未知聚合方式: {method}")

    return weights


# ------------------------------------------------------------
# 12. 聚合客户端参数
# ------------------------------------------------------------
def aggregate_state_dicts(
    client_state_dicts,
    client_num_samples,
    cfg,
):
    """
    服务端聚合多个客户端上传的模型参数。

    核心逻辑：
        如果参数名属于 expert：
            使用 expert_agg 聚合方式。

        否则：
            使用 non_expert_agg 聚合方式。

    也就是说：
        backbone、router 用 non_expert_agg
        experts 用 expert_agg
    """

    agg_cfg = cfg["aggregation"]

    non_expert_method = agg_cfg["non_expert_agg"]
    expert_method = agg_cfg["expert_agg"]

    non_expert_weights = get_aggregation_weights(
        non_expert_method,
        client_num_samples,
    )

    expert_weights = get_aggregation_weights(
        expert_method,
        client_num_samples,
    )

    new_state_dict = {}

    # 所有客户端模型结构一样，所以直接拿第一个客户端的 key
    state_keys = client_state_dicts[0].keys()

    for name in state_keys:
        # 有些 state_dict 里面是整数类型，例如 BatchNorm 的 num_batches_tracked
        # 整数不能做加权平均，所以这里直接使用第一个客户端的值
        if not torch.is_floating_point(client_state_dicts[0][name]):
            new_state_dict[name] = client_state_dicts[0][name].clone()
            continue

        # 判断当前参数属于 expert 还是 non-expert
        if is_expert_param(name):
            weights = expert_weights
        else:
            weights = non_expert_weights

        # 加权平均
        aggregated_tensor = torch.zeros_like(client_state_dicts[0][name])

        for client_id, client_state in enumerate(client_state_dicts):
            aggregated_tensor += client_state[name] * float(weights[client_id])

        new_state_dict[name] = aggregated_tensor

    return new_state_dict


# ------------------------------------------------------------
# 13. 打印客户端划分信息
# ------------------------------------------------------------
def print_partition_summary(client_indices):
    """
    打印每个客户端有多少样本。

    这个函数只是帮助你确认划分是否正常。
    因为已经开启了 TeeLogger，所以这里的 print 也会自动写入日志。
    """

    print("========== 客户端数据划分 ==========")

    for client_id, indices in enumerate(client_indices):
        print(f"client {client_id:02d}: {len(indices)} samples")

    print("===================================")


# ------------------------------------------------------------
# 14. 主训练流程
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径",
    )

    args = parser.parse_args()

    # 读取配置
    cfg = load_config(args.config)

    # 开启自动日志保存
    # 注意：这行之后，后面所有 print 都会同时写入日志文件。
    setup_logging(cfg)

    print(f"配置文件路径: {args.config}")

    # 固定随机种子
    seed = cfg.get("seed", 1)
    set_seed(seed)
    print(f"随机种子: {seed}")

    # 选择设备
    device = get_device(cfg)
    print(f"使用设备: {device}")

    # 加载数据集
    train_set, test_set = build_datasets(cfg)

    # Dirichlet 划分客户端数据
    dataset_cfg = cfg["dataset"]

    client_indices = dirichlet_partition(
        labels=train_set.targets,
        num_clients=dataset_cfg["num_clients"],
        alpha=dataset_cfg["alpha"],
        seed=seed,
    )

    print_partition_summary(client_indices)

    # 构建客户端 DataLoader
    client_loaders = build_client_loaders(
        train_set=train_set,
        client_indices=client_indices,
        cfg=cfg,
        device=device,
    )

    # 构建测试集 DataLoader
    test_loader = build_test_loader(
        test_set=test_set,
        cfg=cfg,
        device=device,
    )

    # 初始化全局模型
    global_model = build_model(cfg)
    global_model.to(device)

    # 打印参数统计
    # 因为已经开启了 TeeLogger，这里的输出也会自动保存到日志。
    print_trainable_param_stats(global_model)

    train_cfg = cfg["train"]
    num_clients = dataset_cfg["num_clients"]
    rounds = train_cfg["rounds"]
    clients_per_round = train_cfg["clients_per_round"]

    if clients_per_round > num_clients:
        raise ValueError("clients_per_round 不能大于 num_clients")

    best_acc = 0.0

    print("========== 训练配置 ==========")
    print(f"rounds            : {rounds}")
    print(f"num_clients       : {num_clients}")
    print(f"clients_per_round : {clients_per_round}")
    print(f"local_epochs      : {train_cfg['local_epochs']}")
    print(f"batch_size        : {train_cfg['batch_size']}")
    print(f"lr                : {train_cfg['lr']}")
    print(f"momentum          : {train_cfg.get('momentum', 0.9)}")
    print(f"weight_decay      : {train_cfg.get('weight_decay', 0.0005)}")
    print(f"non_expert_agg    : {cfg['aggregation']['non_expert_agg']}")
    print(f"expert_agg        : {cfg['aggregation']['expert_agg']}")
    print("==============================")

    # --------------------------------------------------------
    # FL 主循环
    # --------------------------------------------------------
    for round_id in range(1, rounds + 1):
        # 当前全局模型参数
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in global_model.state_dict().items()
        }

        # 每轮固定按顺序选择客户端，不再随机选择
        # 例如：
        #   num_clients=10, clients_per_round=10 时，每轮都是 client 0~9 全部参与
        #   num_clients=10, clients_per_round=5 时，每轮固定使用 client 0~4
        selected_clients = list(range(clients_per_round))

        client_state_dicts = []
        client_num_samples = []
        client_losses = []

        # ----------------------------------------------------
        # 每个客户端本地训练
        # ----------------------------------------------------
        for client_id in selected_clients:
            local_state_dict, num_samples, avg_loss = local_train(
                global_state_dict=global_state_dict,
                train_loader=client_loaders[client_id],
                cfg=cfg,
                device=device,
            )

            client_state_dicts.append(local_state_dict)
            client_num_samples.append(num_samples)
            client_losses.append(avg_loss)

        # ----------------------------------------------------
        # 服务端聚合
        # ----------------------------------------------------
        new_global_state_dict = aggregate_state_dicts(
            client_state_dicts=client_state_dicts,
            client_num_samples=client_num_samples,
            cfg=cfg,
        )

        # 更新全局模型
        global_model.load_state_dict(new_global_state_dict)

        # ----------------------------------------------------
        # 测试全局模型
        # ----------------------------------------------------
        test_acc, test_loss = evaluate(
            model=global_model,
            test_loader=test_loader,
            device=device,
        )

        best_acc = max(best_acc, test_acc)

        avg_client_loss = float(np.mean(client_losses))

        print(
            f"Round {round_id:03d} | "
            f"client_loss={avg_client_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"acc={test_acc:.2f}% | "
            f"best={best_acc:.2f}%"
        )

    print("=" * 80)
    print(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最终 best_acc: {best_acc:.2f}%")
    print("=" * 80)


# ------------------------------------------------------------
# 15. 程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    main()