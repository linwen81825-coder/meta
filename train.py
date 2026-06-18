# train.py
# ------------------------------------------------------------
# 最小版 FL + ResNet18 + Switch-MoE + Meta Expert Aggregation

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

# 新增功能：
# 1. 从测试集划分 server validation set
# 2. 客户端本地训练时统计：
#    - client_loss
#    - expert_activation_frequency
# 3. 当 expert_agg = meta_network 时：
#    - 调用 meta_aggregator.py 里的 MetaExpertAggregator
#    - 用服务器验证集 loss 更新元网络
#    - 再用元网络输出最终 expert 聚合权重
# 4. 保留自动日志保存
# 5. 保留固定顺序客户端，不随机选择
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

from model import ResNet18SwitchMoE, is_expert_param, print_trainable_param_stats

# 你的元网络专家聚合算法
from meta_aggregator import (
    MetaExpertAggregator,
    update_expert_counts,
    counts_to_frequency,
)


# ------------------------------------------------------------
# 0. 日志工具：同时打印到终端和保存到文件
# ------------------------------------------------------------
class TeeLogger:
    """
    把 print() 的内容同时输出到终端和日志文件。
    """

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

        self.terminal.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def get_log_path(cfg):
    """
    日志默认保存到：
        dataset.data_root/logs/train.log
    """

    data_root = cfg["dataset"].get("data_root", "./data")
    log_dir = os.path.join(data_root, "logs")

    os.makedirs(log_dir, exist_ok=True)

    return os.path.join(log_dir, "train.log")


def setup_logging(cfg):
    """
    开启自动日志保存。
    """

    log_path = get_log_path(cfg)

    log_file = open(log_path, "a", encoding="utf-8")

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
    """

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ------------------------------------------------------------
# 2. 固定随机种子
# ------------------------------------------------------------
def set_seed(seed):
    """
    固定随机种子。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# 3. 选择设备
# ------------------------------------------------------------
def get_device(cfg):
    """
    根据配置选择 cuda 或 cpu。
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
    创建 ResNet18SwitchMoE 模型。
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
    """

    dataset_cfg = cfg["dataset"]
    data_root = dataset_cfg.get("data_root", "./data")

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
# 6. 从测试集划分 class-balanced server validation set
# ------------------------------------------------------------
def split_server_validation_from_test_set(test_set, cfg, seed):
    """
    从 CIFAR10 测试集中划出服务器验证集。

    当前数据流：
        train_set 全部用于客户端训练；
        test_set 先划出 server validation set；
        剩下的 test_set 用于最终测试。

    默认做 class-balanced 划分：
        server.val_size = 1000
        num_classes = 10

    那么每类取：
        1000 / 10 = 100 张

    作为 server validation set。
    剩下的测试样本作为 final test set。
    """

    server_cfg = cfg.get("server", {})
    dataset_cfg = cfg["dataset"]

    val_size = server_cfg.get("val_size", 1000)
    num_classes = dataset_cfg["num_classes"]

    total_size = len(test_set)

    if val_size <= 0:
        server_val_set = None
        final_test_set = test_set

        print("========== Server 验证集划分 ==========")
        print("server.val_size <= 0，不划分 server validation set")
        print(f"final test samples: {len(final_test_set)}")
        print("======================================")

        return server_val_set, final_test_set

    if val_size >= total_size:
        raise ValueError("server.val_size 不能大于等于测试集大小")

    if val_size % num_classes != 0:
        raise ValueError(
            "为了做 class-balanced server validation set，"
            "server.val_size 必须能被 num_classes 整除。"
        )

    samples_per_class = val_size // num_classes

    rng = np.random.default_rng(seed)
    labels = np.array(test_set.targets)

    server_val_indices = []
    final_test_indices = []

    # 逐类抽样，保证 server validation 每个类别数量一样
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        # 当前类别前 samples_per_class 张给 server validation
        class_val_indices = class_indices[:samples_per_class]

        # 当前类别剩下的给最终测试
        class_test_indices = class_indices[samples_per_class:]

        server_val_indices.extend(class_val_indices.tolist())
        final_test_indices.extend(class_test_indices.tolist())

    rng.shuffle(server_val_indices)
    rng.shuffle(final_test_indices)

    server_val_set = Subset(test_set, server_val_indices)
    final_test_set = Subset(test_set, final_test_indices)

    print("========== Server 验证集划分 ==========")
    print("划分来源: CIFAR10 test set")
    print(f"server val samples       : {len(server_val_set)}")
    print(f"final test samples       : {len(final_test_set)}")
    print(f"server val per class     : {samples_per_class}")
    print("======================================")

    return server_val_set, final_test_set


# ------------------------------------------------------------
# 7. Dirichlet non-IID 客户端划分
# ------------------------------------------------------------
def dirichlet_partition(labels, num_clients, alpha, seed, min_size=10):
    """
    用 Dirichlet 分布划分 non-IID 客户端数据。

    输入的 labels 是完整 train_set 的标签。
    返回的 client_indices 是相对于 train_set 的索引。
    """

    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    num_classes = int(labels.max()) + 1

    for _ in range(100):
        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                alpha * np.ones(num_clients)
            )

            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)

            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        client_sizes = [len(indices) for indices in client_indices]

        if min(client_sizes) >= min_size:
            break

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


# ------------------------------------------------------------
# 8. 构建每个客户端的 DataLoader
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
# 9. 构建 server validation DataLoader
# ------------------------------------------------------------
def build_server_val_loader(server_val_set, cfg, device):
    """
    构建服务器验证集 DataLoader。

    这个验证集只给元网络训练使用。
    """

    if server_val_set is None:
        return None

    train_cfg = cfg["train"]
    server_cfg = cfg.get("server", {})

    batch_size = server_cfg.get(
        "val_batch_size",
        train_cfg.get("test_batch_size", 256),
    )

    num_workers = train_cfg.get("num_workers", 2)
    pin_memory = device.type == "cuda"

    loader = DataLoader(
        server_val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


# ------------------------------------------------------------
# 10. 构建测试集 DataLoader
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
# 11. 本地训练
# ------------------------------------------------------------
def local_train(global_state_dict, train_loader, cfg, device):
    """
    单个客户端本地训练。

    返回：
        local_state_dict
        num_samples
        avg_loss
        expert_freq

    其中 expert_freq 是当前客户端每个 expert 的激活频率。
    """

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    num_experts = model_cfg["num_experts"]

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

    # 统计当前客户端每个 expert 被激活多少次
    expert_counts = torch.zeros(
        num_experts,
        dtype=torch.long,
    )

    local_epochs = train_cfg["local_epochs"]

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # return_info=True 会返回 router 的 top1 expert id
            logits, info = model(images, return_info=True)

            # 更新 expert 激活次数
            update_expert_counts(
                expert_counts=expert_counts,
                top1_indices=info["top1_indices"],
            )

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)

    expert_freq = counts_to_frequency(expert_counts)
    expert_freq = expert_freq.numpy().tolist()

    local_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    num_samples = len(train_loader.dataset)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return local_state_dict, num_samples, avg_loss, expert_freq


# ------------------------------------------------------------
# 12. 测试全局模型
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, test_loader, device):
    """
    在测试集上评估全局模型。
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
# 13. 普通聚合权重
# ------------------------------------------------------------
def get_aggregation_weights(method, client_num_samples):
    """
    计算 uniform 或 sample_weighted 聚合权重。
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
# 14. 普通聚合客户端参数
# ------------------------------------------------------------
def aggregate_state_dicts(
    client_state_dicts,
    client_num_samples,
    cfg,
):
    """
    普通聚合函数。

    当 expert_agg 是 uniform 或 sample_weighted 时使用这个函数。

    如果 expert_agg = meta_network，
    不走这里，而是调用 MetaExpertAggregator。
    """

    agg_cfg = cfg["aggregation"]

    non_expert_method = agg_cfg["non_expert_agg"]
    expert_method = agg_cfg["expert_agg"]

    if expert_method == "meta_network":
        raise ValueError("expert_agg=meta_network 时不应该调用普通 aggregate_state_dicts")

    non_expert_weights = get_aggregation_weights(
        non_expert_method,
        client_num_samples,
    )

    expert_weights = get_aggregation_weights(
        expert_method,
        client_num_samples,
    )

    new_state_dict = {}
    state_keys = client_state_dicts[0].keys()

    for name in state_keys:
        if not torch.is_floating_point(client_state_dicts[0][name]):
            new_state_dict[name] = client_state_dicts[0][name].clone()
            continue

        if is_expert_param(name):
            weights = expert_weights
        else:
            weights = non_expert_weights

        aggregated_tensor = torch.zeros_like(client_state_dicts[0][name])

        for client_id, client_state in enumerate(client_state_dicts):
            aggregated_tensor += client_state[name] * float(weights[client_id])

        new_state_dict[name] = aggregated_tensor

    return new_state_dict


# ------------------------------------------------------------
# 15. 打印客户端划分信息
# ------------------------------------------------------------
def print_partition_summary(client_indices):
    """
    打印每个客户端有多少样本。
    """

    print("========== 客户端数据划分 ==========")

    for client_id, indices in enumerate(client_indices):
        print(f"client {client_id:02d}: {len(indices)} samples")

    print("===================================")


# ------------------------------------------------------------
# 16. 打印 meta alpha
# ------------------------------------------------------------
def print_meta_alpha(alpha):
    """
    打印元网络输出的专家聚合权重。

    alpha shape:
        [num_experts, num_clients]
    """

    if alpha is None:
        return

    print("========== Meta expert alpha ==========")

    num_experts = alpha.shape[0]

    for expert_id in range(num_experts):
        weights = alpha[expert_id].tolist()
        weights_str = ", ".join([f"{w:.3f}" for w in weights])
        print(f"expert {expert_id}: [{weights_str}]")

    print("=======================================")


# ------------------------------------------------------------
# 17. 主训练流程
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

    cfg = load_config(args.config)

    log_path = setup_logging(cfg)

    print(f"配置文件路径: {args.config}")

    seed = cfg.get("seed", 1)
    set_seed(seed)
    print(f"随机种子: {seed}")

    device = get_device(cfg)
    print(f"使用设备: {device}")

    # --------------------------------------------------------
    # 加载数据
    # --------------------------------------------------------
    train_set, test_set = build_datasets(cfg)

    # --------------------------------------------------------
    # 从测试集划分 server validation set
    #   server_val_set: 给元网络训练
    #   final_test_set: 给最终测试
    # --------------------------------------------------------
    server_val_set, final_test_set = split_server_validation_from_test_set(
        test_set=test_set,
        cfg=cfg,
        seed=seed,
    )

    # --------------------------------------------------------
    # 客户端训练数据使用完整 train_set
    # 不再从 train_set 里划 server validation
    # --------------------------------------------------------
    dataset_cfg = cfg["dataset"]

    client_indices = dirichlet_partition(
        labels=train_set.targets,
        num_clients=dataset_cfg["num_clients"],
        alpha=dataset_cfg["alpha"],
        seed=seed,
    )

    print_partition_summary(client_indices)

    client_loaders = build_client_loaders(
        train_set=train_set,
        client_indices=client_indices,
        cfg=cfg,
        device=device,
    )

    server_val_loader = build_server_val_loader(
        server_val_set=server_val_set,
        cfg=cfg,
        device=device,
    )

    test_loader = build_test_loader(
        test_set=final_test_set,
        cfg=cfg,
        device=device,
    )

    # --------------------------------------------------------
    # 初始化全局模型
    # --------------------------------------------------------
    global_model = build_model(cfg)
    global_model.to(device)

    print_trainable_param_stats(global_model)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    agg_cfg = cfg["aggregation"]

    num_clients = dataset_cfg["num_clients"]
    rounds = train_cfg["rounds"]
    clients_per_round = train_cfg["clients_per_round"]

    if clients_per_round > num_clients:
        raise ValueError("clients_per_round 不能大于 num_clients")

    best_acc = 0.0

    # --------------------------------------------------------
    # 如果 expert_agg=meta_network，就初始化元聚合器
    # --------------------------------------------------------
    expert_agg = agg_cfg["expert_agg"]
    non_expert_agg = agg_cfg["non_expert_agg"]

    meta_aggregator = None

    if expert_agg == "meta_network":
        if server_val_loader is None:
            raise ValueError("使用 meta_network 时，server.val_size 必须大于 0")

        meta_cfg = cfg.get("meta", {})

        meta_aggregator = MetaExpertAggregator(
            num_experts=model_cfg["num_experts"],
            device=device,
            hidden_dim=meta_cfg.get("hidden_dim", 32),
            lr=meta_cfg.get("lr", 1e-3),
            meta_steps=meta_cfg.get("steps", 1),
            max_val_batches=meta_cfg.get("max_val_batches", 4),
            train_log_path=log_path,
        )

        print("已启用 expert_agg = meta_network")
    else:
        print(f"使用普通 expert_agg = {expert_agg}")

    # --------------------------------------------------------
    # 打印训练配置
    # --------------------------------------------------------
    print("========== 训练配置 ==========")
    print(f"rounds            : {rounds}")
    print(f"num_clients       : {num_clients}")
    print(f"clients_per_round : {clients_per_round}")
    print(f"local_epochs      : {train_cfg['local_epochs']}")
    print(f"batch_size        : {train_cfg['batch_size']}")
    print(f"lr                : {train_cfg['lr']}")
    print(f"momentum          : {train_cfg.get('momentum', 0.9)}")
    print(f"weight_decay      : {train_cfg.get('weight_decay', 0.0005)}")
    print(f"non_expert_agg    : {non_expert_agg}")
    print(f"expert_agg        : {expert_agg}")

    if expert_agg == "meta_network":
        meta_cfg = cfg.get("meta", {})
        print("---------- Meta 配置 ----------")
        print(f"meta.hidden_dim    : {meta_cfg.get('hidden_dim', 32)}")
        print(f"meta.lr            : {meta_cfg.get('lr', 1e-3)}")
        print(f"meta.steps         : {meta_cfg.get('steps', 1)}")
        print(f"max_val_batches    : {meta_cfg.get('max_val_batches', 4)}")

    print("==============================")

    # --------------------------------------------------------
    # FL 主循环
    # --------------------------------------------------------
    for round_id in range(1, rounds + 1):
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in global_model.state_dict().items()
        }

        # 固定顺序选择客户端，不随机
        selected_clients = list(range(clients_per_round))

        client_state_dicts = []
        client_num_samples = []
        client_losses = []
        client_expert_freqs = []

        # ----------------------------------------------------
        # 客户端本地训练
        # ----------------------------------------------------
        for client_id in selected_clients:
            local_state_dict, num_samples, avg_loss, expert_freq = local_train(
                global_state_dict=global_state_dict,
                train_loader=client_loaders[client_id],
                cfg=cfg,
                device=device,
            )

            client_state_dicts.append(local_state_dict)
            client_num_samples.append(num_samples)
            client_losses.append(avg_loss)
            client_expert_freqs.append(expert_freq)

        # ----------------------------------------------------
        # 服务端聚合
        # ----------------------------------------------------
        meta_info = None

        if expert_agg == "meta_network":
            new_global_state_dict, meta_info = meta_aggregator.aggregate(
                model=global_model,
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                val_loader=server_val_loader,
                non_expert_agg=non_expert_agg,
            )

        else:
            new_global_state_dict = aggregate_state_dicts(
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                cfg=cfg,
            )

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

        # ----------------------------------------------------
        # 打印日志
        # ----------------------------------------------------
        if meta_info is not None:
            meta_loss = meta_info.get("meta_loss", None)

            print(
                f"Round {round_id:03d} | "
                f"client_loss={avg_client_loss:.4f} | "
                f"meta_loss={meta_loss:.4f} | "
                f"test_loss={test_loss:.4f} | "
                f"acc={test_acc:.2f}% | "
                f"best={best_acc:.2f}%"
            )

            # 每轮都打印 alpha 会比较长。
            # 如果你想看每个 expert 的聚合权重，就取消下面两行注释。
            # alpha = meta_info.get("alpha", None)
            # print_meta_alpha(alpha)

        else:
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
# 18. 程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    main()