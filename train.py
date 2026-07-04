# train.py
# ------------------------------------------------------------
# 最小版 FL + ResNet18 + Switch-MoE + Meta Expert Aggregation
#
# 路径规则：
#   1. CIFAR10 数据集固定放在当前项目目录 ./data
#   2. torchvision 的 download=True 会自动判断：
#      - 如果 ./data 里已有 CIFAR10，就直接加载
#      - 如果没有，就自动下载
#   3. config.yaml 里的 dataset.data_root 只作为日志目录
# ------------------------------------------------------------

import argparse
import os
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import (
    ResNet18SwitchMoE,
    is_expert_param,
    print_trainable_param_stats,
)

from meta_aggregator import (
    MetaExpertAggregator,
    update_expert_counts,
    counts_to_frequency,
    get_expert_id_from_name,
)


# ------------------------------------------------------------
# 0. 路径工具
# ------------------------------------------------------------
def get_project_root():
    """
    当前项目根目录。

    train.py 放在哪个目录，哪个目录就是项目根目录。
    数据集固定放到：
        项目根目录/data
    """
    return os.path.dirname(os.path.abspath(__file__))


def make_abs_path(path, base_dir=None):
    """
    把路径转成绝对路径。

    相对路径默认相对于当前项目根目录。
    """
    path = os.path.expanduser(str(path))

    if os.path.isabs(path):
        return os.path.abspath(path)

    if base_dir is None:
        base_dir = get_project_root()

    return os.path.abspath(os.path.join(base_dir, path))


def get_fixed_dataset_root():
    """
    固定数据集目录。

    不再从 config.yaml 读取数据集路径。
    CIFAR10 永远固定在：
        当前项目目录/data
    """
    return os.path.join(get_project_root(), "data")


def get_log_root(cfg):
    """
    日志目录。

    config.yaml 里的 dataset.data_root 现在只用来保存日志。
    """
    dataset_cfg = cfg.get("dataset", {})
    log_root = dataset_cfg.get("data_root", "./runs/default")
    log_root = make_abs_path(log_root)

    os.makedirs(log_root, exist_ok=True)

    return log_root


def get_log_name_from_root(log_root):
    """
    根据日志目录自动生成日志文件名。

    例如：
        log_root = ./runs/b5f661
        log_name = b5f661.log
    """
    norm_root = os.path.normpath(log_root)
    run_name = os.path.basename(norm_root)

    if run_name == "":
        run_name = "train"

    return f"{run_name}.log"


def get_log_path(cfg):
    """
    日志文件路径。

    规则：
        dataset.data_root 的最后一级目录名作为日志文件名。

    例如：
        dataset.data_root: ./runs/b5f661
        日志文件: ./runs/b5f661/b5f661.log
    """
    log_root = get_log_root(cfg)
    log_name = get_log_name_from_root(log_root)

    return os.path.join(log_root, log_name)


def copy_config_to_log_root(config_path, cfg):
    """
    把本次使用的 config.yaml 复制到日志目录。
    """
    if config_path is None:
        return None

    if not os.path.isfile(config_path):
        return None

    log_root = get_log_root(cfg)
    dst_path = os.path.join(log_root, "config_used.yaml")

    shutil.copy2(config_path, dst_path)

    return dst_path


# ------------------------------------------------------------
# 1. 日志工具：同时打印到终端和保存到文件
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


def setup_logging(cfg, config_path=None):
    """
    开启日志保存。

    每次运行都会覆盖当前日志文件。
    日志文件名由 dataset.data_root 的最后一级目录名自动决定。

    同时在日志开头打印：
        日志开始时间
        日志保存路径
    """
    log_path = get_log_path(cfg)

    log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)

    copy_config_to_log_root(config_path, cfg)

    display_log_path = os.path.relpath(
        log_path,
        get_project_root(),
    )

    if not display_log_path.startswith("."):
        display_log_path = f"./{display_log_path}"

    print("=" * 80)
    print(f"日志开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志保存路径: {display_log_path}")
    print("=" * 80)

    return log_path


# ------------------------------------------------------------
# 2. 读取配置文件
# ------------------------------------------------------------
def load_config(config_path):
    """
    读取 config.yaml。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ------------------------------------------------------------
# 3. 固定随机种子
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
# 4. 选择设备
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
# 5. 构建模型
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
        top_k=model_cfg.get("top_k", 1),
    )

    return model


# ------------------------------------------------------------
# 6. 加载 CIFAR10 数据集
# ------------------------------------------------------------
def build_datasets(cfg):
    """
    加载 CIFAR10 训练集和测试集。

    数据集固定在项目目录 ./data。
    """
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg.get("name", "cifar10").lower()

    if dataset_name != "cifar10":
        raise ValueError(f"当前代码只支持 cifar10，收到 dataset.name={dataset_name}")

    data_root = get_fixed_dataset_root()
    os.makedirs(data_root, exist_ok=True)

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
# 7. 从测试集划分 class-balanced server validation set
# ------------------------------------------------------------
def split_server_validation_from_test_set(test_set, cfg, seed):
    """
    从 CIFAR10 测试集中划出服务器验证集。

    当前数据流：
        train_set 全部用于客户端训练；
        test_set 先划出 server validation set；
        剩下的 test_set 用于最终测试。
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

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        class_val_indices = class_indices[:samples_per_class]
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
# 8. Dirichlet non-IID 客户端划分
# ------------------------------------------------------------
def dirichlet_partition(labels, num_clients, alpha, seed, min_size=10):
    """
    用 Dirichlet 分布划分 non-IID 客户端数据。
    """
    rng = np.random.default_rng(seed)

    labels = np.array(labels)
    num_classes = int(labels.max()) + 1

    for _ in range(100):
        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(alpha * np.ones(num_clients))

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
# 9. 构建每个客户端的 DataLoader
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
# 10. 构建 server validation DataLoader
# ------------------------------------------------------------
def build_server_val_loader(server_val_set, cfg, device):
    """
    构建服务器验证集 DataLoader。
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
# 11. 构建测试集 DataLoader
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
# 12. Router balance loss
# ------------------------------------------------------------
def compute_router_balance_loss(router_probs):
    """
    计算 router balance loss，缓解 expert 激活塌缩。
    """
    if router_probs.dim() == 3:
        num_experts = router_probs.size(-1)
        router_probs = router_probs.reshape(-1, num_experts)

    elif router_probs.dim() == 2:
        num_experts = router_probs.size(-1)

    else:
        raise ValueError(
            f"router_probs 维度不对，期望 [B, E] 或 [B, T, E]，实际是 {router_probs.shape}"
        )

    mean_probs = router_probs.mean(dim=0)
    target_probs = torch.ones_like(mean_probs) / num_experts

    balance_loss = torch.sum((mean_probs - target_probs) ** 2)

    return balance_loss


# ------------------------------------------------------------
# 13. 统计 expert_loss
# ------------------------------------------------------------
def update_expert_loss_stats(
    expert_loss_sums,
    expert_loss_weights,
    per_sample_loss,
    info,
):
    """
    统计每个 expert 对应样本上的平均 CE loss。

    top1:
        一个样本只贡献给被选中的 expert。

    top2/topk:
        一个样本会贡献给 top-k expert。
        这里用 topk_gates 作为权重。
    """
    num_experts = expert_loss_sums.numel()

    loss_cpu = per_sample_loss.detach().cpu().float()

    if "topk_indices" in info:
        expert_indices = info["topk_indices"].detach().cpu()

        if "topk_gates" in info:
            expert_gates = info["topk_gates"].detach().cpu().float()
        else:
            expert_gates = torch.ones_like(
                expert_indices,
                dtype=torch.float32,
            )
            expert_gates = expert_gates / expert_gates.size(1)

        loss_expand = loss_cpu.unsqueeze(1).expand_as(expert_gates)

        flat_indices = expert_indices.reshape(-1)
        flat_gates = expert_gates.reshape(-1)
        flat_losses = loss_expand.reshape(-1)

        loss_sum = torch.bincount(
            flat_indices,
            weights=flat_losses * flat_gates,
            minlength=num_experts,
        )

        weight_sum = torch.bincount(
            flat_indices,
            weights=flat_gates,
            minlength=num_experts,
        )

    else:
        expert_indices = info["top1_indices"].detach().cpu().reshape(-1)

        loss_sum = torch.bincount(
            expert_indices,
            weights=loss_cpu,
            minlength=num_experts,
        )

        weight_sum = torch.bincount(
            expert_indices,
            minlength=num_experts,
        ).float()

    expert_loss_sums += loss_sum
    expert_loss_weights += weight_sum


# ------------------------------------------------------------
# 14. 本地训练
# ------------------------------------------------------------
def local_train(global_state_dict, train_loader, cfg, device):
    """
    单个客户端本地训练。

    返回：
        local_state_dict
        num_samples
        avg_loss
        expert_freq
        expert_count_values
        expert_loss
    """
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    num_experts = model_cfg["num_experts"]
    router_balance_weight = train_cfg.get("router_balance_weight", 0.0)

    model = build_model(cfg)
    model.load_state_dict(global_state_dict)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
    )

    total_loss = 0.0
    total_samples = 0

    expert_counts = torch.zeros(
        num_experts,
        dtype=torch.long,
    )

    expert_loss_sums = torch.zeros(
        num_experts,
        dtype=torch.float32,
    )

    expert_loss_weights = torch.zeros(
        num_experts,
        dtype=torch.float32,
    )

    local_epochs = train_cfg["local_epochs"]

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, info = model(images, return_info=True)

            if "topk_indices" in info:
                expert_indices = info["topk_indices"]
            else:
                expert_indices = info["top1_indices"]

            update_expert_counts(
                expert_counts=expert_counts,
                expert_indices=expert_indices,
            )

            per_sample_ce_loss = criterion(logits, labels)
            ce_loss = per_sample_ce_loss.mean()

            update_expert_loss_stats(
                expert_loss_sums=expert_loss_sums,
                expert_loss_weights=expert_loss_weights,
                per_sample_loss=per_sample_ce_loss,
                info=info,
            )

            balance_loss = torch.tensor(
                0.0,
                device=device,
            )

            if router_balance_weight > 0:
                if "router_probs" not in info:
                    raise ValueError(
                        "model(images, return_info=True) 没有返回 router_probs，"
                        "请先在 model.py 的 info 里加入 router_probs。"
                    )

                balance_loss = compute_router_balance_loss(
                    info["router_probs"]
                )

            loss = ce_loss + router_balance_weight * balance_loss

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            total_loss += per_sample_ce_loss.detach().sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)

    expert_freq = counts_to_frequency(expert_counts)
    expert_freq = expert_freq.numpy().tolist()

    # 原始 expert 激活次数。
    # shape = [num_experts]
    expert_count_values = expert_counts.numpy().tolist()

    expert_loss_values = []

    for expert_id in range(num_experts):
        weight = expert_loss_weights[expert_id].item()

        if weight > 0:
            value = expert_loss_sums[expert_id].item() / weight
        else:
            value = avg_loss

        expert_loss_values.append(float(value))

    local_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    num_samples = len(train_loader.dataset)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        local_state_dict,
        num_samples,
        avg_loss,
        expert_freq,
        expert_count_values,
        expert_loss_values,
    )


# ------------------------------------------------------------
# 15. 测试全局模型
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
# 16. 普通聚合权重
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


def get_expert_activation_count_weights(
    client_expert_counts,
    expert_id,
    num_clients,
):
    """
    按 expert 激活次数计算某一个 expert 的客户端聚合权重。

    输入：
        client_expert_counts:
            shape = [num_clients, num_experts]

        expert_id:
            当前要聚合的 expert 编号。

    权重：
        weight_i,e = count_i,e / sum_j count_j,e

    如果所有客户端该 expert 的激活次数都为 0，
    就退回 uniform。
    """
    if client_expert_counts is None:
        raise ValueError(
            "expert_agg=expert_activation_count_weighted 时，"
            "必须传入 client_expert_counts。"
        )

    count_values = np.array(
        client_expert_counts,
        dtype=np.float64,
    )

    if count_values.ndim != 2:
        raise ValueError(
            "client_expert_counts 应该是二维，"
            f"shape=[num_clients, num_experts]，实际是 {count_values.shape}"
        )

    input_num_clients, num_experts = count_values.shape

    if input_num_clients != num_clients:
        raise ValueError(
            f"client_expert_counts 的客户端数量不一致: "
            f"count_clients={input_num_clients}, num_clients={num_clients}"
        )

    if expert_id < 0 or expert_id >= num_experts:
        raise ValueError(
            f"expert_id 越界: expert_id={expert_id}, num_experts={num_experts}"
        )

    weights = count_values[:, expert_id].copy()
    weights = np.maximum(weights, 0.0)

    weight_sum = weights.sum()

    if weight_sum <= 1e-12:
        weights = np.ones(
            num_clients,
            dtype=np.float64,
        ) / num_clients
    else:
        weights = weights / weight_sum

    return weights


# ------------------------------------------------------------
# 17. 普通聚合客户端参数
# ------------------------------------------------------------
def aggregate_state_dicts(
    client_state_dicts,
    client_num_samples,
    cfg,
    client_expert_counts=None,
):
    """
    普通聚合函数。

    non-expert 参数：
        使用 non_expert_agg，例如 uniform / sample_weighted。

    expert 参数：
        支持：
            uniform
            sample_weighted
            expert_activation_count_weighted

    expert_activation_count_weighted：
        对每个 expert 单独按激活次数加权。
    """
    agg_cfg = cfg["aggregation"]

    non_expert_method = agg_cfg["non_expert_agg"]
    expert_method = agg_cfg["expert_agg"]

    if expert_method == "meta_network":
        raise ValueError(
            "expert_agg=meta_network 时不应该调用普通 aggregate_state_dicts"
        )

    activation_count_methods = {
        "expert_activation_count_weighted",
        "activation_count_weighted",
    }

    num_clients = len(client_num_samples)

    non_expert_weights = get_aggregation_weights(
        non_expert_method,
        client_num_samples,
    )

    if expert_method in activation_count_methods:
        if client_expert_counts is None:
            raise ValueError(
                f"expert_agg={expert_method} 时，"
                "aggregate_state_dicts 必须传入 client_expert_counts。"
            )

        count_values = np.array(
            client_expert_counts,
            dtype=np.float64,
        )

        if count_values.ndim != 2:
            raise ValueError(
                "client_expert_counts 应该是二维，"
                f"shape=[num_clients, num_experts]，实际是 {count_values.shape}"
            )

        if count_values.shape[0] != num_clients:
            raise ValueError(
                f"client_expert_counts 客户端数量不一致: "
                f"count_clients={count_values.shape[0]}, "
                f"num_clients={num_clients}"
            )

        expert_weights = None

    else:
        expert_weights = get_aggregation_weights(
            expert_method,
            client_num_samples,
        )

    new_state_dict = {}

    state_keys = client_state_dicts[0].keys()

    for name in state_keys:
        first_tensor = client_state_dicts[0][name]

        if not torch.is_floating_point(first_tensor):
            new_state_dict[name] = first_tensor.clone()
            continue

        if is_expert_param(name):
            if expert_method in activation_count_methods:
                expert_id = get_expert_id_from_name(name)

                if expert_id is None:
                    raise ValueError(
                        f"参数名包含 expert，但解析不出 expert_id: {name}"
                    )

                weights = get_expert_activation_count_weights(
                    client_expert_counts=client_expert_counts,
                    expert_id=expert_id,
                    num_clients=num_clients,
                )

            else:
                weights = expert_weights

        else:
            weights = non_expert_weights

        aggregated_tensor = torch.zeros_like(first_tensor)

        for client_id, client_state in enumerate(client_state_dicts):
            aggregated_tensor += client_state[name] * float(weights[client_id])

        new_state_dict[name] = aggregated_tensor

    return new_state_dict


# ------------------------------------------------------------
# 18. 打印客户端划分信息
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
# 19. 打印元网络输出 alpha
# ------------------------------------------------------------
def print_meta_alpha(alpha):
    """
    打印元网络输出的专家聚合权重。
    """
    if alpha is None:
        return

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu()

    print("========== Meta expert alpha ==========")

    num_experts = alpha.shape[0]

    for expert_id in range(num_experts):
        weights = alpha[expert_id].tolist()
        weights_str = ", ".join([f"{w:.3f}" for w in weights])
        print(f"expert {expert_id}: [{weights_str}]")

    print("=======================================")


# ------------------------------------------------------------
# 20. 主训练流程
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

    log_path = setup_logging(
        cfg=cfg,
        config_path=args.config,
    )

    print(f"配置文件路径: {args.config}")

    seed = cfg.get("seed", 1)
    set_seed(seed)

    print(f"随机种子: {seed}")

    device = get_device(cfg)

    print(f"使用设备: {device}")

    train_set, test_set = build_datasets(cfg)

    server_val_set, final_test_set = split_server_validation_from_test_set(
        test_set=test_set,
        cfg=cfg,
        seed=seed,
    )

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
            tau=meta_cfg.get("tau", 1.0),
            active_mask=meta_cfg.get("active_mask", False),
            active_threshold=meta_cfg.get("active_threshold", 0.0),
            min_active_clients_per_expert=meta_cfg.get(
                "min_active_clients_per_expert",
                2,
            ),
            input_features=meta_cfg.get(
                "input_features",
                [
                    "loss_z",
                    "sample_ratio",
                    "expert_freq",
                ],
            ),
        )

        print("已启用 expert_agg = meta_network")

    else:
        print(f"使用普通 expert_agg = {expert_agg}")

    print("========== 训练配置 ==========")
    print(f"rounds              : {rounds}")
    print(f"num_clients         : {num_clients}")
    print(f"clients_per_round   : {clients_per_round}")
    print(f"local_epochs        : {train_cfg['local_epochs']}")
    print(f"batch_size          : {train_cfg['batch_size']}")
    print(f"lr                  : {train_cfg['lr']}")
    print(f"momentum            : {train_cfg.get('momentum', 0.9)}")
    print(f"weight_decay        : {train_cfg.get('weight_decay', 0.0005)}")
    print(f"router_balance_w    : {train_cfg.get('router_balance_weight', 0.0)}")
    print(f"model.top_k         : {model_cfg.get('top_k', 1)}")
    print(f"non_expert_agg      : {non_expert_agg}")
    print(f"expert_agg          : {expert_agg}")

    if expert_agg == "meta_network":
        meta_cfg = cfg.get("meta", {})

        print("---------- Meta 配置 ----------")
        print(f"meta.hidden_dim     : {meta_cfg.get('hidden_dim', 32)}")
        print(f"meta.lr             : {meta_cfg.get('lr', 1e-3)}")
        print(f"meta.steps          : {meta_cfg.get('steps', 1)}")
        print(f"meta.tau            : {meta_cfg.get('tau', 1.0)}")
        print(f"max_val_batches     : {meta_cfg.get('max_val_batches', 4)}")
        print(f"meta.active_mask    : {meta_cfg.get('active_mask', False)}")
        print(f"active_threshold    : {meta_cfg.get('active_threshold', 0.0)}")
        print(
            "min_active_clients : "
            f"{meta_cfg.get('min_active_clients_per_expert', 2)}"
        )

        if meta_aggregator is not None:
            input_features = ", ".join(meta_aggregator.input_feature_names)
            print(f"meta.input_features : [{input_features}]")

    print("==============================")

    for round_id in range(1, rounds + 1):
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in global_model.state_dict().items()
        }

        # 当前版本保持固定顺序：选前 clients_per_round 个客户端。
        selected_clients = list(range(clients_per_round))

        client_state_dicts = []
        client_num_samples = []
        client_losses = []
        client_expert_freqs = []
        client_expert_counts = []
        client_expert_losses = []

        for client_id in selected_clients:
            (
                local_state_dict,
                num_samples,
                avg_loss,
                expert_freq,
                expert_count_values,
                expert_loss,
            ) = local_train(
                global_state_dict=global_state_dict,
                train_loader=client_loaders[client_id],
                cfg=cfg,
                device=device,
            )

            client_state_dicts.append(local_state_dict)
            client_num_samples.append(num_samples)
            client_losses.append(avg_loss)
            client_expert_freqs.append(expert_freq)
            client_expert_counts.append(expert_count_values)
            client_expert_losses.append(expert_loss)

        meta_info = None

        if expert_agg == "meta_network":
            new_global_state_dict, meta_info = meta_aggregator.aggregate(
                model=global_model,
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_expert_counts=client_expert_counts,
                client_expert_losses=client_expert_losses,
                val_loader=server_val_loader,
                non_expert_agg=non_expert_agg,
            )

        else:
            new_global_state_dict = aggregate_state_dicts(
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                cfg=cfg,
                client_expert_counts=client_expert_counts,
            )

        global_model.load_state_dict(new_global_state_dict)

        test_acc, test_loss = evaluate(
            model=global_model,
            test_loader=test_loader,
            device=device,
        )

        best_acc = max(best_acc, test_acc)

        avg_client_loss = float(np.mean(client_losses))

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
# 21. 程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    main()