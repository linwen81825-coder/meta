# train.py
# ------------------------------------------------------------
# FL + ResNet18 + Switch-MoE + Meta Expert Aggregation
#
# 新增：
# 1. 客户端训练前，先用当前 global model 收集每个 expert 的 probe gradient
# 2. 服务器每轮用 server validation / meta-train set 动态构造 query expert gradient
# 3. 计算 alignment score:
#       s_{i,e} = <g_query,e, g_client,i,e>
# 4. 传给 meta_aggregator.py，用 alignment score 作为元网络监督信号
# 5. 仍保留原来的 CE meta-loss 分支，方便对比
# ------------------------------------------------------------

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import ResNet18SwitchMoE, is_expert_param, print_trainable_param_stats

from meta_aggregator import (
    MetaExpertAggregator,
    update_expert_counts,
    counts_to_frequency,
)


# ------------------------------------------------------------
# 0. 日志工具
# ------------------------------------------------------------
class TeeLogger:
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
    data_root = cfg["dataset"].get("data_root", "./data")
    log_dir = os.path.join(data_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "train.log")


def setup_logging(cfg):
    log_path = get_log_path(cfg)
    log_file = open(log_path, "a", encoding="utf-8")

    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)

    print("\n" + "=" * 80)
    print(f"日志开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志保存路径: {log_path}")
    print("=" * 80)

    return log_path


def run_without_file_logging(func, *args, **kwargs):
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return result


# ------------------------------------------------------------
# 1. 配置与随机种子
# ------------------------------------------------------------
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    device_name = cfg.get("device", "cuda")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------
# 2. 模型与数据
# ------------------------------------------------------------
def build_model(cfg):
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    model = ResNet18SwitchMoE(
        num_classes=dataset_cfg["num_classes"],
        num_experts=model_cfg["num_experts"],
        expert_hidden_dim=model_cfg["expert_hidden_dim"],
        top_k=model_cfg.get("top_k", 1),
    )
    return model


def build_datasets(cfg):
    dataset_cfg = cfg["dataset"]
    data_root = dataset_cfg.get("data_root", "./data")
    download_flag = dataset_cfg.get("download", True)

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
        download=download_flag,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download_flag,
        transform=test_transform,
    )

    return train_set, test_set


# ------------------------------------------------------------
# 3. server validation 划分
# ------------------------------------------------------------
def split_server_validation_from_test_set(test_set, cfg, seed):
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
# 4. 客户端划分与 DataLoader
# ------------------------------------------------------------
def dirichlet_partition(labels, num_clients, alpha, seed, min_size=10):
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


def build_client_loaders(train_set, client_indices, cfg, device):
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


def build_server_val_loader(server_val_set, cfg, device):
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


def build_test_loader(test_set, cfg, device):
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
# 5. Router balance loss
# ------------------------------------------------------------
def compute_router_balance_loss(router_probs):
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
# 6. 统计 expert_loss
# ------------------------------------------------------------
def update_expert_loss_stats(
    expert_loss_sums,
    expert_loss_weights,
    per_sample_loss,
    info,
):
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
# 7. Gradient alignment: expert probe gradient
# ------------------------------------------------------------
def get_expert_param_items(model, expert_id):
    prefix = f"moe_head.experts.{expert_id}."
    items = []
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            items.append((name, param))
    return items


def set_only_expert_requires_grad(model, num_experts):
    old_requires_grad = {}

    for name, param in model.named_parameters():
        old_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    for expert_id in range(num_experts):
        for _, param in get_expert_param_items(model, expert_id):
            param.requires_grad_(True)

    return old_requires_grad


def restore_requires_grad(model, old_requires_grad):
    for name, param in model.named_parameters():
        if name in old_requires_grad:
            param.requires_grad_(old_requires_grad[name])


def collect_expert_probe_gradients(
    model,
    loader,
    cfg,
    device,
    max_batches=2,
    min_samples=16,
):
    """
    使用当前 global model 收集每个 expert 的 probe gradient。

    路由方式：
        images -> global backbone -> global router -> top-k expert ids

    梯度计算：
        对属于 expert e 的样本，强制走 expert e；
        只对 expert e 的参数计算 CE loss 梯度；
        backbone/router 不参与反向传播。

    返回：
        expert_grads:
            list，长度 E。
            expert_grads[e] = {param_name: grad_cpu}
            如果样本数不足 min_samples，则 expert_grads[e] = None。

        expert_counts:
            list[int]，每个 expert 的 probe 样本数。
    """
    if loader is None:
        return None, None

    model_cfg = cfg["model"]
    num_experts = model_cfg["num_experts"]
    top_k = model_cfg.get("top_k", 1)

    model.to(device)
    model.eval()

    old_requires_grad = set_only_expert_requires_grad(
        model=model,
        num_experts=num_experts,
    )

    model.zero_grad(set_to_none=True)

    criterion = nn.CrossEntropyLoss(reduction="sum")

    expert_counts = [0 for _ in range(num_experts)]

    try:
        for batch_id, (images, labels) in enumerate(loader):
            if batch_id >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = model.backbone(images)
                router_logits = model.moe_head.router(features)
                router_probs = F.softmax(router_logits, dim=-1)

                _, topk_indices = torch.topk(
                    router_probs,
                    k=top_k,
                    dim=-1,
                )

                features = features.detach()

            for expert_id in range(num_experts):
                if topk_indices.dim() == 2:
                    mask = (topk_indices == expert_id).any(dim=1)
                else:
                    mask = topk_indices == expert_id

                count = int(mask.sum().item())
                if count <= 0:
                    continue

                expert_features = features[mask]
                expert_labels = labels[mask]

                logits = model.moe_head.experts[expert_id](expert_features)
                loss = criterion(logits, expert_labels)

                loss.backward()
                expert_counts[expert_id] += count

        expert_grads = []

        for expert_id in range(num_experts):
            count = expert_counts[expert_id]

            if count < min_samples:
                expert_grads.append(None)
                continue

            grad_dict = {}

            for name, param in get_expert_param_items(model, expert_id):
                if param.grad is None:
                    continue

                grad = param.grad.detach().cpu().float().clone()
                grad = grad / float(count)

                grad_dict[name] = grad

            if len(grad_dict) == 0:
                expert_grads.append(None)
            else:
                expert_grads.append(grad_dict)

        return expert_grads, expert_counts

    finally:
        model.zero_grad(set_to_none=True)
        restore_requires_grad(model, old_requires_grad)


def compute_single_client_alignment_scores(
    query_expert_grads,
    query_expert_counts,
    client_expert_grads,
    client_expert_counts,
    min_samples,
    num_experts,
):
    """
    计算单个客户端和 query expert gradients 的点积对齐分数。

    返回：
        scores: shape [E]
        valid : shape [E]
    """
    scores = np.zeros(num_experts, dtype=np.float32)
    valid = np.zeros(num_experts, dtype=np.bool_)

    if query_expert_grads is None or client_expert_grads is None:
        return scores, valid

    for expert_id in range(num_experts):
        if query_expert_grads[expert_id] is None:
            continue
        if client_expert_grads[expert_id] is None:
            continue
        if query_expert_counts[expert_id] < min_samples:
            continue
        if client_expert_counts[expert_id] < min_samples:
            continue

        dot_value = 0.0
        common_count = 0

        query_grad_dict = query_expert_grads[expert_id]
        client_grad_dict = client_expert_grads[expert_id]

        for name, query_grad in query_grad_dict.items():
            client_grad = client_grad_dict.get(name, None)
            if client_grad is None:
                continue

            dot_value += torch.sum(query_grad * client_grad).item()
            common_count += 1

        if common_count == 0:
            continue

        scores[expert_id] = float(dot_value)
        valid[expert_id] = True

    return scores, valid


def stack_alignment_results(client_alignment_scores, client_alignment_valid):
    """
    输入：
        client_alignment_scores: list，长度 C，每个元素 [E]
        client_alignment_valid : list，长度 C，每个元素 [E]

    输出：
        alignment_scores    : [E, C]
        alignment_valid_mask: [E, C]
    """
    scores = np.asarray(client_alignment_scores, dtype=np.float32)
    valid = np.asarray(client_alignment_valid, dtype=np.bool_)

    # [C, E] -> [E, C]
    scores = scores.T
    valid = valid.T

    return scores, valid


def log_alignment_round_summary(
    log_path,
    round_id,
    alignment_scores,
    alignment_valid_mask,
    query_expert_counts,
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    scores = np.asarray(alignment_scores, dtype=np.float32)
    valid = np.asarray(alignment_valid_mask, dtype=np.bool_)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[ALIGN_ROUND_BEGIN] round={round_id}\n")

        num_experts = scores.shape[0]
        for expert_id in range(num_experts):
            valid_scores = scores[expert_id][valid[expert_id]]
            valid_count = int(valid[expert_id].sum())

            if valid_count > 0:
                score_mean = float(valid_scores.mean())
                score_std = float(valid_scores.std())
                score_min = float(valid_scores.min())
                score_max = float(valid_scores.max())
            else:
                score_mean = 0.0
                score_std = 0.0
                score_min = 0.0
                score_max = 0.0

            query_count = 0
            if query_expert_counts is not None:
                query_count = int(query_expert_counts[expert_id])

            f.write(
                f"[ALIGN_ROUND] "
                f"round={round_id} "
                f"expert={expert_id} "
                f"query_count={query_count} "
                f"valid_clients={valid_count} "
                f"score_mean={score_mean:.8f} "
                f"score_std={score_std:.8f} "
                f"score_min={score_min:.8f} "
                f"score_max={score_max:.8f}\n"
            )

        f.write(f"[ALIGN_ROUND_END] round={round_id}\n")


# ------------------------------------------------------------
# 8. 本地训练
# ------------------------------------------------------------
def local_train(global_state_dict, train_loader, cfg, device):
    """
    单个客户端本地训练。

    如果 meta.use_gradient_alignment=True：
        训练前先用当前 global model 收集 client expert probe gradients。

    返回：
        local_state_dict
        num_samples
        avg_loss
        expert_freq
        expert_loss
        client_expert_grads
        client_expert_grad_counts
    """
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    meta_cfg = cfg.get("meta", {})

    num_experts = model_cfg["num_experts"]
    router_balance_weight = train_cfg.get("router_balance_weight", 0.0)

    use_gradient_alignment = meta_cfg.get("use_gradient_alignment", False)
    grad_probe_batches = meta_cfg.get("grad_probe_batches", 2)
    grad_min_samples = meta_cfg.get("grad_min_samples", 16)

    model = build_model(cfg)
    model.load_state_dict(global_state_dict)
    model.to(device)

    client_expert_grads = None
    client_expert_grad_counts = None

    if use_gradient_alignment:
        client_expert_grads, client_expert_grad_counts = collect_expert_probe_gradients(
            model=model,
            loader=train_loader,
            cfg=cfg,
            device=device,
            max_batches=grad_probe_batches,
            min_samples=grad_min_samples,
        )

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

    expert_counts = torch.zeros(num_experts, dtype=torch.long)
    expert_loss_sums = torch.zeros(num_experts, dtype=torch.float32)
    expert_loss_weights = torch.zeros(num_experts, dtype=torch.float32)

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

            balance_loss = torch.tensor(0.0, device=device)

            if router_balance_weight > 0:
                if "router_probs" not in info:
                    raise ValueError(
                        "model(images, return_info=True) 没有返回 router_probs，"
                        "请先在 model.py 的 info 里加入 router_probs。"
                    )

                balance_loss = compute_router_balance_loss(info["router_probs"])

            loss = ce_loss + router_balance_weight * balance_loss
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            total_loss += per_sample_ce_loss.detach().sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)

    expert_freq = counts_to_frequency(expert_counts)
    expert_freq = expert_freq.numpy().tolist()

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
        expert_loss_values,
        client_expert_grads,
        client_expert_grad_counts,
    )


# ------------------------------------------------------------
# 9. 测试
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, test_loader, device):
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
# 10. 普通聚合
# ------------------------------------------------------------
def get_aggregation_weights(method, client_num_samples):
    num_clients = len(client_num_samples)

    if method == "uniform":
        weights = np.ones(num_clients, dtype=np.float64) / num_clients
    elif method == "sample_weighted":
        client_num_samples = np.array(client_num_samples, dtype=np.float64)
        weights = client_num_samples / client_num_samples.sum()
    else:
        raise ValueError(f"未知聚合方式: {method}")

    return weights


def aggregate_state_dicts(
    client_state_dicts,
    client_num_samples,
    cfg,
):
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
# 11. 打印
# ------------------------------------------------------------
def print_partition_summary(client_indices):
    print("========== 客户端数据划分 ==========")

    for client_id, indices in enumerate(client_indices):
        print(f"client {client_id:02d}: {len(indices)} samples")

    print("===================================")


def print_meta_alpha(alpha):
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
# 12. 主流程
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

    train_set, test_set = run_without_file_logging(
        build_datasets,
        cfg,
    )

    server_val_set, final_test_set = split_server_validation_from_test_set(
        test_set=test_set,
        cfg=cfg,
        seed=seed,
    )

    # 不再划分 meta-select；server validation 全部作为 meta/query pool 使用。
    meta_train_set = server_val_set

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

    meta_train_loader = build_server_val_loader(
        server_val_set=meta_train_set,
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
    meta_cfg = cfg.get("meta", {})

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
        if meta_train_loader is None:
            raise ValueError("使用 meta_network 时，server.val_size 必须大于 0")

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
            use_gradient_alignment=meta_cfg.get("use_gradient_alignment", False),
            alignment_fallback=meta_cfg.get("alignment_fallback", "sample_weighted"),
        )

        print("已启用 expert_agg = meta_network")
    else:
        print(f"使用普通 expert_agg = {expert_agg}")

    print("========== 训练配置 ==========")
    print(f"rounds            : {rounds}")
    print(f"num_clients       : {num_clients}")
    print(f"clients_per_round : {clients_per_round}")
    print(f"local_epochs      : {train_cfg['local_epochs']}")
    print(f"batch_size        : {train_cfg['batch_size']}")
    print(f"lr                : {train_cfg['lr']}")
    print(f"momentum          : {train_cfg.get('momentum', 0.9)}")
    print(f"weight_decay      : {train_cfg.get('weight_decay', 0.0005)}")
    print(f"router_balance_w  : {train_cfg.get('router_balance_weight', 0.0)}")
    print(f"model.top_k       : {model_cfg.get('top_k', 1)}")
    print(f"non_expert_agg    : {non_expert_agg}")
    print(f"expert_agg        : {expert_agg}")

    if expert_agg == "meta_network":
        print("---------- Meta 配置 ----------")
        print(f"meta.hidden_dim             : {meta_cfg.get('hidden_dim', 32)}")
        print(f"meta.lr                     : {meta_cfg.get('lr', 1e-3)}")
        print(f"meta.steps                  : {meta_cfg.get('steps', 1)}")
        print(f"meta.tau                    : {meta_cfg.get('tau', 1.0)}")
        print(f"max_val_batches             : {meta_cfg.get('max_val_batches', 4)}")
        print(f"meta.active_mask            : {meta_cfg.get('active_mask', False)}")
        print(f"active_threshold            : {meta_cfg.get('active_threshold', 0.0)}")
        print(
            "min_active_clients          : "
            f"{meta_cfg.get('min_active_clients_per_expert', 2)}"
        )
        print(f"meta.use_gradient_alignment : {meta_cfg.get('use_gradient_alignment', False)}")
        print(f"meta.grad_probe_batches     : {meta_cfg.get('grad_probe_batches', 2)}")
        print(f"meta.query_grad_batches     : {meta_cfg.get('query_grad_batches', 2)}")
        print(f"meta.grad_min_samples       : {meta_cfg.get('grad_min_samples', 16)}")
        print(f"meta.alignment_fallback     : {meta_cfg.get('alignment_fallback', 'sample_weighted')}")

        if meta_aggregator is not None:
            input_features = ", ".join(meta_aggregator.input_feature_names)
            print(f"meta.input_features         : [{input_features}]")

    print("==============================")

    use_gradient_alignment = meta_cfg.get("use_gradient_alignment", False)
    grad_min_samples = meta_cfg.get("grad_min_samples", 16)
    query_grad_batches = meta_cfg.get("query_grad_batches", 2)

    for round_id in range(1, rounds + 1):
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in global_model.state_dict().items()
        }

        selected_clients = list(range(clients_per_round))

        query_expert_grads = None
        query_expert_grad_counts = None

        if expert_agg == "meta_network" and use_gradient_alignment:
            query_expert_grads, query_expert_grad_counts = collect_expert_probe_gradients(
                model=global_model,
                loader=meta_train_loader,
                cfg=cfg,
                device=device,
                max_batches=query_grad_batches,
                min_samples=grad_min_samples,
            )

        client_state_dicts = []
        client_num_samples = []
        client_losses = []
        client_expert_freqs = []
        client_expert_losses = []

        client_alignment_scores = []
        client_alignment_valid = []

        for client_id in selected_clients:
            (
                local_state_dict,
                num_samples,
                avg_loss,
                expert_freq,
                expert_loss,
                client_expert_grads,
                client_expert_grad_counts,
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
            client_expert_losses.append(expert_loss)

            if expert_agg == "meta_network" and use_gradient_alignment:
                scores_i, valid_i = compute_single_client_alignment_scores(
                    query_expert_grads=query_expert_grads,
                    query_expert_counts=query_expert_grad_counts,
                    client_expert_grads=client_expert_grads,
                    client_expert_counts=client_expert_grad_counts,
                    min_samples=grad_min_samples,
                    num_experts=model_cfg["num_experts"],
                )

                client_alignment_scores.append(scores_i)
                client_alignment_valid.append(valid_i)

                del client_expert_grads

        alignment_scores = None
        alignment_valid_mask = None

        if expert_agg == "meta_network" and use_gradient_alignment:
            alignment_scores, alignment_valid_mask = stack_alignment_results(
                client_alignment_scores=client_alignment_scores,
                client_alignment_valid=client_alignment_valid,
            )

            log_alignment_round_summary(
                log_path=log_path,
                round_id=round_id,
                alignment_scores=alignment_scores,
                alignment_valid_mask=alignment_valid_mask,
                query_expert_counts=query_expert_grad_counts,
            )

        meta_info = None

        if expert_agg == "meta_network":
            new_global_state_dict, meta_info = meta_aggregator.aggregate(
                model=global_model,
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_expert_losses=client_expert_losses,
                val_loader=meta_train_loader,
                non_expert_agg=non_expert_agg,
                alignment_scores=alignment_scores,
                alignment_valid_mask=alignment_valid_mask,
            )
        else:
            new_global_state_dict = aggregate_state_dicts(
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                cfg=cfg,
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
            meta_loss = meta_info.get("meta_loss", 0.0)

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


if __name__ == "__main__":
    main()