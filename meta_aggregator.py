# meta_aggregator.py
# ------------------------------------------------------------
# 元网络专家聚合模块
#
# 这个文件只做一件事：
#   用服务器验证集训练一个元网络，让它输出 expert 聚合权重。
#
# 算法流程：
#   客户端统计特征 [client_loss_z, sample_ratio, expert_activation_frequency]
#       ↓
#   元网络输出每个 expert 的客户端聚合权重 alpha
#       ↓
#   用 alpha 临时聚合 expert 参数
#       ↓
#   把临时参数放进模型，在 server validation set 上算 CE loss
#       ↓
#   用 CE loss 反向更新元网络
#       ↓
#   用更新后的元网络重新输出最终 alpha
#       ↓
#   聚合得到最终 expert 参数
#
# 日志：
#   元网络输入和最终聚合权重都会追加写入当前实验的 train.log。
#   日志路径由 train.py 传进来，不能写死。
#
# 注意：
#   这里不能用 model.load_state_dict() 做临时模型。
#   因为 load_state_dict() 不可微，验证集 loss 不能反传到元网络。
#
#   所以这里用 torch.func.functional_call()。
#   这样临时聚合出来的 expert 参数仍然和元网络 alpha 有计算图关系。
# ------------------------------------------------------------

import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from model import is_expert_param


# ------------------------------------------------------------
# 1. 从参数名里解析 expert id
# ------------------------------------------------------------
def get_expert_id_from_name(name):
    """
    从参数名里解析 expert 编号。

    例如：
        moe_head.experts.0.net.0.weight -> 0
        moe_head.experts.1.net.2.weight -> 1
        moe_head.experts.3.net.4.bias   -> 3

    如果这个参数不是 expert 参数，就返回 None。
    """

    match = re.search(r"experts\.(\d+)", name)

    if match is None:
        return None

    return int(match.group(1))


# ------------------------------------------------------------
# 2. 普通聚合权重
# ------------------------------------------------------------
def get_basic_weights(method, client_num_samples):
    """
    计算普通客户端聚合权重。

    当前支持：
        uniform:
            每个客户端权重相同。

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
        raise ValueError(f"未知普通聚合方式: {method}")

    return weights


# ------------------------------------------------------------
# 3. 统计 expert 激活次数的小工具
# ------------------------------------------------------------
def update_expert_counts(expert_counts, expert_indices):
    """
    根据 expert id 更新 expert 激活次数。

    支持两种输入：

        top1:
            expert_indices shape = [B]

        topk:
            expert_indices shape = [B, K]

    top2 时，一个样本会贡献两个 expert 激活。
    """

    num_experts = expert_counts.numel()

    # top2 情况下 expert_indices 是 [B, K]
    # bincount 需要一维，所以这里拉平
    expert_indices = expert_indices.detach().cpu().reshape(-1)

    batch_counts = torch.bincount(
        expert_indices,
        minlength=num_experts,
    )

    expert_counts += batch_counts


def counts_to_frequency(expert_counts):
    """
    把 expert 激活次数转换成 expert 激活频率。

    例如：
        counts = [10, 20, 0, 70]

    转成：
        freq = [0.1, 0.2, 0.0, 0.7]
    """

    total = expert_counts.sum().item()

    if total <= 0:
        return torch.zeros_like(expert_counts, dtype=torch.float32)

    return expert_counts.float() / total


# ------------------------------------------------------------
# 4. 元网络
# ------------------------------------------------------------
class MetaWeightNet(nn.Module):
    """
    元网络。

    对每一个 client-expert pair 输入三个统计特征：

        [client_loss_z, sample_ratio, expert_activation_frequency]

    输出一个 score。

    然后对同一个 expert 下的所有客户端 score 做 softmax，
    得到这个 expert 的客户端聚合权重 alpha。

    输入维度：
        3

    输出维度：
        1
    """

    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        x:
            shape: [N, 3]

        return:
            score:
                shape: [N]
        """

        score = self.net(x).squeeze(-1)
        return score


# ------------------------------------------------------------
# 5. Meta Expert Aggregator
# ------------------------------------------------------------
class MetaExpertAggregator:
    """
    元网络专家聚合器。

    它负责：
        1. 根据客户端统计特征生成 alpha
        2. 用 alpha 临时聚合 expert 参数
        3. 在 server validation set 上计算 CE loss
        4. 用 CE loss 更新元网络
        5. 用更新后的元网络输出最终 alpha
        6. 返回最终聚合后的完整 state_dict
    """

    def __init__(
        self,
        num_experts,
        device,
        hidden_dim=32,
        lr=1e-3,
        meta_steps=1,
        max_val_batches=4,
        train_log_path=None,
    ):
        self.num_experts = num_experts
        self.device = device
        self.meta_steps = meta_steps
        self.max_val_batches = max_val_batches

        # 现在元网络输入是 3 维：
        # [loss_z, sample_ratio, expert_freq]
        self.meta_net = MetaWeightNet(
            input_dim=3,
            hidden_dim=hidden_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr=lr,
        )

        # 当前是第几轮聚合。
        # 只用于写日志，不影响训练。
        self.round_id = 0

        # 统一写入当前实验的训练日志文件。
        # 这个路径由 train.py 传进来，不能写死。
        if train_log_path is None:
            raise ValueError(
                "MetaExpertAggregator 必须传入 train_log_path，不能使用写死日志路径。"
            )

        self.train_log_path = train_log_path

    # --------------------------------------------------------
    # 5.1 构造元网络输入特征
    # --------------------------------------------------------
    def build_meta_features(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
    ):
        """
        构造元网络输入特征。

        输入：
            client_losses:
                list 或 tensor
                shape: [num_clients]

            client_expert_freqs:
                list 或 tensor
                shape: [num_clients, num_experts]

            client_num_samples:
                list 或 tensor
                shape: [num_clients]

        输出：
            features:
                shape: [num_experts, num_clients, 3]

        其中每个 feature 是：
            [loss_z, sample_ratio, expert_activation_frequency]
        """

        # ----------------------------------------------------
        # 1. client loss -> loss_z
        # ----------------------------------------------------
        losses = torch.as_tensor(
            client_losses,
            dtype=torch.float32,
            device=self.device,
        )

        loss_mean = losses.mean()
        loss_std = losses.std(unbiased=False).clamp_min(1e-6)
        loss_z = (losses - loss_mean) / loss_std

        # ----------------------------------------------------
        # 2. expert activation frequency
        # ----------------------------------------------------
        expert_freqs = torch.as_tensor(
            client_expert_freqs,
            dtype=torch.float32,
            device=self.device,
        )

        if expert_freqs.dim() != 2:
            raise ValueError(
                "client_expert_freqs 应该是二维，shape=[num_clients, num_experts]"
            )

        num_clients, num_experts = expert_freqs.shape

        if num_experts != self.num_experts:
            raise ValueError(
                f"expert 数量不一致: 当前 aggregator num_experts={self.num_experts}, "
                f"但是输入 expert_freqs.shape={expert_freqs.shape}"
            )

        # ----------------------------------------------------
        # 3. sample ratio
        # ----------------------------------------------------
        sample_counts = torch.as_tensor(
            client_num_samples,
            dtype=torch.float32,
            device=self.device,
        )

        if sample_counts.numel() != num_clients:
            raise ValueError(
                f"client_num_samples 数量不一致: "
                f"num_samples={sample_counts.numel()}, num_clients={num_clients}"
            )

        sample_ratio = sample_counts / sample_counts.sum().clamp_min(1e-6)

        # ----------------------------------------------------
        # 4. 拼接成 [E, C, 3]
        # ----------------------------------------------------
        loss_feature = loss_z.unsqueeze(0).expand(num_experts, num_clients)
        sample_feature = sample_ratio.unsqueeze(0).expand(num_experts, num_clients)
        freq_feature = expert_freqs.transpose(0, 1)

        features = torch.stack(
            [
                loss_feature,
                sample_feature,
                freq_feature,
            ],
            dim=-1,
        )

        return features

    # --------------------------------------------------------
    # 5.2 把元网络输入写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_inputs(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
    ):
        """
        记录元网络输入特征到当前实验 train.log。
        """

        os.makedirs(
            os.path.dirname(self.train_log_path),
            exist_ok=True,
        )

        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
        )

        features = features.detach().cpu()

        raw_losses = torch.as_tensor(
            client_losses,
            dtype=torch.float32,
        ).detach().cpu()

        sample_counts = torch.as_tensor(
            client_num_samples,
            dtype=torch.float32,
        ).detach().cpu()

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(f"[META_INPUT_BEGIN] round={self.round_id}\n")

            num_experts, num_clients, _ = features.shape

            for expert_id in range(num_experts):
                for client_id in range(num_clients):
                    loss_z = features[expert_id, client_id, 0].item()
                    sample_ratio = features[expert_id, client_id, 1].item()
                    expert_freq = features[expert_id, client_id, 2].item()
                    raw_loss = raw_losses[client_id].item()
                    num_samples = sample_counts[client_id].item()

                    f.write(
                        f"[META_INPUT] "
                        f"round={self.round_id} "
                        f"expert={expert_id} "
                        f"client={client_id} "
                        f"loss_z={loss_z:.8f} "
                        f"sample_ratio={sample_ratio:.8f} "
                        f"expert_freq={expert_freq:.8f} "
                        f"raw_client_loss={raw_loss:.8f} "
                        f"client_num_samples={num_samples:.0f}\n"
                    )

            f.write(f"[META_INPUT_END] round={self.round_id}\n")

    # --------------------------------------------------------
    # 5.3 把最终聚合权重 alpha 写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_alpha(
        self,
        alpha,
        client_num_samples,
    ):
        """
        记录元网络最终输出的 expert 聚合权重到当前实验 train.log。
        """

        os.makedirs(
            os.path.dirname(self.train_log_path),
            exist_ok=True,
        )

        alpha = alpha.detach().cpu()

        sample_weighted = get_basic_weights(
            method="sample_weighted",
            client_num_samples=client_num_samples,
        )

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(f"[META_ALPHA_BEGIN] round={self.round_id}\n")

            num_experts, num_clients = alpha.shape

            for expert_id in range(num_experts):
                for client_id in range(num_clients):
                    alpha_value = alpha[expert_id, client_id].item()
                    sample_weight = float(sample_weighted[client_id])

                    f.write(
                        f"[META_ALPHA] "
                        f"round={self.round_id} "
                        f"expert={expert_id} "
                        f"client={client_id} "
                        f"alpha={alpha_value:.8f} "
                        f"sample_weighted_weight={sample_weight:.8f}\n"
                    )

            f.write(f"[META_ALPHA_END] round={self.round_id}\n")

    # --------------------------------------------------------
    # 5.4 元网络输出 alpha
    # --------------------------------------------------------
    def compute_alpha(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
    ):
        """
        用元网络计算 expert 聚合权重 alpha。

        输出：
            alpha:
                shape: [num_experts, num_clients]
        """

        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
        )

        num_experts, num_clients, input_dim = features.shape

        flat_features = features.reshape(num_experts * num_clients, input_dim)
        flat_scores = self.meta_net(flat_features)
        scores = flat_scores.reshape(num_experts, num_clients)

        alpha = F.softmax(scores, dim=1)

        return alpha

    # --------------------------------------------------------
    # 5.5 根据 alpha 构造聚合后的 state_dict
    # --------------------------------------------------------
    def build_aggregated_state_dict(
        self,
        client_state_dicts,
        client_num_samples,
        non_expert_agg,
        alpha,
        device,
    ):
        """
        构造完整的聚合 state_dict。

        non-expert 参数：
            使用 uniform 或 sample_weighted。

        expert 参数：
            使用元网络输出的 alpha。
        """

        if not isinstance(device, torch.device):
            device = torch.device(device)

        alpha = alpha.to(device)

        non_expert_weights = get_basic_weights(
            method=non_expert_agg,
            client_num_samples=client_num_samples,
        )

        new_state_dict = {}
        state_keys = client_state_dicts[0].keys()

        for name in state_keys:
            first_tensor = client_state_dicts[0][name]

            if not torch.is_floating_point(first_tensor):
                new_state_dict[name] = first_tensor.to(device).clone()
                continue

            # ------------------------------------------------
            # expert 参数：使用元网络 alpha 聚合
            # ------------------------------------------------
            if is_expert_param(name):
                expert_id = get_expert_id_from_name(name)

                if expert_id is None:
                    raise ValueError(f"参数名包含 experts 但解析不出 expert id: {name}")

                aggregated_tensor = torch.zeros_like(
                    first_tensor,
                    device=device,
                    dtype=first_tensor.dtype,
                )

                for client_id, client_state in enumerate(client_state_dicts):
                    weight = alpha[expert_id, client_id]
                    tensor = client_state[name].to(device)

                    aggregated_tensor = aggregated_tensor + weight * tensor

                new_state_dict[name] = aggregated_tensor

            # ------------------------------------------------
            # non-expert 参数：使用普通聚合
            # ------------------------------------------------
            else:
                aggregated_tensor = torch.zeros_like(
                    first_tensor,
                    device=device,
                    dtype=first_tensor.dtype,
                )

                for client_id, client_state in enumerate(client_state_dicts):
                    weight = float(non_expert_weights[client_id])
                    tensor = client_state[name].to(device)

                    aggregated_tensor = aggregated_tensor + weight * tensor

                new_state_dict[name] = aggregated_tensor

        return new_state_dict

    # --------------------------------------------------------
    # 5.6 在 server validation set 上计算 CE loss
    # --------------------------------------------------------
    def compute_validation_loss(self, model, temp_state_dict, val_loader):
        """
        用临时聚合参数在 server validation set 上计算 CE loss。

        关键点：
            这里用 functional_call，不用 load_state_dict。
        """

        model.eval()

        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0

        for batch_id, (images, labels) in enumerate(val_loader):
            if batch_id >= self.max_val_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = functional_call(
                model,
                temp_state_dict,
                (images,),
            )

            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss = total_loss + loss * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("server validation loader 为空，无法计算 meta loss")

        avg_loss = total_loss / total_samples

        return avg_loss

    # --------------------------------------------------------
    # 5.7 主函数：更新元网络并返回最终聚合模型
    # --------------------------------------------------------
    def aggregate(
        self,
        model,
        client_state_dicts,
        client_num_samples,
        client_losses,
        client_expert_freqs,
        val_loader,
        non_expert_agg="sample_weighted",
    ):
        """
        元网络专家聚合主入口。
        """

        model.to(self.device)

        self.round_id += 1

        self.log_meta_inputs(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
        )

        meta_loss_value = None

        # ----------------------------------------------------
        # 第一步：用验证集 CE loss 更新元网络
        # ----------------------------------------------------
        for _ in range(self.meta_steps):
            self.optimizer.zero_grad()

            alpha = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
            )

            temp_state_dict = self.build_aggregated_state_dict(
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                non_expert_agg=non_expert_agg,
                alpha=alpha,
                device=self.device,
            )

            meta_loss = self.compute_validation_loss(
                model=model,
                temp_state_dict=temp_state_dict,
                val_loader=val_loader,
            )

            meta_loss.backward()
            self.optimizer.step()

            meta_loss_value = meta_loss.item()

        # ----------------------------------------------------
        # 第二步：用更新后的元网络输出最终 alpha
        # ----------------------------------------------------
        with torch.no_grad():
            final_alpha = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
            )

            self.log_meta_alpha(
                alpha=final_alpha,
                client_num_samples=client_num_samples,
            )

            final_state_dict = self.build_aggregated_state_dict(
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                non_expert_agg=non_expert_agg,
                alpha=final_alpha,
                device=torch.device("cpu"),
            )

        info = {
            "meta_loss": meta_loss_value,
            "alpha": final_alpha.detach().cpu(),
        }

        return final_state_dict, info