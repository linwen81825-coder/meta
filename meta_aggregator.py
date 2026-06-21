# meta_aggregator.py
# ------------------------------------------------------------
# 元网络专家聚合模块
#
# 当前支持通过 config.yaml 控制元网络输入特征：
#
# meta:
#   input_features:
#     - loss_z
#     - sample_ratio
#     - expert_freq
#     - delta_norm_z
#
# 当前只支持四个输入特征：
#   loss_z:
#       标准化客户端训练 loss。
#
#   sample_ratio:
#       当前客户端样本数占比。
#
#   expert_freq:
#       当前客户端上当前 expert 的激活频率。
#
#   delta_norm_z:
#       当前客户端当前 expert 参数更新幅度的标准化值。
#       计算方式：
#           delta_norm[i, e] = || theta_i,e - theta_global,e ||
#       然后对每个 expert，在 client 维度做 z-score。
#
# 注意：
#   这四个特征全部按配置判断。
#   配置里有哪个，就计算哪个；
#   配置里没有，就不计算。
#
# 日志：
#   元网络输入、最终聚合权重、元网络诊断信息都会追加写入当前实验的 train.log。
#   日志路径由 train.py 传进来，不能写死。
#
# 重要：
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
    # bincount 需要一维，所以这里统一拉平。
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

    输入维度由 config.yaml 里的 meta.input_features 决定。

    例如：
        input_features = [loss_z, sample_ratio, expert_freq]
        input_dim = 3

        input_features = [loss_z, sample_ratio, expert_freq, delta_norm_z]
        input_dim = 4
    """

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("MetaWeightNet 的 input_dim 必须大于 0")

        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        x:
            shape: [N, input_dim]

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
        input_features=None,
    ):
        self.num_experts = num_experts
        self.device = device
        self.meta_steps = meta_steps
        self.max_val_batches = max_val_batches

        # ----------------------------------------------------
        # 元网络输入特征配置
        # ----------------------------------------------------
        if input_features is None:
            input_features = [
                "loss_z",
                "sample_ratio",
                "expert_freq",
            ]

        allowed_features = {
            "loss_z",
            "sample_ratio",
            "expert_freq",
            "delta_norm_z",
        }

        if not isinstance(input_features, (list, tuple)):
            raise TypeError(
                "meta.input_features 必须是 list，例如："
                "['loss_z', 'sample_ratio', 'expert_freq']"
            )

        input_features = list(input_features)

        if len(input_features) == 0:
            raise ValueError("meta.input_features 不能为空")

        unknown_features = [
            name for name in input_features
            if name not in allowed_features
        ]

        if len(unknown_features) > 0:
            raise ValueError(
                f"未知 meta input feature: {unknown_features}. "
                f"当前只支持: {sorted(allowed_features)}"
            )

        # 保存元网络实际输入特征名，train.py 会打印到日志。
        self.input_feature_names = input_features

        # 根据输入特征数量决定元网络 input_dim。
        self.meta_net = MetaWeightNet(
            input_dim=len(self.input_feature_names),
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
    # 5.1 基础检查：客户端数量
    # --------------------------------------------------------
    def get_num_clients(self, client_num_samples):
        """
        根据 client_num_samples 得到本轮参与聚合的客户端数量。
        """

        num_clients = len(client_num_samples)

        if num_clients <= 0:
            raise ValueError("本轮客户端数量为空")

        return num_clients

    # --------------------------------------------------------
    # 5.2 特征：loss_z
    # --------------------------------------------------------
    def build_loss_z_feature(
        self,
        client_losses,
        num_clients,
    ):
        """
        构造 loss_z 特征。

        输入：
            client_losses: [C]

        输出：
            loss_z_feature: [E, C]
        """

        losses = torch.as_tensor(
            client_losses,
            dtype=torch.float32,
            device=self.device,
        )

        if losses.numel() != num_clients:
            raise ValueError(
                f"client_losses 数量不一致: "
                f"losses={losses.numel()}, num_clients={num_clients}"
            )

        loss_mean = losses.mean()
        loss_std = losses.std(unbiased=False).clamp_min(1e-6)
        loss_z = (losses - loss_mean) / loss_std

        loss_z_feature = loss_z.unsqueeze(0).expand(
            self.num_experts,
            num_clients,
        )

        return loss_z_feature

    # --------------------------------------------------------
    # 5.3 特征：sample_ratio
    # --------------------------------------------------------
    def build_sample_ratio_feature(
        self,
        client_num_samples,
        num_clients,
    ):
        """
        构造 sample_ratio 特征。

        输入：
            client_num_samples: [C]

        输出：
            sample_ratio_feature: [E, C]
        """

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

        sample_ratio_feature = sample_ratio.unsqueeze(0).expand(
            self.num_experts,
            num_clients,
        )

        return sample_ratio_feature

    # --------------------------------------------------------
    # 5.4 特征：expert_freq
    # --------------------------------------------------------
    def build_expert_freq_feature(
        self,
        client_expert_freqs,
        num_clients,
    ):
        """
        构造 expert_freq 特征。

        输入：
            client_expert_freqs: [C, E]

        输出：
            expert_freq_feature: [E, C]
        """

        expert_freqs = torch.as_tensor(
            client_expert_freqs,
            dtype=torch.float32,
            device=self.device,
        )

        if expert_freqs.dim() != 2:
            raise ValueError(
                "client_expert_freqs 应该是二维，shape=[num_clients, num_experts]"
            )

        input_num_clients, input_num_experts = expert_freqs.shape

        if input_num_clients != num_clients:
            raise ValueError(
                f"client_expert_freqs 客户端数量不一致: "
                f"freq_clients={input_num_clients}, num_clients={num_clients}"
            )

        if input_num_experts != self.num_experts:
            raise ValueError(
                f"expert 数量不一致: 当前 aggregator num_experts={self.num_experts}, "
                f"但是输入 expert_freqs.shape={expert_freqs.shape}"
            )

        expert_freq_feature = expert_freqs.transpose(0, 1)

        return expert_freq_feature

    # --------------------------------------------------------
    # 5.5 计算 expert delta norm
    # --------------------------------------------------------
    def compute_expert_delta_norms(
        self,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        """
        计算每个 client-expert pair 的参数更新幅度。

        输入：
            client_state_dicts:
                本轮每个客户端训练后的 state_dict。

            global_state_dict:
                本轮客户端训练前的全局模型 state_dict。

        输出：
            delta_norms:
                shape: [num_experts, num_clients]

        计算：
            delta_norm[e, i] =
                || theta_{i,e} - theta_{global,e} ||_2

        注意：
            这里只统计 expert 参数。
            非 expert 参数不参与 delta_norm_z。
        """

        if client_state_dicts is None:
            raise ValueError(
                "使用 delta_norm_z 时，必须传入 client_state_dicts。"
            )

        if global_state_dict is None:
            raise ValueError(
                "使用 delta_norm_z 时，必须传入 global_state_dict。"
            )

        if len(client_state_dicts) != num_clients:
            raise ValueError(
                f"client_state_dicts 数量不一致: "
                f"state_dicts={len(client_state_dicts)}, num_clients={num_clients}"
            )

        # 用 float64 累积平方和，数值更稳。
        delta_sq = torch.zeros(
            self.num_experts,
            num_clients,
            dtype=torch.float64,
        )

        for client_id, client_state in enumerate(client_state_dicts):
            for name, client_tensor in client_state.items():
                if not is_expert_param(name):
                    continue

                expert_id = get_expert_id_from_name(name)

                if expert_id is None:
                    continue

                if not torch.is_floating_point(client_tensor):
                    continue

                if name not in global_state_dict:
                    raise KeyError(f"global_state_dict 缺少参数: {name}")

                global_tensor = global_state_dict[name]

                # 这里放在 CPU 上算，避免额外占 GPU 显存。
                client_tensor = client_tensor.detach().cpu().float()
                global_tensor = global_tensor.detach().cpu().float()

                diff = client_tensor - global_tensor

                delta_sq[expert_id, client_id] += float(
                    torch.sum(diff * diff).item()
                )

        delta_norms = torch.sqrt(delta_sq).float().to(self.device)

        return delta_norms

    # --------------------------------------------------------
    # 5.6 特征：delta_norm_z
    # --------------------------------------------------------
    def build_delta_norm_z_feature(
        self,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        """
        构造 delta_norm_z 特征。

        输入：
            client_state_dicts:
                本轮每个客户端训练后的参数。

            global_state_dict:
                本轮客户端训练前的全局参数。

        输出：
            delta_norm_z: [E, C]

        计算：
            1. 先算每个 client-expert 的 expert 参数更新幅度。
            2. 对每个 expert，在 client 维度做 z-score。
        """

        delta_norm = self.compute_expert_delta_norms(
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
            num_clients=num_clients,
        )
        # delta_norm: [E, C]

        delta_mean = delta_norm.mean(
            dim=1,
            keepdim=True,
        )

        delta_std = delta_norm.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-6)

        delta_norm_z = (delta_norm - delta_mean) / delta_std

        return delta_norm_z

    # --------------------------------------------------------
    # 5.7 按配置构造元网络输入特征
    # --------------------------------------------------------
    def build_meta_features(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_state_dicts=None,
        global_state_dict=None,
    ):
        """
        构造元网络输入特征。

        当前支持：
            loss_z
            sample_ratio
            expert_freq
            delta_norm_z

        重点：
            四个特征全部按 self.input_feature_names 判断。
            配置里写了哪个，就计算哪个；
            配置里没写，就不计算。

        输出：
            features:
                shape: [num_experts, num_clients, input_dim]
        """

        num_clients = self.get_num_clients(
            client_num_samples=client_num_samples,
        )

        selected_features = []

        for feature_name in self.input_feature_names:
            if feature_name == "loss_z":
                feature = self.build_loss_z_feature(
                    client_losses=client_losses,
                    num_clients=num_clients,
                )

            elif feature_name == "sample_ratio":
                feature = self.build_sample_ratio_feature(
                    client_num_samples=client_num_samples,
                    num_clients=num_clients,
                )

            elif feature_name == "expert_freq":
                feature = self.build_expert_freq_feature(
                    client_expert_freqs=client_expert_freqs,
                    num_clients=num_clients,
                )

            elif feature_name == "delta_norm_z":
                feature = self.build_delta_norm_z_feature(
                    client_state_dicts=client_state_dicts,
                    global_state_dict=global_state_dict,
                    num_clients=num_clients,
                )

            else:
                # 正常情况下不会走到这里，因为 __init__ 已经校验过。
                raise ValueError(f"未知 meta input feature: {feature_name}")

            selected_features.append(feature)

        # features: [E, C, input_dim]
        features = torch.stack(
            selected_features,
            dim=-1,
        )

        return features

    # --------------------------------------------------------
    # 5.8 计算 alpha 诊断信息
    # --------------------------------------------------------
    def compute_alpha_stats(self, scores, alpha):
        """
        根据 softmax 前 scores 和 softmax 后 alpha 计算诊断指标。

        score_std:
            softmax 前 score 的整体标准差。

        alpha_entropy:
            alpha 的归一化熵。
            每个 expert 单独计算熵，再对 expert 求平均。
            越接近 1，说明越接近 uniform。
        """

        with torch.no_grad():
            scores_detached = scores.detach().float()
            alpha_detached = alpha.detach().float()

            num_clients = alpha_detached.size(1)

            score_std = scores_detached.std(unbiased=False).item()
            score_abs_mean = scores_detached.abs().mean().item()

            eps = 1e-12
            entropy = -torch.sum(
                alpha_detached * torch.log(alpha_detached.clamp_min(eps)),
                dim=1,
            )

            if num_clients > 1:
                entropy = entropy / np.log(num_clients)

            alpha_entropy = entropy.mean().item()

            alpha_std = alpha_detached.std(unbiased=False).item()
            alpha_max_mean = alpha_detached.max(dim=1).values.mean().item()
            alpha_min_mean = alpha_detached.min(dim=1).values.mean().item()

        stats = {
            "score_std": score_std,
            "score_abs_mean": score_abs_mean,
            "alpha_entropy": alpha_entropy,
            "alpha_std": alpha_std,
            "alpha_max_mean": alpha_max_mean,
            "alpha_min_mean": alpha_min_mean,
        }

        return stats

    # --------------------------------------------------------
    # 5.9 计算元网络梯度范数
    # --------------------------------------------------------
    def compute_meta_grad_norm(self):
        """
        计算元网络参数梯度范数。

        返回：
            grad_norm:
                所有元网络参数梯度拼起来后的 L2 norm。

            grad_max_abs:
                梯度绝对值最大值。
        """

        total_sq = 0.0
        grad_max_abs = 0.0

        for param in self.meta_net.parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach().float()

            total_sq += torch.sum(grad * grad).item()

            current_max = grad.abs().max().item()
            grad_max_abs = max(grad_max_abs, current_max)

        grad_norm = total_sq ** 0.5

        return grad_norm, grad_max_abs

    # --------------------------------------------------------
    # 5.10 把元网络输入写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_inputs(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_state_dicts=None,
        global_state_dict=None,
    ):
        """
        记录元网络输入特征到当前实验 train.log。

        会按照 self.input_feature_names 自动记录当前元网络实际输入。
        """

        os.makedirs(
            os.path.dirname(self.train_log_path),
            exist_ok=True,
        )

        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
        )

        features = features.detach().cpu()

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(f"[META_INPUT_BEGIN] round={self.round_id}\n")

            num_experts, num_clients, input_dim = features.shape

            for expert_id in range(num_experts):
                for client_id in range(num_clients):
                    feature_texts = []

                    for feature_idx, feature_name in enumerate(self.input_feature_names):
                        value = features[expert_id, client_id, feature_idx].item()
                        feature_texts.append(f"{feature_name}={value:.8f}")

                    feature_text = " ".join(feature_texts)

                    f.write(
                        f"[META_INPUT] "
                        f"round={self.round_id} "
                        f"expert={expert_id} "
                        f"client={client_id} "
                        f"{feature_text}\n"
                    )

            f.write(f"[META_INPUT_END] round={self.round_id}\n")

    # --------------------------------------------------------
    # 5.11 把最终聚合权重 alpha 写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_alpha(
        self,
        alpha,
        client_num_samples,
    ):
        """
        记录元网络最终输出的 expert 聚合权重到当前实验 train.log。

        这里仍然记录 sample_weighted_weight，
        只是为了方便你对比 meta alpha 和普通样本加权的差异。
        它不代表 sample_ratio 一定作为元网络输入。
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
    # 5.12 记录 meta 训练诊断信息
    # --------------------------------------------------------
    def log_meta_diagnostics(
        self,
        step_id,
        meta_loss_value,
        alpha_stats,
        meta_grad_norm,
        meta_grad_max_abs,
    ):
        """
        记录每个 meta step 的诊断信息。
        """

        os.makedirs(
            os.path.dirname(self.train_log_path),
            exist_ok=True,
        )

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[META_DIAG] "
                f"round={self.round_id} "
                f"step={step_id} "
                f"meta_loss={meta_loss_value:.8f} "
                f"score_std={alpha_stats['score_std']:.8f} "
                f"score_abs_mean={alpha_stats['score_abs_mean']:.8f} "
                f"alpha_entropy={alpha_stats['alpha_entropy']:.8f} "
                f"alpha_std={alpha_stats['alpha_std']:.8f} "
                f"alpha_max_mean={alpha_stats['alpha_max_mean']:.8f} "
                f"alpha_min_mean={alpha_stats['alpha_min_mean']:.8f} "
                f"meta_grad_norm={meta_grad_norm:.8f} "
                f"meta_grad_max_abs={meta_grad_max_abs:.8f}\n"
            )

    # --------------------------------------------------------
    # 5.13 记录最终 alpha 的诊断信息
    # --------------------------------------------------------
    def log_meta_final_diagnostics(self, alpha_stats):
        """
        记录元网络更新完成后，最终 alpha 的诊断信息。
        """

        os.makedirs(
            os.path.dirname(self.train_log_path),
            exist_ok=True,
        )

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[META_DIAG_FINAL] "
                f"round={self.round_id} "
                f"score_std={alpha_stats['score_std']:.8f} "
                f"score_abs_mean={alpha_stats['score_abs_mean']:.8f} "
                f"alpha_entropy={alpha_stats['alpha_entropy']:.8f} "
                f"alpha_std={alpha_stats['alpha_std']:.8f} "
                f"alpha_max_mean={alpha_stats['alpha_max_mean']:.8f} "
                f"alpha_min_mean={alpha_stats['alpha_min_mean']:.8f}\n"
            )

    # --------------------------------------------------------
    # 5.14 元网络输出 alpha
    # --------------------------------------------------------
    def compute_alpha(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_state_dicts=None,
        global_state_dict=None,
        return_stats=False,
    ):
        """
        用元网络计算 expert 聚合权重 alpha。

        输出：
            alpha:
                shape: [num_experts, num_clients]

        如果 return_stats=True：
            返回：
                alpha, stats
        """

        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
        )

        num_experts, num_clients, input_dim = features.shape

        flat_features = features.reshape(num_experts * num_clients, input_dim)

        # [E*C]
        flat_scores = self.meta_net(flat_features)

        # [E, C]
        scores = flat_scores.reshape(num_experts, num_clients)

        # 对每个 expert，在 client 维度做 softmax。
        alpha = F.softmax(scores, dim=1)

        if return_stats:
            stats = self.compute_alpha_stats(
                scores=scores,
                alpha=alpha,
            )

            return alpha, stats

        return alpha

    # --------------------------------------------------------
    # 5.15 根据 alpha 构造聚合后的 state_dict
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
    # 5.16 在 server validation set 上计算 CE loss
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
    # 5.17 主函数：更新元网络并返回最终聚合模型
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

        # 当前轮客户端训练前的全局模型参数。
        # delta_norm_z 需要用它和 client_state_dicts 做差。
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        self.round_id += 1

        self.log_meta_inputs(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
        )

        meta_loss_value = None

        # ----------------------------------------------------
        # 第一步：用验证集 CE loss 更新元网络
        # ----------------------------------------------------
        for step_id in range(1, self.meta_steps + 1):
            self.optimizer.zero_grad()

            # 1. 元网络输出 alpha，alpha 带计算图。
            alpha, alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                return_stats=True,
            )

            # 2. 用 alpha 构造临时聚合模型参数。
            temp_state_dict = self.build_aggregated_state_dict(
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                non_expert_agg=non_expert_agg,
                alpha=alpha,
                device=self.device,
            )

            # 3. 用临时模型在 server validation set 上算 CE loss。
            meta_loss = self.compute_validation_loss(
                model=model,
                temp_state_dict=temp_state_dict,
                val_loader=val_loader,
            )

            # 4. 反向传播。
            meta_loss.backward()

            # 5. 记录元网络梯度范数。
            meta_grad_norm, meta_grad_max_abs = self.compute_meta_grad_norm()

            meta_loss_value = meta_loss.item()

            self.log_meta_diagnostics(
                step_id=step_id,
                meta_loss_value=meta_loss_value,
                alpha_stats=alpha_stats,
                meta_grad_norm=meta_grad_norm,
                meta_grad_max_abs=meta_grad_max_abs,
            )

            # 6. 更新元网络。
            self.optimizer.step()

        # ----------------------------------------------------
        # 第二步：用更新后的元网络输出最终 alpha
        # ----------------------------------------------------
        with torch.no_grad():
            final_alpha, final_alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                return_stats=True,
            )

            self.log_meta_final_diagnostics(
                alpha_stats=final_alpha_stats,
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
            "score_std": final_alpha_stats["score_std"],
            "alpha_entropy": final_alpha_stats["alpha_entropy"],
        }

        return final_state_dict, info