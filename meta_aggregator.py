# meta_aggregator.py
# ------------------------------------------------------------
# 元网络专家聚合模块
#
# 这个文件只做一件事：
#   用服务器验证集训练一个元网络，让它输出 expert 聚合权重。
#
# 算法流程：
#   客户端统计特征 [client_loss_z, expert_activation_frequency]
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
#   元网络输入、最终聚合权重、元网络诊断信息都会追加写入当前实验的 train.log。
#   日志路径由 train.py 传进来，不能写死。
#
# 当前版本：
#   已删除 sample_ratio 作为元网络输入。
#
#   元网络输入变为：
#       [loss_z, expert_freq]
#
#   其中：
#       loss_z:
#           当前客户端训练 loss 在本轮客户端中的相对高低。
#
#       expert_freq:
#           当前客户端上当前 expert 的激活频率。
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

    对每一个 client-expert pair 输入两个统计特征：

        [client_loss_z, expert_activation_frequency]

    输出一个 score。

    然后对同一个 expert 下的所有客户端 score 做 softmax，
    得到这个 expert 的客户端聚合权重 alpha。

    输入维度：
        2

    输出维度：
        1
    """

    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        x:
            shape: [N, 2]

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

        # 现在元网络输入是 2 维：
        # [loss_z, expert_freq]
        self.meta_net = MetaWeightNet(
            input_dim=2,
            hidden_dim=hidden_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr=lr,
        )

        # 当前是第几轮聚合。
        # 只用于写日志，不影响训练。
        self.round_id = 0
        # 当前元网络实际使用的输入特征。
        # 注意：
        #   这里要和 build_meta_features() 里 stack 的顺序保持一致。
        #   当前版本已经删除 sample_ratio，所以只剩：
        #       [loss_z, expert_freq]
        self.input_feature_names = [
            "loss_z",
            "expert_freq",
        ]

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

                注意：
                    当前版本不再把 sample_ratio 作为元网络输入。
                    这里保留 client_num_samples 参数，是为了兼容 aggregate()
                    的调用接口，避免 train.py 也跟着改。

        输出：
            features:
                shape: [num_experts, num_clients, 2]

        其中每个 feature 是：
            [loss_z, expert_activation_frequency]
        """

        # 保留这个变量只是为了说明：
        # 当前版本 client_num_samples 不参与元网络输入。
        _ = client_num_samples

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
        # 3. 拼接成 [E, C, 2]
        # ----------------------------------------------------

        # loss_z: [C] -> [E, C]
        loss_feature = loss_z.unsqueeze(0).expand(num_experts, num_clients)

        # expert_freqs: [C, E] -> [E, C]
        freq_feature = expert_freqs.transpose(0, 1)

        # features: [E, C, 2]
        features = torch.stack(
            [
                loss_feature,
                freq_feature,
            ],
            dim=-1,
        )

        return features

    # --------------------------------------------------------
    # 5.2 计算 alpha 诊断信息
    # --------------------------------------------------------
    def compute_alpha_stats(self, scores, alpha):
        """
        根据 softmax 前 scores 和 softmax 后 alpha 计算诊断指标。

        score_std:
            softmax 前 score 的整体标准差。
            如果很小，说明 score 没拉开，alpha 很容易接近 uniform。

        alpha_entropy:
            alpha 的归一化熵。
            每个 expert 单独计算熵，再对 expert 求平均。
            越接近 1，说明越接近 uniform。

        alpha_std:
            alpha 的整体标准差。
            越小越接近 uniform。

        alpha_max_mean:
            每个 expert 最大客户端权重的平均值。
            如果接近 1 / num_clients，说明没选出明显重要客户端。
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
    # 5.3 计算元网络梯度范数
    # --------------------------------------------------------
    def compute_meta_grad_norm(self):
        """
        计算元网络参数梯度范数。

        返回：
            grad_norm:
                所有元网络参数梯度拼起来后的 L2 norm。

            grad_max_abs:
                梯度绝对值最大值。

        用途：
            判断 validation loss 是否真的在推动元网络。
            如果 grad_norm 长期接近 0，说明 meta loss 对元网络几乎没有有效梯度。
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
    # 5.4 把元网络输入写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_inputs(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
    ):
        """
        记录元网络输入特征到当前实验 train.log。

        当前版本记录：
            loss_z
            expert_freq
            raw_client_loss
            client_num_samples

        注意：
            sample_ratio 已经不再作为元网络输入，所以这里不再记录 sample_ratio。
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
                    expert_freq = features[expert_id, client_id, 1].item()
                    raw_loss = raw_losses[client_id].item()
                    num_samples = sample_counts[client_id].item()

                    f.write(
                        f"[META_INPUT] "
                        f"round={self.round_id} "
                        f"expert={expert_id} "
                        f"client={client_id} "
                        f"loss_z={loss_z:.8f} "
                        f"expert_freq={expert_freq:.8f} "
                        f"raw_client_loss={raw_loss:.8f} "
                        f"client_num_samples={num_samples:.0f}\n"
                    )

            f.write(f"[META_INPUT_END] round={self.round_id}\n")

    # --------------------------------------------------------
    # 5.5 把最终聚合权重 alpha 写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_alpha(
        self,
        alpha,
        client_num_samples,
    ):
        """
        记录元网络最终输出的 expert 聚合权重到当前实验 train.log。

        注意：
            虽然 sample_ratio 不再作为元网络输入，
            但是这里仍然记录 sample_weighted_weight，
            方便你对比 meta alpha 是否还会偏向大客户端。
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
    # 5.6 记录 meta 训练诊断信息
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

        日志字段：
            score_std:
                softmax 前 score 标准差。

            alpha_entropy:
                alpha 归一化熵，越接近 1 越 uniform。

            meta_grad_norm:
                元网络梯度 L2 norm。

            meta_grad_max_abs:
                元网络最大梯度绝对值。
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
    # 5.7 记录最终 alpha 的诊断信息
    # --------------------------------------------------------
    def log_meta_final_diagnostics(self, alpha_stats):
        """
        记录元网络更新完成后，最终 alpha 的诊断信息。

        注意：
            这里没有 grad_norm，因为最终 alpha 是 no_grad 下重新计算的。
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
    # 5.8 元网络输出 alpha
    # --------------------------------------------------------
    def compute_alpha(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
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

            其中 stats 包含：
                score_std
                alpha_entropy
                alpha_std
                alpha_max_mean
                alpha_min_mean
        """

        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
        )

        num_experts, num_clients, input_dim = features.shape

        flat_features = features.reshape(num_experts * num_clients, input_dim)

        # [E*C]
        flat_scores = self.meta_net(flat_features)

        # [E, C]
        scores = flat_scores.reshape(num_experts, num_clients)

        # 对每个 expert，在 client 维度做 softmax
        alpha = F.softmax(scores, dim=1)

        if return_stats:
            stats = self.compute_alpha_stats(
                scores=scores,
                alpha=alpha,
            )

            return alpha, stats

        return alpha

    # --------------------------------------------------------
    # 5.9 根据 alpha 构造聚合后的 state_dict
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
    # 5.10 在 server validation set 上计算 CE loss
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
    # 5.11 主函数：更新元网络并返回最终聚合模型
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
        for step_id in range(1, self.meta_steps + 1):
            self.optimizer.zero_grad()

            # 1. 元网络输出 alpha，alpha 带计算图
            alpha, alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                return_stats=True,
            )

            # 2. 用 alpha 构造临时聚合模型参数
            temp_state_dict = self.build_aggregated_state_dict(
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                non_expert_agg=non_expert_agg,
                alpha=alpha,
                device=self.device,
            )

            # 3. 用临时模型在 server validation set 上算 CE loss
            meta_loss = self.compute_validation_loss(
                model=model,
                temp_state_dict=temp_state_dict,
                val_loader=val_loader,
            )

            # 4. 反向传播
            meta_loss.backward()

            # 5. 记录元网络梯度范数
            meta_grad_norm, meta_grad_max_abs = self.compute_meta_grad_norm()

            meta_loss_value = meta_loss.item()

            self.log_meta_diagnostics(
                step_id=step_id,
                meta_loss_value=meta_loss_value,
                alpha_stats=alpha_stats,
                meta_grad_norm=meta_grad_norm,
                meta_grad_max_abs=meta_grad_max_abs,
            )

            # 6. 更新元网络
            self.optimizer.step()

        # ----------------------------------------------------
        # 第二步：用更新后的元网络输出最终 alpha
        # ----------------------------------------------------
        with torch.no_grad():
            final_alpha, final_alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
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