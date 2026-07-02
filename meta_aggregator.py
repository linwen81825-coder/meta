# meta_aggregator.py
# ------------------------------------------------------------
# 元网络专家聚合模块
#
# 新增：
# 1. 支持 gradient alignment supervised meta loss:
#       meta_loss = - mean_e sum_i alpha[e,i] * alignment_score[e,i]
# 2. alignment_valid_mask 控制每个 expert 只聚合有效客户端
# 3. 有效客户端数 < 2 的 expert 自动 fallback
# 4. 保留原来的 validation CE meta-loss 分支，但不再使用 meta-select / best-step selection
# ------------------------------------------------------------

import copy
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
    match = re.search(r"experts\.(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


# ------------------------------------------------------------
# 2. 普通聚合权重
# ------------------------------------------------------------
def get_basic_weights(method, client_num_samples):
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
    num_experts = expert_counts.numel()
    expert_indices = expert_indices.detach().cpu().reshape(-1)

    batch_counts = torch.bincount(
        expert_indices,
        minlength=num_experts,
    )
    expert_counts += batch_counts


def counts_to_frequency(expert_counts):
    total = expert_counts.sum().item()
    if total <= 0:
        return torch.zeros_like(expert_counts, dtype=torch.float32)
    return expert_counts.float() / total


# ------------------------------------------------------------
# 4. 元网络：DeepSets 风格打分器
# ------------------------------------------------------------
class MetaWeightNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("MetaWeightNet 的 input_dim 必须大于 0")
        if hidden_dim <= 0:
            raise ValueError("MetaWeightNet 的 hidden_dim 必须大于 0")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                "DeepSets 版 MetaWeightNet 需要三维输入："
                "[num_experts, num_clients, input_dim]，"
                f"当前 x.shape={tuple(x.shape)}"
            )

        num_experts, num_clients, input_dim = x.shape

        if input_dim != self.input_dim:
            raise ValueError(
                f"MetaWeightNet 输入维度不一致: "
                f"input_dim={input_dim}, expected={self.input_dim}"
            )

        flat_x = x.reshape(num_experts * num_clients, input_dim)
        flat_h = self.encoder(flat_x)
        h = flat_h.reshape(num_experts, num_clients, self.hidden_dim)

        context = h.mean(dim=1, keepdim=True)
        context = context.expand(-1, num_clients, -1)

        score_input = torch.cat(
            [h, context, h - context],
            dim=-1,
        )

        flat_score_input = score_input.reshape(
            num_experts * num_clients,
            self.hidden_dim * 3,
        )

        flat_scores = self.score_head(flat_score_input).squeeze(-1)
        scores = flat_scores.reshape(num_experts, num_clients)

        return scores


# ------------------------------------------------------------
# 5. Meta Expert Aggregator
# ------------------------------------------------------------
class MetaExpertAggregator:
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
        tau=1.0,
        active_mask=False,
        active_threshold=0.0,
        min_active_clients_per_expert=2,
        use_gradient_alignment=False,
        alignment_fallback="sample_weighted",
    ):
        self.num_experts = num_experts
        self.device = device
        self.meta_steps = meta_steps
        self.max_val_batches = max_val_batches

        if tau <= 0:
            raise ValueError(f"meta.tau 必须大于 0，当前 tau={tau}")
        self.tau = float(tau)

        self.active_mask = bool(active_mask)
        self.active_threshold = float(active_threshold)

        if min_active_clients_per_expert < 1:
            raise ValueError(
                "min_active_clients_per_expert 必须 >= 1，"
                f"当前值为 {min_active_clients_per_expert}"
            )
        self.min_active_clients_per_expert = int(min_active_clients_per_expert)

        self.use_gradient_alignment = bool(use_gradient_alignment)
        self.alignment_fallback = alignment_fallback

        if input_features is None:
            input_features = [
                "loss_z",
                "sample_ratio",
                "expert_freq",
            ]

        allowed_features = {
            "loss_z",
            "loss_raw",
            "sample_ratio",
            "expert_freq",
            "expert_loss_z",
            "expert_loss_raw",
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

        self.input_feature_names = input_features

        self.meta_net = MetaWeightNet(
            input_dim=len(self.input_feature_names),
            hidden_dim=hidden_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr=lr,
        )

        self.round_id = 0

        if train_log_path is None:
            raise ValueError(
                "MetaExpertAggregator 必须传入 train_log_path，不能使用写死日志路径。"
            )

        self.train_log_path = train_log_path

    # --------------------------------------------------------
    # 5.1 特征构造
    # --------------------------------------------------------
    def get_num_clients(self, client_num_samples):
        num_clients = len(client_num_samples)
        if num_clients <= 0:
            raise ValueError("本轮客户端数量为空")
        return num_clients

    def build_loss_z_feature(self, client_losses, num_clients):
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

    def build_loss_raw_feature(self, client_losses, num_clients):
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

        loss_raw_feature = losses.unsqueeze(0).expand(
            self.num_experts,
            num_clients,
        )

        return loss_raw_feature

    def build_sample_ratio_feature(self, client_num_samples, num_clients):
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

    def build_expert_freq_feature(self, client_expert_freqs, num_clients):
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

    def build_expert_loss_z_feature(self, client_expert_losses, num_clients):
        if client_expert_losses is None:
            raise ValueError(
                "使用 expert_loss_z 时，必须从 train.py 传入 client_expert_losses。"
            )

        expert_losses = torch.as_tensor(
            client_expert_losses,
            dtype=torch.float32,
            device=self.device,
        )

        if expert_losses.dim() != 2:
            raise ValueError(
                "client_expert_losses 应该是二维，shape=[num_clients, num_experts]"
            )

        input_num_clients, input_num_experts = expert_losses.shape

        if input_num_clients != num_clients:
            raise ValueError(
                f"client_expert_losses 客户端数量不一致: "
                f"loss_clients={input_num_clients}, num_clients={num_clients}"
            )

        if input_num_experts != self.num_experts:
            raise ValueError(
                f"expert 数量不一致: 当前 aggregator num_experts={self.num_experts}, "
                f"但是输入 expert_losses.shape={expert_losses.shape}"
            )

        expert_loss = expert_losses.transpose(0, 1)

        loss_mean = expert_loss.mean(
            dim=1,
            keepdim=True,
        )

        loss_std = expert_loss.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-6)

        expert_loss_z = (expert_loss - loss_mean) / loss_std
        return expert_loss_z

    def build_expert_loss_raw_feature(self, client_expert_losses, num_clients):
        if client_expert_losses is None:
            raise ValueError(
                "使用 expert_loss_raw 时，必须从 train.py 传入 client_expert_losses。"
            )

        expert_losses = torch.as_tensor(
            client_expert_losses,
            dtype=torch.float32,
            device=self.device,
        )

        if expert_losses.dim() != 2:
            raise ValueError(
                "client_expert_losses 应该是二维，shape=[num_clients, num_experts]"
            )

        input_num_clients, input_num_experts = expert_losses.shape

        if input_num_clients != num_clients:
            raise ValueError(
                f"client_expert_losses 客户端数量不一致: "
                f"loss_clients={input_num_clients}, num_clients={num_clients}"
            )

        if input_num_experts != self.num_experts:
            raise ValueError(
                f"expert 数量不一致: 当前 aggregator num_experts={self.num_experts}, "
                f"但是输入 expert_losses.shape={expert_losses.shape}"
            )

        expert_loss_raw = expert_losses.transpose(0, 1)
        return expert_loss_raw

    def compute_expert_delta_norms(
        self,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        if client_state_dicts is None:
            raise ValueError("使用 delta_norm_z 时，必须传入 client_state_dicts")

        if global_state_dict is None:
            raise ValueError("使用 delta_norm_z 时，必须传入 global_state_dict")

        if len(client_state_dicts) != num_clients:
            raise ValueError(
                f"client_state_dicts 数量不一致: "
                f"state_dicts={len(client_state_dicts)}, num_clients={num_clients}"
            )

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

                client_tensor = client_tensor.detach().cpu().float()
                global_tensor = global_tensor.detach().cpu().float()

                diff = client_tensor - global_tensor
                delta_sq[expert_id, client_id] += float(
                    torch.sum(diff * diff).item()
                )

        delta_norms = torch.sqrt(delta_sq).float().to(self.device)
        return delta_norms

    def build_delta_norm_z_feature(
        self,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        delta_norm = self.compute_expert_delta_norms(
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
            num_clients=num_clients,
        )

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

    def build_meta_features(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_expert_losses=None,
        client_state_dicts=None,
        global_state_dict=None,
    ):
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
            elif feature_name == "loss_raw":
                feature = self.build_loss_raw_feature(
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
            elif feature_name == "expert_loss_z":
                feature = self.build_expert_loss_z_feature(
                    client_expert_losses=client_expert_losses,
                    num_clients=num_clients,
                )
            elif feature_name == "expert_loss_raw":
                feature = self.build_expert_loss_raw_feature(
                    client_expert_losses=client_expert_losses,
                    num_clients=num_clients,
                )
            elif feature_name == "delta_norm_z":
                feature = self.build_delta_norm_z_feature(
                    client_state_dicts=client_state_dicts,
                    global_state_dict=global_state_dict,
                    num_clients=num_clients,
                )
            else:
                raise ValueError(f"未知 meta input feature: {feature_name}")

            selected_features.append(feature)

        features = torch.stack(
            selected_features,
            dim=-1,
        )

        return features

    # --------------------------------------------------------
    # 5.2 mask / alpha / stats
    # --------------------------------------------------------
    def build_active_mask(
        self,
        client_expert_freqs,
        num_clients,
    ):
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
                f"expert 数量不一致: 当前 num_experts={self.num_experts}, "
                f"但是输入 expert_freqs.shape={expert_freqs.shape}"
            )

        expert_freq_feature = expert_freqs.transpose(0, 1)
        active_mask = expert_freq_feature > self.active_threshold

        active_count = active_mask.sum(dim=1, keepdim=True)
        too_few_active = active_count < self.min_active_clients_per_expert

        if too_few_active.any():
            active_mask = torch.where(
                too_few_active,
                torch.ones_like(active_mask, dtype=torch.bool),
                active_mask,
            )

        return active_mask

    def build_fallback_alpha(
        self,
        client_num_samples,
        num_experts,
        num_clients,
        device,
    ):
        fallback_weights = get_basic_weights(
            method=self.alignment_fallback,
            client_num_samples=client_num_samples,
        )

        fallback_weights = torch.as_tensor(
            fallback_weights,
            dtype=torch.float32,
            device=device,
        )

        fallback_alpha = fallback_weights.unsqueeze(0).expand(
            num_experts,
            num_clients,
        )

        return fallback_alpha

    def compute_alpha_stats(self, scores, alpha):
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

    def compute_alpha_from_features(
        self,
        features,
        client_expert_freqs,
        client_num_samples=None,
        return_stats=False,
        alignment_valid_mask=None,
    ):
        num_experts, num_clients, input_dim = features.shape

        if num_experts != self.num_experts:
            raise ValueError(
                f"features 里的 expert 数量不一致: "
                f"features_num_experts={num_experts}, self.num_experts={self.num_experts}"
            )

        if input_dim != len(self.input_feature_names):
            raise ValueError(
                f"features 的 input_dim 不一致: "
                f"features_input_dim={input_dim}, expected={len(self.input_feature_names)}"
            )

        scores = self.meta_net(features)

        if alignment_valid_mask is not None:
            if client_num_samples is None:
                raise ValueError("使用 alignment_valid_mask 时必须传入 client_num_samples")

            valid_mask = torch.as_tensor(
                alignment_valid_mask,
                dtype=torch.bool,
                device=self.device,
            )

            if valid_mask.shape != (num_experts, num_clients):
                raise ValueError(
                    f"alignment_valid_mask shape 不一致: "
                    f"{tuple(valid_mask.shape)} vs {(num_experts, num_clients)}"
                )

            fallback_alpha = self.build_fallback_alpha(
                client_num_samples=client_num_samples,
                num_experts=num_experts,
                num_clients=num_clients,
                device=self.device,
            )

            alpha_rows = []

            for expert_id in range(num_experts):
                valid_count = int(valid_mask[expert_id].sum().item())

                if valid_count >= self.min_active_clients_per_expert:
                    row_scores = scores[expert_id].masked_fill(
                        ~valid_mask[expert_id],
                        -1e9,
                    )
                    row_alpha = F.softmax(row_scores / self.tau, dim=0)
                else:
                    row_alpha = fallback_alpha[expert_id]

                alpha_rows.append(row_alpha)

            alpha = torch.stack(alpha_rows, dim=0)

        else:
            if self.active_mask:
                active_mask = self.build_active_mask(
                    client_expert_freqs=client_expert_freqs,
                    num_clients=num_clients,
                )
                scores_for_softmax = scores.masked_fill(
                    ~active_mask,
                    -1e9,
                )
            else:
                scores_for_softmax = scores

            alpha = F.softmax(scores_for_softmax / self.tau, dim=1)

        if return_stats:
            stats = self.compute_alpha_stats(
                scores=scores,
                alpha=alpha,
            )
            return alpha, stats

        return alpha

    def compute_alpha(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_expert_losses=None,
        client_state_dicts=None,
        global_state_dict=None,
        return_stats=False,
        alignment_valid_mask=None,
    ):
        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_expert_losses=client_expert_losses,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
        )

        return self.compute_alpha_from_features(
            features=features,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            return_stats=return_stats,
            alignment_valid_mask=alignment_valid_mask,
        )

    # --------------------------------------------------------
    # 5.3 聚合模型
    # --------------------------------------------------------
    def build_aggregated_state_dict(
        self,
        client_state_dicts,
        client_num_samples,
        non_expert_agg,
        alpha,
        device,
    ):
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
    # 5.4 loss
    # --------------------------------------------------------
    def compute_validation_loss(self, model, temp_state_dict, val_loader):
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

    @torch.no_grad()
    def compute_validation_loss_no_grad(self, model, temp_state_dict, val_loader):
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
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("selection validation loader 为空，无法选择 best meta step")

        return total_loss / total_samples

    def compute_alignment_meta_loss(
        self,
        alpha,
        alignment_scores,
        alignment_valid_mask,
    ):
        scores = torch.as_tensor(
            alignment_scores,
            dtype=torch.float32,
            device=self.device,
        )

        valid_mask = torch.as_tensor(
            alignment_valid_mask,
            dtype=torch.bool,
            device=self.device,
        )

        if scores.shape != alpha.shape:
            raise ValueError(
                f"alignment_scores shape 不一致: "
                f"{tuple(scores.shape)} vs alpha {tuple(alpha.shape)}"
            )

        if valid_mask.shape != alpha.shape:
            raise ValueError(
                f"alignment_valid_mask shape 不一致: "
                f"{tuple(valid_mask.shape)} vs alpha {tuple(alpha.shape)}"
            )

        loss_terms = []

        for expert_id in range(self.num_experts):
            valid_count = int(valid_mask[expert_id].sum().item())

            if valid_count < self.min_active_clients_per_expert:
                continue

            loss_e = -torch.sum(
                alpha[expert_id][valid_mask[expert_id]]
                * scores[expert_id][valid_mask[expert_id]]
            )

            loss_terms.append(loss_e)

        if len(loss_terms) == 0:
            return None

        meta_loss = torch.stack(loss_terms).mean()
        return meta_loss

    @torch.no_grad()
    def compute_alignment_meta_loss_no_grad(
        self,
        alpha,
        alignment_scores,
        alignment_valid_mask,
    ):
        meta_loss = self.compute_alignment_meta_loss(
            alpha=alpha,
            alignment_scores=alignment_scores,
            alignment_valid_mask=alignment_valid_mask,
        )

        if meta_loss is None:
            return None

        return float(meta_loss.item())

    # --------------------------------------------------------
    # 5.5 日志
    # --------------------------------------------------------
    def compute_meta_grad_norm(self):
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

    def log_meta_inputs(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_expert_losses=None,
        client_state_dicts=None,
        global_state_dict=None,
    ):
        os.makedirs(
            os.path.dirname(self.train_log_path),
            exist_ok=True,
        )

        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_expert_losses=client_expert_losses,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
        )

        features = features.detach().cpu()

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(f"[META_INPUT_BEGIN] round={self.round_id}\n")

            num_experts, num_clients, _ = features.shape

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

    def log_meta_alpha(self, alpha, client_num_samples):
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

    def log_meta_diagnostics(
        self,
        step_id,
        meta_loss_value,
        alpha_stats,
        meta_grad_norm,
        meta_grad_max_abs,
    ):
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

    def log_meta_final_diagnostics(self, alpha_stats):
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

    def log_alignment_diagnostics(
        self,
        alignment_scores,
        alignment_valid_mask,
        alpha=None,
    ):
        os.makedirs(
            os.path.dirname(self.train_log_path),
            exist_ok=True,
        )

        scores = np.asarray(alignment_scores, dtype=np.float32)
        valid = np.asarray(alignment_valid_mask, dtype=np.bool_)

        alpha_np = None
        if alpha is not None:
            alpha_np = alpha.detach().cpu().numpy()

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(f"[META_ALIGN_BEGIN] round={self.round_id}\n")

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

                if alpha_np is not None and valid_count > 0:
                    valid_alpha = alpha_np[expert_id][valid[expert_id]]
                    alpha_mean = float(valid_alpha.mean())
                    alpha_max = float(valid_alpha.max())
                    alpha_min = float(valid_alpha.min())
                else:
                    alpha_mean = 0.0
                    alpha_max = 0.0
                    alpha_min = 0.0

                f.write(
                    f"[META_ALIGN] "
                    f"round={self.round_id} "
                    f"expert={expert_id} "
                    f"valid_clients={valid_count} "
                    f"score_mean={score_mean:.8f} "
                    f"score_std={score_std:.8f} "
                    f"score_min={score_min:.8f} "
                    f"score_max={score_max:.8f} "
                    f"alpha_mean={alpha_mean:.8f} "
                    f"alpha_min={alpha_min:.8f} "
                    f"alpha_max={alpha_max:.8f}\n"
                )

            f.write(f"[META_ALIGN_END] round={self.round_id}\n")

    # --------------------------------------------------------
    # 5.6 原 CE meta-loss 分支
    # --------------------------------------------------------
    def aggregate_with_ce_meta_loss(
        self,
        model,
        client_state_dicts,
        client_num_samples,
        client_losses,
        client_expert_freqs,
        client_expert_losses,
        val_loader,
        non_expert_agg="sample_weighted",
    ):
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        self.log_meta_inputs(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_expert_losses=client_expert_losses,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
        )

        meta_loss_value = 0.0

        for step_id in range(1, self.meta_steps + 1):
            self.optimizer.zero_grad()

            alpha, alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_expert_losses=client_expert_losses,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                return_stats=True,
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

            meta_grad_norm, meta_grad_max_abs = self.compute_meta_grad_norm()
            meta_loss_value = meta_loss.item()

            self.log_meta_diagnostics(
                step_id=step_id,
                meta_loss_value=meta_loss_value,
                alpha_stats=alpha_stats,
                meta_grad_norm=meta_grad_norm,
                meta_grad_max_abs=meta_grad_max_abs,
            )

            self.optimizer.step()

        with torch.no_grad():
            final_alpha, final_alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_expert_losses=client_expert_losses,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                return_stats=True,
            )

        self.log_meta_final_diagnostics(final_alpha_stats)
        self.log_meta_alpha(final_alpha, client_num_samples)

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

    # --------------------------------------------------------
    # 5.7 Gradient alignment meta-loss 分支
    # --------------------------------------------------------
    def aggregate_with_gradient_alignment(
        self,
        model,
        client_state_dicts,
        client_num_samples,
        client_losses,
        client_expert_freqs,
        client_expert_losses,
        non_expert_agg,
        alignment_scores,
        alignment_valid_mask,
    ):
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        self.log_meta_inputs(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_expert_losses=client_expert_losses,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
        )

        meta_loss_value = 0.0

        for step_id in range(1, self.meta_steps + 1):
            self.optimizer.zero_grad()

            alpha, alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_expert_losses=client_expert_losses,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                return_stats=True,
                alignment_valid_mask=alignment_valid_mask,
            )

            meta_loss = self.compute_alignment_meta_loss(
                alpha=alpha,
                alignment_scores=alignment_scores,
                alignment_valid_mask=alignment_valid_mask,
            )

            if meta_loss is None:
                meta_loss_value = 0.0
                meta_grad_norm = 0.0
                meta_grad_max_abs = 0.0
            else:
                meta_loss.backward()
                meta_grad_norm, meta_grad_max_abs = self.compute_meta_grad_norm()
                meta_loss_value = meta_loss.item()
                self.optimizer.step()

            self.log_meta_diagnostics(
                step_id=step_id,
                meta_loss_value=meta_loss_value,
                alpha_stats=alpha_stats,
                meta_grad_norm=meta_grad_norm,
                meta_grad_max_abs=meta_grad_max_abs,
            )

        with torch.no_grad():
            final_alpha, final_alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_expert_losses=client_expert_losses,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                return_stats=True,
                alignment_valid_mask=alignment_valid_mask,
            )

        self.log_alignment_diagnostics(
            alignment_scores=alignment_scores,
            alignment_valid_mask=alignment_valid_mask,
            alpha=final_alpha,
        )

        self.log_meta_final_diagnostics(final_alpha_stats)
        self.log_meta_alpha(final_alpha, client_num_samples)

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

    # --------------------------------------------------------
    # 5.8 主入口
    # --------------------------------------------------------
    def aggregate(
        self,
        model,
        client_state_dicts,
        client_num_samples,
        client_losses,
        client_expert_freqs,
        client_expert_losses,
        val_loader,
        non_expert_agg="sample_weighted",
        alignment_scores=None,
        alignment_valid_mask=None,
    ):
        model.to(self.device)
        self.round_id += 1

        if self.use_gradient_alignment:
            if alignment_scores is None or alignment_valid_mask is None:
                raise ValueError(
                    "meta.use_gradient_alignment=True 时，必须传入 "
                    "alignment_scores 和 alignment_valid_mask。"
                )

            return self.aggregate_with_gradient_alignment(
                model=model,
                client_state_dicts=client_state_dicts,
                client_num_samples=client_num_samples,
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_expert_losses=client_expert_losses,
                non_expert_agg=non_expert_agg,
                alignment_scores=alignment_scores,
                alignment_valid_mask=alignment_valid_mask,
            )

        return self.aggregate_with_ce_meta_loss(
            model=model,
            client_state_dicts=client_state_dicts,
            client_num_samples=client_num_samples,
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_expert_losses=client_expert_losses,
            val_loader=val_loader,
            non_expert_agg=non_expert_agg,
        )