# meta_aggregator.py
# ------------------------------------------------------------
# 元网络专家聚合模块
#
# 本版新增：
# 1. expert embedding
#    让元网络显式知道当前输入属于哪个 expert。
#
# 2. score 标准化
#    对每个 expert 的 client score 做 z-score，再进入 softmax。
#    解决 meta score 差距太小，alpha 过度接近 uniform 的问题。
#
# 3. soft reliability
#    根据 expert_freq 对低激活 client-expert 做软惩罚。
#    不像 hard active_mask 那样直接踢掉客户端。
#
# 4. alpha EMA
#    平滑元网络最终 alpha，减少后期掉点。
#
# 5. score 日志
#    新增 [META_SCORE]，记录每个 expert-client 的：
#        score_raw
#        score_norm
#        reliability
#        score_final
#        alpha
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
        moe_head.experts.3.net.4.bias -> 3

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

    expert_indices = expert_indices.detach().cpu().reshape(-1)

    batch_counts = torch.bincount(
        expert_indices,
        minlength=num_experts,
    )

    expert_counts += batch_counts


def counts_to_frequency(expert_counts):
    """
    把 expert 激活次数转换成 expert 激活频率。
    """
    total = expert_counts.sum().item()

    if total <= 0:
        return torch.zeros_like(expert_counts, dtype=torch.float32)

    return expert_counts.float() / total


# ------------------------------------------------------------
# 4. 元网络：共享 MLP
# ------------------------------------------------------------
class MetaWeightNet(nn.Module):
    """
    元网络。

    输入：
        x:
            shape = [N, input_dim]

    输出：
        score:
            shape = [N]

    注意：
        expert embedding 在 MetaExpertAggregator 里拼到输入上。
        因此这里的 input_dim 已经是：
            原始特征维度 + expert_embedding_dim
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
        if x.dim() != 2:
            raise ValueError(
                "MetaWeightNet 输入 x 应该是二维，"
                f"shape=[N, input_dim]，实际是 {x.shape}"
            )

        if x.size(1) != self.input_dim:
            raise ValueError(
                f"MetaWeightNet 输入维度不一致: "
                f"x.input_dim={x.size(1)}, self.input_dim={self.input_dim}"
            )

        score = self.net(x).squeeze(-1)

        return score


# ------------------------------------------------------------
# 5. Meta Expert Aggregator
# ------------------------------------------------------------
class MetaExpertAggregator:
    """
    元网络专家聚合器。

    负责：
        1. 构造 client-expert 特征
        2. 输出每个 expert 对各客户端的 score
        3. score 标准化 / soft reliability / softmax 得到 alpha
        4. 用 server validation loss 更新元网络
        5. 用最终 alpha 聚合 expert 参数
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
        tau=1.0,
        active_mask=False,
        active_threshold=0.0,
        min_active_clients_per_expert=2,

        # 新增：score 标准化
        score_norm=False,
        score_norm_scale=0.5,

        # 新增：soft reliability
        soft_reliability=False,
        reliability_threshold=0.05,
        reliability_scale=0.03,
        reliability_beta=0.5,

        # 新增：alpha EMA
        alpha_ema=False,
        alpha_ema_beta=0.7,

        # 新增：expert embedding
        use_expert_embedding=False,
        expert_embedding_dim=4,
    ):
        self.num_experts = num_experts
        self.device = device
        self.meta_steps = meta_steps
        self.max_val_batches = max_val_batches
        self.eps = 1e-12

        if tau <= 0:
            raise ValueError(f"meta.tau 必须大于 0，当前 tau={tau}")

        self.tau = float(tau)

        # hard active mask 保留接口，但不建议当前主线打开
        self.active_mask = bool(active_mask)
        self.active_threshold = float(active_threshold)

        if min_active_clients_per_expert < 1:
            raise ValueError(
                "min_active_clients_per_expert 必须 >= 1，"
                f"当前值为 {min_active_clients_per_expert}"
            )

        self.min_active_clients_per_expert = int(min_active_clients_per_expert)

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
            "expert_loss_z",
            "delta_norm_z",
            "expert_count_log_z",
            "grad_cos_z",
            "val_delta_dot_z",
        }

        if not isinstance(input_features, (list, tuple)):
            raise TypeError(
                "meta.input_features 必须是 list，例如："
                "['expert_freq', 'expert_loss_z', 'delta_norm_z']"
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

        # --------------------------------------------------------
        # score norm 配置
        # --------------------------------------------------------
        self.score_norm = bool(score_norm)
        self.score_norm_scale = float(score_norm_scale)

        if self.score_norm_scale <= 0:
            raise ValueError(
                f"score_norm_scale 必须大于 0，当前值为 {self.score_norm_scale}"
            )

        # --------------------------------------------------------
        # soft reliability 配置
        # --------------------------------------------------------
        self.soft_reliability = bool(soft_reliability)
        self.reliability_threshold = float(reliability_threshold)
        self.reliability_scale = float(reliability_scale)
        self.reliability_beta = float(reliability_beta)

        if self.reliability_scale <= 0:
            raise ValueError(
                f"reliability_scale 必须大于 0，当前值为 {self.reliability_scale}"
            )

        if self.reliability_beta < 0:
            raise ValueError(
                f"reliability_beta 不能小于 0，当前值为 {self.reliability_beta}"
            )

        # --------------------------------------------------------
        # alpha EMA 配置
        # --------------------------------------------------------
        self.alpha_ema = bool(alpha_ema)
        self.alpha_ema_beta = float(alpha_ema_beta)

        if not (0.0 <= self.alpha_ema_beta < 1.0):
            raise ValueError(
                f"alpha_ema_beta 必须满足 0 <= beta < 1，"
                f"当前值为 {self.alpha_ema_beta}"
            )

        self.alpha_ema_state = None

        # --------------------------------------------------------
        # expert embedding 配置
        # --------------------------------------------------------
        self.use_expert_embedding = bool(use_expert_embedding)
        self.expert_embedding_dim = int(expert_embedding_dim)

        if self.expert_embedding_dim < 0:
            raise ValueError(
                f"expert_embedding_dim 不能小于 0，当前值为 {self.expert_embedding_dim}"
            )

        if self.use_expert_embedding and self.expert_embedding_dim == 0:
            raise ValueError(
                "use_expert_embedding=True 时，expert_embedding_dim 必须大于 0"
            )

        if self.use_expert_embedding:
            self.expert_embedding = nn.Embedding(
                num_embeddings=self.num_experts,
                embedding_dim=self.expert_embedding_dim,
            ).to(device)
        else:
            self.expert_embedding = None

        meta_input_dim = len(self.input_feature_names)

        if self.use_expert_embedding:
            meta_input_dim += self.expert_embedding_dim

        self.meta_net = MetaWeightNet(
            input_dim=meta_input_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        optimizer_params = list(self.meta_net.parameters())

        if self.expert_embedding is not None:
            optimizer_params += list(self.expert_embedding.parameters())

        self.optimizer = torch.optim.Adam(
            optimizer_params,
            lr=lr,
        )

        self.round_id = 0

        if train_log_path is None:
            raise ValueError(
                "MetaExpertAggregator 必须传入 train_log_path，不能使用写死日志路径。"
            )

        self.train_log_path = train_log_path

    # --------------------------------------------------------
    # 5.1 日志工具
    # --------------------------------------------------------
    def append_log(self, text):
        log_dir = os.path.dirname(self.train_log_path)

        if log_dir != "":
            os.makedirs(log_dir, exist_ok=True)

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(text)

    # --------------------------------------------------------
    # 5.2 基础检查：客户端数量
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
    # 5.3 每个 expert 内按 client 维度做 z-score
    # --------------------------------------------------------
    def zscore_by_expert(self, values):
        """
        对 shape=[num_experts, num_clients] 的特征做 z-score。

        对每个 expert e，在所有 client 上标准化。
        """
        if values.dim() != 2:
            raise ValueError(
                f"zscore_by_expert 输入必须是二维 [E, C]，实际是 {values.shape}"
            )

        mean = values.mean(
            dim=1,
            keepdim=True,
        )

        std = values.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-6)

        return (values - mean) / std

    # --------------------------------------------------------
    # 5.4 特征：loss_z
    # --------------------------------------------------------
    def build_loss_z_feature(
        self,
        client_losses,
        num_clients,
    ):
        """
        构造 loss_z 特征。

        输入：
            client_losses:
                [C]

        输出：
            loss_z_feature:
                [E, C]
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
    # 5.5 特征：sample_ratio
    # --------------------------------------------------------
    def build_sample_ratio_feature(
        self,
        client_num_samples,
        num_clients,
    ):
        """
        构造 sample_ratio 特征。

        输入：
            client_num_samples:
                [C]

        输出：
            sample_ratio_feature:
                [E, C]
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
    # 5.6 特征：expert_freq
    # --------------------------------------------------------
    def build_expert_freq_feature(
        self,
        client_expert_freqs,
        num_clients,
    ):
        """
        构造 expert_freq 特征。

        输入：
            client_expert_freqs:
                [C, E]

        输出：
            expert_freq_feature:
                [E, C]
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
    # 5.7 特征：expert_count_log_z
    # --------------------------------------------------------
    def build_expert_count_log_z_feature(
        self,
        client_expert_freqs,
        client_num_samples,
        num_clients,
    ):
        """
        构造 expert_count_log_z 特征。

        count_{i,e} ≈ num_samples_i * expert_freq_{i,e}

        然后：
            log_count_{i,e} = log(1 + count_{i,e})

        最后对每个 expert，在 client 维度做 z-score。

        输出：
            expert_count_log_z:
                [E, C]
        """
        expert_freqs = torch.as_tensor(
            client_expert_freqs,
            dtype=torch.float32,
            device=self.device,
        )

        sample_counts = torch.as_tensor(
            client_num_samples,
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

        if sample_counts.numel() != num_clients:
            raise ValueError(
                f"client_num_samples 数量不一致: "
                f"num_samples={sample_counts.numel()}, num_clients={num_clients}"
            )

        expert_count = expert_freqs * sample_counts.unsqueeze(1)

        expert_count_log = torch.log1p(expert_count).transpose(0, 1)

        expert_count_log_z = self.zscore_by_expert(expert_count_log)

        return expert_count_log_z

    # --------------------------------------------------------
    # 5.8 特征：expert_loss_z
    # --------------------------------------------------------
    def build_expert_loss_z_feature(
        self,
        client_expert_losses,
        num_clients,
    ):
        """
        构造 expert_loss_z 特征。

        输入：
            client_expert_losses:
                shape: [num_clients, num_experts]

        输出：
            expert_loss_z:
                shape: [num_experts, num_clients]
        """
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

        expert_loss_z = self.zscore_by_expert(expert_loss)

        return expert_loss_z

    # --------------------------------------------------------
    # 5.9 计算 expert delta norm
    # --------------------------------------------------------
    def compute_expert_delta_norms(
        self,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        """
        计算每个 client-expert pair 的参数更新幅度。

        输出：
            delta_norms:
                shape: [num_experts, num_clients]
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

    # --------------------------------------------------------
    # 5.10 特征：delta_norm_z
    # --------------------------------------------------------
    def build_delta_norm_z_feature(
        self,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        """
        构造 delta_norm_z 特征。

        输出：
            delta_norm_z:
                [E, C]
        """
        delta_norm = self.compute_expert_delta_norms(
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
            num_clients=num_clients,
        )

        delta_norm_z = self.zscore_by_expert(delta_norm)

        return delta_norm_z

    # --------------------------------------------------------
    # 5.11 在 global model 上计算 server meta-val 的 expert 梯度
    # --------------------------------------------------------
    def compute_global_expert_val_gradients(self, model, val_loader):
        """
        在当前 global model 上，用 server meta-validation set 计算 expert 参数梯度。

        返回：
            expert_grads:
                dict[name] = grad tensor on CPU

            expert_grad_sq:
                shape = [num_experts]
                每个 expert 的梯度平方和。
        """
        if val_loader is None:
            raise ValueError(
                "使用 grad_cos_z / val_delta_dot_z 时，val_loader 不能为 None。"
            )

        model.to(self.device)

        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)

        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0

        for batch_id, (images, labels) in enumerate(val_loader):
            if batch_id >= self.max_val_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss = total_loss + loss * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("server validation loader 为空，无法计算 expert 梯度")

        avg_loss = total_loss / total_samples
        avg_loss.backward()

        expert_grads = {}

        expert_grad_sq = torch.zeros(
            self.num_experts,
            dtype=torch.float64,
        )

        for name, param in model.named_parameters():
            if not is_expert_param(name):
                continue

            expert_id = get_expert_id_from_name(name)

            if expert_id is None:
                continue

            if param.grad is None:
                grad_cpu = torch.zeros_like(
                    param.detach().cpu().float()
                )
            else:
                grad_cpu = param.grad.detach().cpu().float()

            expert_grads[name] = grad_cpu

            grad_double = grad_cpu.double()

            expert_grad_sq[expert_id] += torch.sum(
                grad_double * grad_double
            ).item()

        model.zero_grad(set_to_none=True)

        if was_training:
            model.train()
        else:
            model.eval()

        return expert_grads, expert_grad_sq

    # --------------------------------------------------------
    # 5.12 特征：grad_cos_z 和 val_delta_dot_z
    # --------------------------------------------------------
    def build_direction_quality_features(
        self,
        model,
        val_loader,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        """
        构造方向性质量特征：

        1. grad_cos_z:
            cosine(Δθ_{i,e}, -g_e)

        2. val_delta_dot_z:
            - <g_e, Δθ_{i,e}>

        其中：
            Δθ_{i,e} = θ_{i,e}^{local} - θ_e^{global}
            g_e = ∇_{θ_e} L_val(global_model)
        """
        if client_state_dicts is None:
            raise ValueError(
                "使用 grad_cos_z / val_delta_dot_z 时，必须传入 client_state_dicts。"
            )

        if global_state_dict is None:
            raise ValueError(
                "使用 grad_cos_z / val_delta_dot_z 时，必须传入 global_state_dict。"
            )

        if len(client_state_dicts) != num_clients:
            raise ValueError(
                f"client_state_dicts 数量不一致: "
                f"state_dicts={len(client_state_dicts)}, num_clients={num_clients}"
            )

        expert_grads, expert_grad_sq = self.compute_global_expert_val_gradients(
            model=model,
            val_loader=val_loader,
        )

        delta_sq = torch.zeros(
            self.num_experts,
            num_clients,
            dtype=torch.float64,
        )

        val_delta_dot = torch.zeros(
            self.num_experts,
            num_clients,
            dtype=torch.float64,
        )

        for client_id, client_state in enumerate(client_state_dicts):
            for name, grad_cpu in expert_grads.items():
                if name not in client_state:
                    raise KeyError(f"client_state_dict 缺少参数: {name}")

                if name not in global_state_dict:
                    raise KeyError(f"global_state_dict 缺少参数: {name}")

                expert_id = get_expert_id_from_name(name)

                if expert_id is None:
                    continue

                client_tensor = client_state[name]

                if not torch.is_floating_point(client_tensor):
                    continue

                global_tensor = global_state_dict[name]

                client_tensor = client_tensor.detach().cpu().float()
                global_tensor = global_tensor.detach().cpu().float()

                diff = client_tensor - global_tensor

                diff_double = diff.double()
                grad_double = grad_cpu.double()

                delta_sq[expert_id, client_id] += torch.sum(
                    diff_double * diff_double
                ).item()

                val_delta_dot[expert_id, client_id] += -torch.sum(
                    grad_double * diff_double
                ).item()

        delta_norm = torch.sqrt(delta_sq).float().to(self.device)

        expert_grad_norm = torch.sqrt(expert_grad_sq).float().to(self.device)
        expert_grad_norm = expert_grad_norm.unsqueeze(1)

        val_delta_dot = val_delta_dot.float().to(self.device)

        grad_cos = val_delta_dot / (
            expert_grad_norm * delta_norm + self.eps
        )

        grad_cos_z = self.zscore_by_expert(grad_cos)
        val_delta_dot_z = self.zscore_by_expert(val_delta_dot)

        features = {
            "grad_cos_z": grad_cos_z,
            "val_delta_dot_z": val_delta_dot_z,
        }

        return features

    # --------------------------------------------------------
    # 5.13 构造本轮派生特征缓存，避免重复计算 val 梯度
    # --------------------------------------------------------
    def build_derived_feature_cache(
        self,
        model,
        val_loader,
        client_expert_freqs,
        client_num_samples,
        client_state_dicts,
        global_state_dict,
        num_clients,
    ):
        """
        构造派生特征缓存。

        这样做是为了避免：
            compute_alpha 每个 meta step 都重复计算 server val 梯度。
        """
        cache = {}

        if "expert_count_log_z" in self.input_feature_names:
            cache["expert_count_log_z"] = self.build_expert_count_log_z_feature(
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                num_clients=num_clients,
            )

        need_direction_features = (
            "grad_cos_z" in self.input_feature_names
            or "val_delta_dot_z" in self.input_feature_names
        )

        if need_direction_features:
            direction_features = self.build_direction_quality_features(
                model=model,
                val_loader=val_loader,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                num_clients=num_clients,
            )

            cache.update(direction_features)

        return cache

    # --------------------------------------------------------
    # 5.14 按配置构造元网络输入特征
    # --------------------------------------------------------
    def build_meta_features(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_expert_losses=None,
        client_state_dicts=None,
        global_state_dict=None,
        derived_feature_cache=None,
    ):
        """
        构造元网络输入特征。

        输出：
            features:
                shape: [num_experts, num_clients, input_dim]
        """
        num_clients = self.get_num_clients(
            client_num_samples=client_num_samples,
        )

        if derived_feature_cache is None:
            derived_feature_cache = {}

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

            elif feature_name == "expert_count_log_z":
                if feature_name in derived_feature_cache:
                    feature = derived_feature_cache[feature_name]
                else:
                    feature = self.build_expert_count_log_z_feature(
                        client_expert_freqs=client_expert_freqs,
                        client_num_samples=client_num_samples,
                        num_clients=num_clients,
                    )

            elif feature_name == "expert_loss_z":
                feature = self.build_expert_loss_z_feature(
                    client_expert_losses=client_expert_losses,
                    num_clients=num_clients,
                )

            elif feature_name == "delta_norm_z":
                feature = self.build_delta_norm_z_feature(
                    client_state_dicts=client_state_dicts,
                    global_state_dict=global_state_dict,
                    num_clients=num_clients,
                )

            elif feature_name in {"grad_cos_z", "val_delta_dot_z"}:
                if feature_name not in derived_feature_cache:
                    raise ValueError(
                        f"缺少派生特征 {feature_name}。"
                        "请确认 aggregate() 中已经调用 build_derived_feature_cache。"
                    )

                feature = derived_feature_cache[feature_name]

            else:
                raise ValueError(f"未知 meta input feature: {feature_name}")

            selected_features.append(feature)

        features = torch.stack(
            selected_features,
            dim=-1,
        )

        return features

    # --------------------------------------------------------
    # 5.15 拼接 expert embedding
    # --------------------------------------------------------
    def append_expert_embedding(self, features):
        """
        给每个 client-expert 输入拼接 expert embedding。

        features:
            shape = [E, C, F]

        return:
            shape = [E, C, F + expert_embedding_dim]
        """
        if not self.use_expert_embedding:
            return features

        num_experts, num_clients, _ = features.shape

        expert_ids = torch.arange(
            num_experts,
            device=features.device,
            dtype=torch.long,
        )

        expert_ids = expert_ids.view(num_experts, 1).expand(
            num_experts,
            num_clients,
        )

        expert_embed = self.expert_embedding(expert_ids)

        features = torch.cat(
            [features, expert_embed],
            dim=-1,
        )

        return features

    # --------------------------------------------------------
    # 5.16 根据 expert_freq 构造 hard active mask
    # --------------------------------------------------------
    def build_active_mask(
        self,
        client_expert_freqs,
        num_clients,
    ):
        """
        构造 active mask。

        注意：
            hard active_mask 当前不建议主线使用。
            保留只是为了消融对比。
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

    # --------------------------------------------------------
    # 5.17 score 标准化
    # --------------------------------------------------------
    def normalize_scores_by_expert(self, scores):
        """
        对每个 expert 的 client score 做标准化。

        scores:
            shape = [E, C]

        return:
            normalized_scores:
                shape = [E, C]

        作用：
            元网络 raw score 差距太小时，softmax 会接近 uniform。
            标准化后，softmax 能看到更明显的相对差距。
        """
        if not self.score_norm:
            return scores

        mean = scores.mean(
            dim=1,
            keepdim=True,
        )

        std = scores.std(
            dim=1,
            unbiased=False,
            keepdim=True,
        ).clamp_min(1e-6)

        normalized_scores = (scores - mean) / std

        normalized_scores = normalized_scores * self.score_norm_scale

        return normalized_scores

    # --------------------------------------------------------
    # 5.18 soft reliability
    # --------------------------------------------------------
    def compute_soft_reliability(
        self,
        client_expert_freqs,
        num_clients,
    ):
        """
        根据 expert_freq 计算 soft reliability。

        返回：
            reliability:
                shape = [E, C]

        reliability 接近 1：
            该 client-expert 激活频率足够，比较可靠。

        reliability 接近 0：
            该 client-expert 激活频率太低，只做软惩罚，不硬删除。
        """
        expert_freq_feature = self.build_expert_freq_feature(
            client_expert_freqs=client_expert_freqs,
            num_clients=num_clients,
        )

        if not self.soft_reliability:
            return torch.ones_like(expert_freq_feature)

        reliability = torch.sigmoid(
            (expert_freq_feature - self.reliability_threshold)
            / self.reliability_scale
        )

        reliability = reliability.clamp_min(1e-6)

        return reliability

    def apply_soft_reliability(
        self,
        scores,
        client_expert_freqs,
        num_clients,
    ):
        """
        对低激活 client-expert 做软惩罚。

        scores:
            shape = [E, C]

        return:
            adjusted_scores:
                shape = [E, C]

            reliability:
                shape = [E, C]
        """
        reliability = self.compute_soft_reliability(
            client_expert_freqs=client_expert_freqs,
            num_clients=num_clients,
        )

        if not self.soft_reliability:
            return scores, reliability

        adjusted_scores = scores + self.reliability_beta * torch.log(
            reliability.clamp_min(1e-6)
        )

        return adjusted_scores, reliability

    # --------------------------------------------------------
    # 5.19 alpha EMA
    # --------------------------------------------------------
    def apply_alpha_ema(self, alpha):
        """
        对最终 alpha 做 EMA 平滑。

        注意：
            这不是混合 sample_weighted，也不是混合 uniform。
            它只混合元网络自己上一轮输出的 alpha。

        公式：
            alpha_final = beta * alpha_prev + (1 - beta) * alpha_current
        """
        if not self.alpha_ema:
            return alpha

        current_alpha = alpha.detach()

        if self.alpha_ema_state is None:
            self.alpha_ema_state = current_alpha.clone()
            return alpha

        if self.alpha_ema_state.shape != current_alpha.shape:
            self.alpha_ema_state = current_alpha.clone()
            return alpha

        ema_alpha = (
            self.alpha_ema_beta * self.alpha_ema_state.to(current_alpha.device)
            + (1.0 - self.alpha_ema_beta) * current_alpha
        )

        ema_alpha = ema_alpha / ema_alpha.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)

        self.alpha_ema_state = ema_alpha.detach().clone()

        return ema_alpha

    # --------------------------------------------------------
    # 5.20 计算 alpha 诊断信息
    # --------------------------------------------------------
    def compute_alpha_stats(self, scores_raw, scores_for_softmax, alpha):
        """
        根据 softmax 前后信息计算诊断指标。
        """
        with torch.no_grad():
            raw = scores_raw.detach().float()
            softmax_scores = scores_for_softmax.detach().float()
            alpha_detached = alpha.detach().float()

            num_clients = alpha_detached.size(1)

            score_raw_std = raw.std(unbiased=False).item()
            score_raw_abs_mean = raw.abs().mean().item()

            score_final_std = softmax_scores.std(unbiased=False).item()
            score_final_abs_mean = softmax_scores.abs().mean().item()

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
                "score_std": score_raw_std,
                "score_abs_mean": score_raw_abs_mean,
                "score_final_std": score_final_std,
                "score_final_abs_mean": score_final_abs_mean,
                "alpha_entropy": alpha_entropy,
                "alpha_std": alpha_std,
                "alpha_max_mean": alpha_max_mean,
                "alpha_min_mean": alpha_min_mean,
            }

        return stats

    # --------------------------------------------------------
    # 5.21 计算元网络梯度范数
    # --------------------------------------------------------
    def compute_meta_grad_norm(self):
        """
        计算元网络参数梯度范数。
        """
        total_sq = 0.0
        grad_max_abs = 0.0

        params = list(self.meta_net.parameters())

        if self.expert_embedding is not None:
            params += list(self.expert_embedding.parameters())

        for param in params:
            if param.grad is None:
                continue

            grad = param.grad.detach().float()

            total_sq += torch.sum(grad * grad).item()

            current_max = grad.abs().max().item()
            grad_max_abs = max(grad_max_abs, current_max)

        grad_norm = total_sq ** 0.5

        return grad_norm, grad_max_abs

    # --------------------------------------------------------
    # 5.22 把元网络输入写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_inputs(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_expert_losses=None,
        client_state_dicts=None,
        global_state_dict=None,
        derived_feature_cache=None,
    ):
        """
        记录元网络输入特征到当前实验 train.log。
        """
        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_expert_losses=client_expert_losses,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
            derived_feature_cache=derived_feature_cache,
        )

        features = features.detach().cpu()

        lines = []
        lines.append(f"[META_INPUT_BEGIN] round={self.round_id}\n")

        num_experts, num_clients, _ = features.shape

        for expert_id in range(num_experts):
            for client_id in range(num_clients):
                feature_texts = []

                for feature_idx, feature_name in enumerate(self.input_feature_names):
                    value = features[expert_id, client_id, feature_idx].item()
                    feature_texts.append(f"{feature_name}={value:.8f}")

                feature_text = " ".join(feature_texts)

                lines.append(
                    f"[META_INPUT] "
                    f"round={self.round_id} "
                    f"expert={expert_id} "
                    f"client={client_id} "
                    f"{feature_text}\n"
                )

        lines.append(f"[META_INPUT_END] round={self.round_id}\n")

        self.append_log("".join(lines))

    # --------------------------------------------------------
    # 5.23 把最终聚合权重 alpha 写入 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_alpha(
        self,
        alpha,
        client_num_samples,
    ):
        """
        记录元网络最终输出的 expert 聚合权重到当前实验 train.log。
        """
        alpha = alpha.detach().cpu()

        sample_weighted = get_basic_weights(
            method="sample_weighted",
            client_num_samples=client_num_samples,
        )

        uniform = get_basic_weights(
            method="uniform",
            client_num_samples=client_num_samples,
        )

        lines = []
        lines.append(f"[META_ALPHA_BEGIN] round={self.round_id}\n")

        num_experts, num_clients = alpha.shape

        for expert_id in range(num_experts):
            alpha_sum = alpha[expert_id].sum().item()
            alpha_max = alpha[expert_id].max().item()
            alpha_min = alpha[expert_id].min().item()

            lines.append(
                f"[META_ALPHA_SUMMARY] "
                f"round={self.round_id} "
                f"expert={expert_id} "
                f"alpha_sum={alpha_sum:.8f} "
                f"alpha_max={alpha_max:.8f} "
                f"alpha_min={alpha_min:.8f}\n"
            )

            for client_id in range(num_clients):
                alpha_value = alpha[expert_id, client_id].item()
                sample_weight = float(sample_weighted[client_id])
                uniform_weight = float(uniform[client_id])

                lines.append(
                    f"[META_ALPHA] "
                    f"round={self.round_id} "
                    f"expert={expert_id} "
                    f"client={client_id} "
                    f"alpha={alpha_value:.8f} "
                    f"sample_weighted_weight={sample_weight:.8f} "
                    f"uniform_weight={uniform_weight:.8f}\n"
                )

        lines.append(f"[META_ALPHA_END] round={self.round_id}\n")

        self.append_log("".join(lines))

    # --------------------------------------------------------
    # 5.24 记录 score 日志
    # --------------------------------------------------------
    def log_meta_scores(
        self,
        score_pack,
        alpha,
    ):
        """
        记录每个 expert-client 的 score。

        score_raw:
            元网络原始输出。

        score_norm:
            score_norm 后的输出。
            如果没开 score_norm，则等于 score_raw。

        reliability:
            soft reliability 系数。

        score_final:
            进入 softmax 前的最终 score。
            包括 score_norm 和 soft reliability 惩罚。

        alpha:
            softmax 后权重。
        """
        scores_raw = score_pack["scores_raw"].detach().cpu()
        scores_norm = score_pack["scores_norm"].detach().cpu()
        scores_final = score_pack["scores_final"].detach().cpu()
        reliability = score_pack["reliability"].detach().cpu()
        alpha = alpha.detach().cpu()

        lines = []
        lines.append(f"[META_SCORE_BEGIN] round={self.round_id}\n")

        num_experts, num_clients = scores_raw.shape

        for expert_id in range(num_experts):
            raw_row = scores_raw[expert_id]
            norm_row = scores_norm[expert_id]
            final_row = scores_final[expert_id]

            lines.append(
                f"[META_SCORE_SUMMARY] "
                f"round={self.round_id} "
                f"expert={expert_id} "
                f"raw_mean={raw_row.mean().item():.8f} "
                f"raw_std={raw_row.std(unbiased=False).item():.8f} "
                f"norm_mean={norm_row.mean().item():.8f} "
                f"norm_std={norm_row.std(unbiased=False).item():.8f} "
                f"final_mean={final_row.mean().item():.8f} "
                f"final_std={final_row.std(unbiased=False).item():.8f}\n"
            )

            for client_id in range(num_clients):
                lines.append(
                    f"[META_SCORE] "
                    f"round={self.round_id} "
                    f"expert={expert_id} "
                    f"client={client_id} "
                    f"score_raw={scores_raw[expert_id, client_id].item():.8f} "
                    f"score_norm={scores_norm[expert_id, client_id].item():.8f} "
                    f"reliability={reliability[expert_id, client_id].item():.8f} "
                    f"score_final={scores_final[expert_id, client_id].item():.8f} "
                    f"alpha={alpha[expert_id, client_id].item():.8f}\n"
                )

        lines.append(f"[META_SCORE_END] round={self.round_id}\n")

        self.append_log("".join(lines))

    # --------------------------------------------------------
    # 5.25 记录 meta 训练诊断信息
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
        self.append_log(
            f"[META_DIAG] "
            f"round={self.round_id} "
            f"step={step_id} "
            f"meta_loss={meta_loss_value:.8f} "
            f"score_std={alpha_stats['score_std']:.8f} "
            f"score_abs_mean={alpha_stats['score_abs_mean']:.8f} "
            f"score_final_std={alpha_stats['score_final_std']:.8f} "
            f"score_final_abs_mean={alpha_stats['score_final_abs_mean']:.8f} "
            f"alpha_entropy={alpha_stats['alpha_entropy']:.8f} "
            f"alpha_std={alpha_stats['alpha_std']:.8f} "
            f"alpha_max_mean={alpha_stats['alpha_max_mean']:.8f} "
            f"alpha_min_mean={alpha_stats['alpha_min_mean']:.8f} "
            f"meta_grad_norm={meta_grad_norm:.8f} "
            f"meta_grad_max_abs={meta_grad_max_abs:.8f}\n"
        )

    # --------------------------------------------------------
    # 5.26 记录最终 alpha 的诊断信息
    # --------------------------------------------------------
    def log_meta_final_diagnostics(self, alpha_stats):
        """
        记录元网络更新完成后，最终 alpha 的诊断信息。
        """
        self.append_log(
            f"[META_DIAG_FINAL] "
            f"round={self.round_id} "
            f"score_std={alpha_stats['score_std']:.8f} "
            f"score_abs_mean={alpha_stats['score_abs_mean']:.8f} "
            f"score_final_std={alpha_stats['score_final_std']:.8f} "
            f"score_final_abs_mean={alpha_stats['score_final_abs_mean']:.8f} "
            f"alpha_entropy={alpha_stats['alpha_entropy']:.8f} "
            f"alpha_std={alpha_stats['alpha_std']:.8f} "
            f"alpha_max_mean={alpha_stats['alpha_max_mean']:.8f} "
            f"alpha_min_mean={alpha_stats['alpha_min_mean']:.8f}\n"
        )

    # --------------------------------------------------------
    # 5.27 元网络输出 alpha
    # --------------------------------------------------------
    def compute_alpha(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
        client_expert_losses=None,
        client_state_dicts=None,
        global_state_dict=None,
        derived_feature_cache=None,
        return_stats=False,
        return_scores=False,
    ):
        """
        用元网络计算 expert 聚合权重 alpha。
        """
        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_expert_losses=client_expert_losses,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
            derived_feature_cache=derived_feature_cache,
        )

        features = self.append_expert_embedding(features)

        num_experts, num_clients, input_dim = features.shape

        flat_features = features.reshape(
            num_experts * num_clients,
            input_dim,
        )

        flat_scores = self.meta_net(flat_features)

        scores_raw = flat_scores.reshape(
            num_experts,
            num_clients,
        )

        # 1. score 标准化
        scores_norm = self.normalize_scores_by_expert(scores_raw)

        # 2. soft reliability
        scores_final, reliability = self.apply_soft_reliability(
            scores=scores_norm,
            client_expert_freqs=client_expert_freqs,
            num_clients=num_clients,
        )

        # 3. hard active mask：保留接口，不建议主线打开
        scores_for_softmax = scores_final

        if self.active_mask:
            active_mask = self.build_active_mask(
                client_expert_freqs=client_expert_freqs,
                num_clients=num_clients,
            )

            scores_for_softmax = scores_for_softmax.masked_fill(
                ~active_mask,
                -1e9,
            )

        alpha = F.softmax(scores_for_softmax / self.tau, dim=1)

        stats = None
        score_pack = None

        if return_stats:
            stats = self.compute_alpha_stats(
                scores_raw=scores_raw,
                scores_for_softmax=scores_final,
                alpha=alpha,
            )

        if return_scores:
            score_pack = {
                "scores_raw": scores_raw,
                "scores_norm": scores_norm,
                "scores_final": scores_final,
                "reliability": reliability,
            }

        if return_stats and return_scores:
            return alpha, stats, score_pack

        if return_stats:
            return alpha, stats

        if return_scores:
            return alpha, score_pack

        return alpha

    # --------------------------------------------------------
    # 5.28 根据 alpha 构造聚合后的 state_dict
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
    # 5.29 在 server validation set 上计算 CE loss
    # --------------------------------------------------------
    def compute_validation_loss(self, model, temp_state_dict, val_loader):
        """
        用临时聚合参数在 server validation set 上计算 CE loss。

        注意：
            这个函数用于训练 meta_net，不能加 torch.no_grad()。
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
    # 5.30 主函数：更新元网络并返回最终聚合模型
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
    ):
        """
        元网络专家聚合主入口。
        """
        model.to(self.device)

        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        self.round_id += 1

        num_clients = self.get_num_clients(
            client_num_samples=client_num_samples,
        )

        derived_feature_cache = self.build_derived_feature_cache(
            model=model,
            val_loader=val_loader,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
            num_clients=num_clients,
        )

        self.log_meta_inputs(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
            client_expert_losses=client_expert_losses,
            client_state_dicts=client_state_dicts,
            global_state_dict=global_state_dict,
            derived_feature_cache=derived_feature_cache,
        )

        meta_loss_value = None

        for step_id in range(1, self.meta_steps + 1):
            self.optimizer.zero_grad()

            alpha, alpha_stats = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_expert_losses=client_expert_losses,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                derived_feature_cache=derived_feature_cache,
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
            final_alpha, final_alpha_stats, final_score_pack = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
                client_expert_losses=client_expert_losses,
                client_state_dicts=client_state_dicts,
                global_state_dict=global_state_dict,
                derived_feature_cache=derived_feature_cache,
                return_stats=True,
                return_scores=True,
            )

            # 只对最终用于聚合的 alpha 做 EMA。
            # 训练 meta loss 的 alpha 不做 EMA，避免影响反向传播。
            final_alpha = self.apply_alpha_ema(final_alpha)

            # EMA 后重新算一次 alpha 统计，保证日志和实际聚合一致。
            final_alpha_stats = self.compute_alpha_stats(
                scores_raw=final_score_pack["scores_raw"],
                scores_for_softmax=final_score_pack["scores_final"],
                alpha=final_alpha,
            )

        self.log_meta_final_diagnostics(
            alpha_stats=final_alpha_stats,
        )

        self.log_meta_scores(
            score_pack=final_score_pack,
            alpha=final_alpha,
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
            "score_final_std": final_alpha_stats["score_final_std"],
            "alpha_entropy": final_alpha_stats["alpha_entropy"],
        }

        return final_state_dict, info