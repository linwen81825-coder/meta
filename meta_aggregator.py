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
#   元网络输入和最终聚合权重都会追加写入：
#       ./data/logs/train.log
#
#   注意：
#       这里不会用 print() 打印详细输入和权重，
#       所以控制台不会显示这些内容。
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
def update_expert_counts(expert_counts, top1_indices):
    """
    根据一个 batch 的 top1 expert id 更新 expert 激活次数。

    参数：
        expert_counts:
            shape: [num_experts]
            记录当前客户端每个 expert 被激活多少次。

        top1_indices:
            shape: [batch_size]
            每个样本被 router 分配到的 expert id。

    用法：
        logits, info = model(images, return_info=True)
        update_expert_counts(expert_counts, info["top1_indices"])
    """

    num_experts = expert_counts.numel()

    batch_counts = torch.bincount(
        top1_indices.detach().cpu(),
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
    # 例如：
    #   ./data/meta2/logs/train.log
    #
    # 注意：
    #   这里直接 open 文件写入，不使用 print，
    #   所以控制台不会显示这些详细内容。
    if train_log_path is None:
        raise ValueError("MetaExpertAggregator 必须传入 train_log_path，不能使用写死日志路径。")

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

        三个输入含义：
            loss_z:
                当前客户端 loss 在本轮客户端里的相对高低。
                这个特征做标准化。

            sample_ratio:
                当前客户端样本数占比。
                这个特征不做标准化，因为它本身就是 0~1 的比例，
                也是 FedAvg 的核心信息。

            expert_activation_frequency:
                当前 expert 在当前客户端上的激活频率。
        """

        # ----------------------------------------------------
        # 1. client loss -> loss_z
        # ----------------------------------------------------
        losses = torch.as_tensor(
            client_losses,
            dtype=torch.float32,
            device=self.device,
        )

        # 对 client loss 做标准化。
        # 标准化后表示本轮中每个客户端 loss 的相对高低。
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

        # sample_ratio 本身就是归一化比例，不再做 z-score 标准化。
        sample_ratio = sample_counts / sample_counts.sum().clamp_min(1e-6)

        # ----------------------------------------------------
        # 4. 拼接成 [E, C, 3]
        # ----------------------------------------------------

        # loss_z: [C] -> [E, C]
        loss_feature = loss_z.unsqueeze(0).expand(num_experts, num_clients)

        # sample_ratio: [C] -> [E, C]
        sample_feature = sample_ratio.unsqueeze(0).expand(num_experts, num_clients)

        # expert_freqs: [C, E] -> [E, C]
        freq_feature = expert_freqs.transpose(0, 1)

        # features: [E, C, 3]
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
    # 5.2 把元网络输入写入原始 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_inputs(
        self,
        client_losses,
        client_expert_freqs,
        client_num_samples,
    ):
        """
        记录元网络输入特征到原始训练日志文件。

        日志路径：
            ./data/logs/train.log

        注意：
            这里不使用 print，所以控制台不会显示。
            只会追加写入 train.log。

        每一行表示一个 expert-client pair：
            [META_INPUT] round=... expert=... client=...
            loss_z=... sample_ratio=... expert_freq=...
            raw_client_loss=... client_num_samples=...
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
    # 5.3 把最终聚合权重 alpha 写入原始 train.log，不打印到控制台
    # --------------------------------------------------------
    def log_meta_alpha(
        self,
        alpha,
        client_num_samples,
    ):
        """
        记录元网络最终输出的 expert 聚合权重到原始训练日志文件。

        日志路径：
            ./data/logs/train.log

        注意：
            这里不使用 print，所以控制台不会显示。
            只会追加写入 train.log。

        每一行表示一个 expert-client pair：
            [META_ALPHA] round=... expert=... client=...
            alpha=... sample_weighted_weight=...

        其中：
            alpha:
                元网络最终输出的 expert 聚合权重。

            sample_weighted_weight:
                普通 FedAvg/sample_weighted 的客户端权重。
                记录它是为了方便你对比 meta alpha 和 FedAvg 权重差异。
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

        含义：
            alpha[e, i] 表示：
                聚合 expert e 时，client i 的权重。

        对每个 expert 来说：
            sum_i alpha[e, i] = 1
        """

        features = self.build_meta_features(
            client_losses=client_losses,
            client_expert_freqs=client_expert_freqs,
            client_num_samples=client_num_samples,
        )

        num_experts, num_clients, input_dim = features.shape

        # [E, C, 3] -> [E*C, 3]
        flat_features = features.reshape(num_experts * num_clients, input_dim)

        # [E*C]
        flat_scores = self.meta_net(flat_features)

        # [E, C]
        scores = flat_scores.reshape(num_experts, num_clients)

        # 对每个 expert，在 client 维度做 softmax。
        # 每个 expert 都会得到一组客户端权重。
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

        参数：
            alpha:
                shape: [num_experts, num_clients]
        """

        # 确保 alpha 和要聚合的参数在同一个 device 上。
        # meta 更新阶段 device 一般是 cuda；
        # 最终返回 state_dict 阶段 device 一般是 cpu。
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

            # BatchNorm 的 num_batches_tracked 是整数，不能加权平均。
            # 这里沿用最简单做法：直接取第一个客户端的值。
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

                    # 注意：
                    # 这里 weight 是元网络输出的，带计算图。
                    # 所以 aggregated_tensor 也会带计算图。
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

        原因：
            load_state_dict 会切断计算图，meta_net 无法收到梯度。
            functional_call 可以让验证集 loss 反传到 alpha，再反传到 meta_net。
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

        输入：
            model:
                当前全局模型结构，用于 functional_call。

            client_state_dicts:
                本轮每个客户端训练后的参数。

            client_num_samples:
                本轮每个客户端样本数量。

            client_losses:
                本轮每个客户端平均训练 loss。
                shape: [num_clients]

            client_expert_freqs:
                本轮每个客户端 expert 激活频率。
                shape: [num_clients, num_experts]

            val_loader:
                server validation set 的 DataLoader。

            non_expert_agg:
                non-expert 参数的聚合方式。
                一般用 sample_weighted。

        输出：
            final_state_dict:
                更新元网络后，用最终 alpha 聚合得到的完整模型参数。

            info:
                一些日志信息。
        """

        model.to(self.device)

        # 当前 aggregate 被调用一次，就认为进入新的一轮 FL 聚合。
        # 这个 round_id 只用于日志。
        self.round_id += 1

        # 记录本轮元网络输入特征。
        # 只写入 ./data/logs/train.log，不打印到控制台。
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

            # 1. 元网络输出 alpha，alpha 带计算图
            alpha = self.compute_alpha(
                client_losses=client_losses,
                client_expert_freqs=client_expert_freqs,
                client_num_samples=client_num_samples,
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

            # 4. 反向更新元网络
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

            # 记录最终 expert 聚合权重。
            # 只写入 ./data/logs/train.log，不打印到控制台。
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