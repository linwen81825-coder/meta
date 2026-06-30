# model.py
# ------------------------------------------------------------
# ResNet18 + Switch-MoE Adapter + Shared Classifier
#
# 本版把原来的：
#   ResNet18 backbone
#   -> router
#   -> expert 直接输出 logits
#   -> top-k logits 加权
#
# 改成：
#   ResNet18 backbone
#   -> router
#   -> expert adapter 输出 feature 修正量
#   -> 残差融合 feature
#   -> shared classifier 输出 logits
#
# 这样 expert 不再是完整分类器，而是小的特征修正器。
#
# 好处：
#   1. expert 参数量明显降低
#   2. 所有 expert 共用同一个 classifier，输出空间更统一
#   3. expert 更新更像 feature adapter 更新，更适合服务端专家聚合
#   4. is_expert_param 仍然通过参数名里的 "experts" 判断，不影响聚合代码
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ------------------------------------------------------------
# 1. 单个 Expert Adapter
# ------------------------------------------------------------
class ExpertAdapter(nn.Module):
    """
    一个 expert adapter。

    输入：
        feature: [B, feature_dim]

    输出：
        delta_feature: [B, feature_dim]

    注意：
        这个 expert 不再直接输出 num_classes logits。
        它只负责输出一个 feature 修正量。

    结构：
        Linear(feature_dim, hidden_dim)
        ReLU
        Linear(hidden_dim, feature_dim)

    推荐：
        expert_hidden_dim 可以设置为 256 或 512。
        不建议继续用 2048，否则 adapter 仍然比较大。
    """

    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x):
        """
        x:
            [B, feature_dim]

        return:
            delta_feature:
                [B, feature_dim]
        """

        return self.net(x)


# ------------------------------------------------------------
# 2. Switch-MoE Adapter Head
# ------------------------------------------------------------
class SwitchMoEHead(nn.Module):
    """
    Switch-MoE Adapter Head。

    它包含三部分：

    1. router:
        根据 backbone feature 为每个样本选择 top-k 个 expert。

    2. experts:
        多个 ExpertAdapter。
        每个 expert 输出 feature 修正量，而不是 logits。

    3. classifier:
        共享分类头。
        所有 expert 修正后的 feature 都进入同一个 classifier。

    前向过程：

        features = backbone(x)

        router_probs = softmax(router(features))

        top-k experts 输出 delta_feature

        moe_delta = sum_k gate_k * expert_k(features)

        fused_features = features + moe_delta

        logits = classifier(fused_features)
    """

    def __init__(
        self,
        feature_dim=512,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=256,
        top_k=1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k

        if self.top_k < 1:
            raise ValueError("top_k 必须 >= 1")

        if self.top_k > self.num_experts:
            raise ValueError("top_k 不能大于 num_experts")

        # ----------------------------------------------------
        # router:
        # 输入 backbone feature，输出每个 expert 的路由分数。
        #
        # router_logits: [B, num_experts]
        # ----------------------------------------------------
        self.router = nn.Linear(feature_dim, num_experts)

        # ----------------------------------------------------
        # experts:
        # 现在 expert 是 adapter，不再是分类器。
        #
        # 参数名仍然类似：
        #   moe_head.experts.0.net.0.weight
        #   moe_head.experts.1.net.2.bias
        #
        # 所以 is_expert_param(name) 仍然可以用 "experts" 判断。
        # ----------------------------------------------------
        self.experts = nn.ModuleList([
            ExpertAdapter(
                feature_dim=feature_dim,
                hidden_dim=expert_hidden_dim,
            )
            for _ in range(num_experts)
        ])

        # ----------------------------------------------------
        # shared classifier:
        # 所有 expert adapter 修正后的 feature 共用这个分类头。
        # ----------------------------------------------------
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features, return_info=False):
        """
        features:
            [B, feature_dim]

        return:
            logits:
                [B, num_classes]
        """

        # ----------------------------------------------------
        # 1. router 计算每个样本分配给每个 expert 的概率
        # ----------------------------------------------------
        router_logits = self.router(features)              # [B, E]
        router_probs = F.softmax(router_logits, dim=-1)    # [B, E]

        # ----------------------------------------------------
        # 2. top-k routing
        # ----------------------------------------------------
        topk_probs, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1,
        )

        # topk_probs:
        #   [B, top_k]
        #
        # topk_indices:
        #   [B, top_k]

        # ----------------------------------------------------
        # 3. 对 top-k 概率重新归一化，得到 gate
        # ----------------------------------------------------
        topk_gates = topk_probs / topk_probs.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        # 为了兼容 train.py / 日志统计逻辑，保留 top1 信息。
        top1_probs = topk_probs[:, 0]       # [B]
        top1_indices = topk_indices[:, 0]   # [B]

        batch_size = features.size(0)
        device = features.device

        # ----------------------------------------------------
        # 4. 计算 MoE adapter 输出的 feature 修正量
        # ----------------------------------------------------
        moe_delta = torch.zeros(
            batch_size,
            self.feature_dim,
            device=device,
            dtype=features.dtype,
        )

        for k_id in range(self.top_k):
            # 当前 top-k 位置选择的 expert id
            selected_expert_ids = topk_indices[:, k_id]    # [B]

            # 当前 top-k 位置对应的 gate
            selected_gates = topk_gates[:, k_id]           # [B]

            for expert_id, expert in enumerate(self.experts):
                # 找出当前 top-k 位置中选择了这个 expert 的样本
                mask = selected_expert_ids == expert_id

                if mask.sum() == 0:
                    continue

                expert_features = features[mask]           # [N_e, feature_dim]

                # expert adapter 输出 feature 修正量
                expert_delta = expert(expert_features)     # [N_e, feature_dim]

                # 用 gate 加权
                expert_gate = selected_gates[mask].unsqueeze(1)
                expert_delta = expert_delta * expert_gate

                # top_k > 1 时，多个 expert 的修正量相加
                moe_delta[mask] += expert_delta

        # ----------------------------------------------------
        # 5. 残差融合
        # ----------------------------------------------------
        fused_features = features + moe_delta

        # ----------------------------------------------------
        # 6. 共享分类头输出 logits
        # ----------------------------------------------------
        logits = self.classifier(fused_features)

        if return_info:
            info = {
                "router_probs": router_probs,       # [B, num_experts]
                "top1_probs": top1_probs,           # [B]
                "top1_indices": top1_indices,       # [B]
                "topk_probs": topk_probs,           # [B, top_k]
                "topk_gates": topk_gates,           # [B, top_k]
                "topk_indices": topk_indices,       # [B, top_k]
            }

            return logits, info

        return logits


# ------------------------------------------------------------
# 3. ResNet18 + Switch-MoE Adapter 总模型
# ------------------------------------------------------------
class ResNet18SwitchMoE(nn.Module):
    """
    总模型：

        image
        ↓
        ResNet18 backbone
        ↓
        feature
        ↓
        SwitchMoEHead
            router
            expert adapters
            shared classifier
        ↓
        logits

    注意：
        ResNet18 的原始 fc 被删掉。
        最终分类由 moe_head.classifier 这个共享分类头完成。
    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=256,
        top_k=1,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 1. 构建 ResNet18 backbone
        # ----------------------------------------------------
        self.backbone = resnet18(weights=None)

        # CIFAR10 图像是 32x32，不是 ImageNet 的 224x224。
        # 原版 ResNet18 第一层是 7x7 conv + stride=2，不太适合 CIFAR10。
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # CIFAR10 图片较小，去掉 maxpool。
        self.backbone.maxpool = nn.Identity()

        # 删除 ResNet18 原始 fc。
        self.backbone.fc = nn.Identity()

        feature_dim = 512

        # ----------------------------------------------------
        # 2. 构建 MoE Adapter Head
        # ----------------------------------------------------
        self.moe_head = SwitchMoEHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            top_k=top_k,
        )

    def forward(self, x, return_info=False):
        """
        x:
            [B, 3, 32, 32]

        return:
            logits:
                [B, num_classes]
        """

        features = self.backbone(x)    # [B, 512]

        if return_info:
            logits, info = self.moe_head(
                features,
                return_info=True,
            )

            return logits, info

        logits = self.moe_head(
            features,
            return_info=False,
        )

        return logits


# ------------------------------------------------------------
# 4. 判断参数是不是 expert 参数
# ------------------------------------------------------------
def is_expert_param(name):
    """
    判断一个参数是否属于 expert。

    后面 FL 聚合时会用到：

        如果 is_expert_param(name) == True:
            用 expert_agg 聚合

        否则：
            用 non_expert_agg 聚合

    当前 adapter 版本里，expert 参数名仍然包含 "experts"。
    例如：
        moe_head.experts.0.net.0.weight
        moe_head.experts.1.net.2.bias

    shared classifier 不属于 expert 参数。
    router 也不属于 expert 参数。
    """

    return "experts" in name


# ------------------------------------------------------------
# 5. 打印参数统计
# ------------------------------------------------------------
def print_trainable_param_stats(model):
    """
    打印可训练参数中：

        expert 参数量
        non-expert 参数量
        total 参数量
        expert 参数占比
    """

    expert_params = 0
    non_expert_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_param = param.numel()

        if is_expert_param(name):
            expert_params += num_param
        else:
            non_expert_params += num_param

    total_params = expert_params + non_expert_params

    if total_params > 0:
        expert_ratio = expert_params / total_params * 100.0
    else:
        expert_ratio = 0.0

    print("========== 可训练参数统计 ==========")
    print(f"expert params     : {expert_params:,}")
    print(f"non-expert params : {non_expert_params:,}")
    print(f"total trainable   : {total_params:,}")
    print(f"expert ratio      : {expert_ratio:.2f}%")
    print("===================================")


# ------------------------------------------------------------
# 6. 简单自测
# ------------------------------------------------------------
if __name__ == "__main__":
    model = ResNet18SwitchMoE(
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=256,
        top_k=2,
    )

    print_trainable_param_stats(model)

    x = torch.randn(4, 3, 32, 32)

    logits, info = model(
        x,
        return_info=True,
    )

    print("logits shape:", logits.shape)
    print("top1 expert indices:", info["top1_indices"])
    print("top1 gate probs:", info["top1_probs"])
    print("topk expert indices:", info["topk_indices"])
    print("topk gate probs:", info["topk_gates"])