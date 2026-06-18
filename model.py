# model.py
# ------------------------------------------------------------
# 最小版 ResNet18 + Switch-MoE 模型
#
# 功能：
# 1. ResNet18 作为 backbone 提取图像特征
# 2. Switch Router 根据特征为每个样本选择一个 expert
# 3. 每个 expert 内部包含分类头，也就是 expert 直接输出 logits
# 4. 所有参数都参与本地训练，不冻结 backbone
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ------------------------------------------------------------
# 1. 单个 Expert
# ------------------------------------------------------------
class Expert(nn.Module):
    """
    一个 expert 就是一个 MLP 分类器。

    输入：
        feature: [B, feature_dim]

    输出：
        logits: [B, num_classes]

    重点：
        最后一层 Linear(hidden_dim, num_classes) 就是分类头。
        所以分类头被放进 expert 里面，而不是放在外面共享。
    """

    def __init__(self, feature_dim=512, hidden_dim=2048, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            # 第一层：把 ResNet18 输出的 512 维特征映射到更大的 expert hidden 空间
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),

            # 第二层：继续增加 expert 的表达能力，也增加 expert 参数量
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            # 第三层：分类头
            # 注意：这个分类头属于当前 expert
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        x: [B, feature_dim]
        """
        return self.net(x)


# ------------------------------------------------------------
# 2. Switch-MoE Head
# ------------------------------------------------------------
class SwitchMoEHead(nn.Module):
    """
    Switch-MoE 分类头。

    它包含两部分：
    1. router：决定每个样本走哪个 expert
    2. experts：多个 expert，每个 expert 都能独立分类

    第一版使用 top-1 routing：
        每个样本只选择一个 expert。
    """

    def __init__(
        self,
        feature_dim=512,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=2048,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_classes = num_classes

        # router 是一个线性层
        # 输入 feature，输出每个 expert 的分数
        #
        # router_logits: [B, num_experts]
        self.router = nn.Linear(feature_dim, num_experts)

        # experts 是专家列表
        #
        # 参数名会类似：
        #   moe_head.experts.0.net.0.weight
        #   moe_head.experts.1.net.0.weight
        #
        # 后面做 FL 聚合时，只要参数名包含 "experts"，
        # 就认为它是 expert 参数。
        self.experts = nn.ModuleList([
            Expert(
                feature_dim=feature_dim,
                hidden_dim=expert_hidden_dim,
                num_classes=num_classes,
            )
            for _ in range(num_experts)
        ])

    def forward(self, features, return_info=False):
        """
        features: [B, feature_dim]

        return:
            logits: [B, num_classes]
        """

        # ----------------------------------------------------
        # 1. router 计算每个样本分配给每个 expert 的概率
        # ----------------------------------------------------
        router_logits = self.router(features)              # [B, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)    # [B, num_experts]

        # ----------------------------------------------------
        # 2. top-1 routing：每个样本选择概率最大的 expert
        # ----------------------------------------------------
        top1_probs, top1_indices = torch.max(router_probs, dim=-1)
        # top1_probs:   [B]
        # top1_indices: [B]

        batch_size = features.size(0)
        device = features.device

        # 创建最终输出 logits
        # 每个样本只会由它被分配到的 expert 产生 logits
        final_logits = torch.zeros(
            batch_size,
            self.num_classes,
            device=device,
            dtype=features.dtype,
        )

        # ----------------------------------------------------
        # 3. 按 expert 分组处理样本
        # ----------------------------------------------------
        for expert_id, expert in enumerate(self.experts):
            # 找出当前 batch 中被分配给这个 expert 的样本
            mask = top1_indices == expert_id

            # 如果当前 expert 没有被任何样本选中，就跳过
            if mask.sum() == 0:
                continue

            # 取出属于当前 expert 的样本特征
            expert_features = features[mask]  # [N_e, feature_dim]

            # 当前 expert 对这些样本进行分类
            expert_logits = expert(expert_features)  # [N_e, num_classes]

            # 用 router 的 top1 概率作为 gate
            #
            # 这样写表示：
            # expert 负责输出分类结果，
            # router 的选择置信度影响最终 logits。
            expert_gate = top1_probs[mask].unsqueeze(1)  # [N_e, 1]
            expert_logits = expert_logits * expert_gate

            # 把当前 expert 的输出放回 final_logits 对应位置
            final_logits[mask] = expert_logits

        if return_info:
            info = {
                "router_probs": router_probs,       # 每个样本对每个 expert 的概率
                "top1_probs": top1_probs,           # 每个样本被选中 expert 的概率
                "top1_indices": top1_indices,       # 每个样本选择的 expert id
            }
            return final_logits, info

        return final_logits


# ------------------------------------------------------------
# 3. ResNet18 + Switch-MoE 总模型
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
          ↓
        logits

    注意：
        ResNet18 的原始 fc 被删掉了。
        分类任务交给每个 expert 内部自己的分类头完成。
    """

    def __init__(
        self,
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=2048,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 1. 构建 ResNet18 backbone
        # ----------------------------------------------------
        self.backbone = resnet18(weights=None)

        # CIFAR10 图像是 32x32，不是 ImageNet 的 224x224
        # 原版 ResNet18 第一层是 7x7 conv + stride=2，不太适合 CIFAR10
        # 这里改成 3x3 conv + stride=1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # CIFAR10 图片很小，去掉 maxpool，避免特征图过早变小
        self.backbone.maxpool = nn.Identity()

        # ResNet18 最后的 fc 原本是分类头
        # 这里删掉，因为分类头要放到 expert 里面
        self.backbone.fc = nn.Identity()

        # ResNet18 输出特征维度是 512
        feature_dim = 512

        # ----------------------------------------------------
        # 2. 构建 Switch-MoE 分类头
        # ----------------------------------------------------
        self.moe_head = SwitchMoEHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
        )

    def forward(self, x, return_info=False):
        """
        x: [B, 3, 32, 32]

        return:
            logits: [B, num_classes]
        """

        # ResNet18 提取图像特征
        features = self.backbone(x)  # [B, 512]

        # Switch-MoE head 输出分类 logits
        if return_info:
            logits, info = self.moe_head(features, return_info=True)
            return logits, info

        logits = self.moe_head(features, return_info=False)
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
        expert 参数占比

    现在没有冻结 backbone，
    所以 backbone、router、experts 都会被统计进去。
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
    expert_ratio = expert_params / total_params * 100 if total_params > 0 else 0.0

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
    # 构建模型
    model = ResNet18SwitchMoE(
        num_classes=10,
        num_experts=4,
        expert_hidden_dim=2048,
    )

    # 打印可训练参数统计
    print_trainable_param_stats(model)

    # 构造一个假 batch
    x = torch.randn(4, 3, 32, 32)

    # 前向传播
    logits, info = model(x, return_info=True)

    print("logits shape:", logits.shape)
    print("top1 expert indices:", info["top1_indices"])
    print("top1 gate probs:", info["top1_probs"])