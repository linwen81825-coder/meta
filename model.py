# model.py
# ------------------------------------------------------------
# ResNet18 + Switch-MoE model for CIFAR-10.
#
# Normalization change in this version:
# - All BatchNorm layers inside torchvision ResNet18 are replaced at
#   construction time with GroupNorm.
# - GroupNorm has no running_mean, running_var or num_batches_tracked.
# - The public model API and state-dict naming used by train.py and
#   meta_aggregator.py remain unchanged except for BN/GN state entries.
# ------------------------------------------------------------

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ------------------------------------------------------------
# 1. Single expert
# ------------------------------------------------------------


class Expert(nn.Module):
    """MLP expert classifier.

    Input:
        features: [B, feature_dim]

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 2048,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                feature_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------
# 2. Switch-MoE head
# ------------------------------------------------------------

class SwitchMoEHead(nn.Module):
    """Switch-MoE classification head with hard Top-K routing.

    top_k=1:
        Forward uses a hard gate equal to 1 for the selected expert. A
        straight-through expression preserves gradient flow to the router.

    top_k>1:
        The selected probabilities are renormalized to sum to 1 and the
        selected expert logits are combined using those gates.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 10,
        num_experts: int = 4,
        expert_hidden_dim: int = 2048,
        top_k: int = 1,
    ) -> None:
        super().__init__()

        self.num_experts = int(num_experts)
        self.num_classes = int(num_classes)
        self.top_k = int(top_k)

        if self.num_experts < 1:
            raise ValueError(
                "num_experts must be >= 1"
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be >= 1"
            )

        if self.top_k > self.num_experts:
            raise ValueError(
                "top_k cannot exceed num_experts"
            )

        self.router = nn.Linear(
            feature_dim,
            self.num_experts,
        )

        self.experts = nn.ModuleList(
            [
                Expert(
                    feature_dim=feature_dim,
                    hidden_dim=expert_hidden_dim,
                    num_classes=self.num_classes,
                )
                for _ in range(
                    self.num_experts
                )
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        return_info: bool = False,
    ):
        router_logits = self.router(
            features
        )

        router_probs = F.softmax(
            router_logits,
            dim=-1,
        )

        topk_probs, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1,
        )

        if self.top_k == 1:
            # Forward value is exactly 1; backward gradient follows topk_probs.
            topk_gates = (
                topk_probs
                + (
                    torch.ones_like(
                        topk_probs
                    )
                    - topk_probs
                ).detach()
            )
        else:
            topk_gates = (
                topk_probs
                / topk_probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
            )

        top1_probs = topk_probs[:, 0]
        top1_indices = topk_indices[:, 0]

        batch_size = features.size(0)
        final_logits = torch.zeros(
            batch_size,
            self.num_classes,
            device=features.device,
            dtype=features.dtype,
        )

        for k_id in range(self.top_k):
            selected_expert_ids = topk_indices[
                :,
                k_id,
            ]
            selected_gates = topk_gates[
                :,
                k_id,
            ]

            for expert_id, expert in enumerate(
                self.experts
            ):
                mask = (
                    selected_expert_ids
                    == expert_id
                )

                if not torch.any(mask):
                    continue

                expert_features = features[
                    mask
                ]

                expert_logits = expert(
                    expert_features
                )

                expert_gate = selected_gates[
                    mask
                ].unsqueeze(1)

                final_logits[mask] += (
                    expert_logits
                    * expert_gate
                )

        if return_info:
            info = {
                "router_probs": router_probs,
                "top1_probs": top1_probs,
                "top1_indices": top1_indices,
                "topk_probs": topk_probs,
                "topk_gates": topk_gates,
                "topk_indices": topk_indices,
            }
            return final_logits, info

        return final_logits


# ------------------------------------------------------------
# 3. ResNet18 + Switch-MoE
# ------------------------------------------------------------

class ResNet18SwitchMoE(nn.Module):
    """CIFAR-style ResNet18 backbone with GroupNorm and Switch-MoE head."""

    def __init__(
        self,
        num_classes: int = 10,
        num_experts: int = 4,
        expert_hidden_dim: int = 2048,
        top_k: int = 1,
    ) -> None:
        super().__init__()

        # Passing norm_layer here is the complete BN -> GN conversion.
        # Every normalization layer created by torchvision ResNet18,
        # including downsample branches, becomes GroupNorm.
        self.backbone = resnet18(
            weights=None,
            norm_layer=partial(nn.GroupNorm, 32),
        )

        # CIFAR-10 uses 32x32 inputs, so use a 3x3 stride-1 stem and remove
        # the ImageNet max-pooling stage.
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        feature_dim = 512

        self.moe_head = SwitchMoEHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            top_k=top_k,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
    ):
        features = self.backbone(x)

        if return_info:
            return self.moe_head(
                features,
                return_info=True,
            )

        return self.moe_head(
            features,
            return_info=False,
        )


# ------------------------------------------------------------
# 4. Expert-parameter identification for FL aggregation
# ------------------------------------------------------------

def is_expert_param(
    name: str,
) -> bool:
    """Return True when a state/parameter name belongs to an expert."""

    return "experts" in name


# ------------------------------------------------------------
# 5. Trainable-parameter statistics
# ------------------------------------------------------------

def print_trainable_param_stats(
    model: nn.Module,
) -> None:
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

    total_params = (
        expert_params
        + non_expert_params
    )

    expert_ratio = (
        expert_params
        / total_params
        * 100.0
        if total_params > 0
        else 0.0
    )

    print(
        "========== 可训练参数统计 =========="
    )
    print(
        f"expert params     : "
        f"{expert_params:,}"
    )
    print(
        f"non-expert params : "
        f"{non_expert_params:,}"
    )
    print(
        f"total trainable   : "
        f"{total_params:,}"
    )
    print(
        f"expert ratio      : "
        f"{expert_ratio:.2f}%"
    )
    print(
        "==================================="
    )
