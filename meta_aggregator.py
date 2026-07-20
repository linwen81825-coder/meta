import os
import re
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from model import is_expert_param


LogFunction = Callable[[str], None]


def get_expert_id_from_name(name: str) -> Optional[int]:
    """从 state_dict 参数名中解析专家编号。"""
    match = re.search(r"experts\.(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def get_basic_weights(
    method: str,
    client_num_samples: Sequence[int],
) -> np.ndarray:
    """计算非专家参数的普通聚合权重。"""
    num_clients = len(client_num_samples)

    if num_clients <= 0:
        raise ValueError("客户端数量不能为空")

    if method == "uniform":
        return (
            np.ones(
                num_clients,
                dtype=np.float64,
            )
            / num_clients
        )

    if method == "sample_weighted":
        counts = np.asarray(
            client_num_samples,
            dtype=np.float64,
        )

        if np.any(counts < 0):
            raise ValueError(
                "client_num_samples 不能包含负数"
            )

        total = float(counts.sum())

        if total <= 0:
            raise ValueError(
                "client_num_samples 总和必须大于 0"
            )

        return counts / total

    raise ValueError(
        f"未知普通聚合方式: {method}"
    )


def update_expert_counts(
    expert_counts: torch.Tensor,
    expert_indices: torch.Tensor,
) -> None:
    """
    累加硬 Top-K 专家激活次数。

    Top-1 时每个样本贡献1次激活；
    Top-K 时每个样本贡献K次激活。
    """
    flat_indices = (
        expert_indices
        .detach()
        .cpu()
        .reshape(-1)
    )

    batch_counts = torch.bincount(
        flat_indices,
        minlength=expert_counts.numel(),
    )

    expert_counts += batch_counts.to(
        expert_counts.dtype
    )


def counts_to_frequency(
    expert_counts: torch.Tensor,
) -> torch.Tensor:
    """
    将客户端内部的专家激活次数转换为比例。

    分母使用全部专家的激活次数之和，
    因此同时适用于 Top-1 和 Top-K。
    """
    total = expert_counts.sum().item()

    if total <= 0:
        return torch.zeros_like(
            expert_counts,
            dtype=torch.float32,
        )

    return (
        expert_counts.float()
        / float(total)
    )


class MetaWeightNet(nn.Module):
    """对每个“专家-客户端”组合计算聚合分数。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "MetaWeightNet 的 input_dim 必须大于 0"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "MetaWeightNet 的 hidden_dim 必须大于 0"
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.ReLU(inplace=False),
        )

        self.score_head = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 3,
                self.hidden_dim,
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                self.hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "MetaWeightNet 输入必须为 "
                "[num_experts, num_clients, input_dim]，"
                f"实际为 {tuple(x.shape)}"
            )

        (
            num_experts,
            num_clients,
            input_dim,
        ) = x.shape

        if input_dim != self.input_dim:
            raise ValueError(
                "MetaWeightNet 输入维度不一致: "
                f"{input_dim} != {self.input_dim}"
            )

        flat = x.reshape(
            num_experts * num_clients,
            input_dim,
        )

        h = self.encoder(flat).reshape(
            num_experts,
            num_clients,
            self.hidden_dim,
        )

        context = (
            h.mean(
                dim=1,
                keepdim=True,
            )
            .expand(
                -1,
                num_clients,
                -1,
            )
        )

        score_input = torch.cat(
            [
                h,
                context,
                h - context,
            ],
            dim=-1,
        )

        scores = (
            self.score_head(
                score_input.reshape(
                    num_experts * num_clients,
                    self.hidden_dim * 3,
                )
            )
            .squeeze(-1)
        )

        return scores.reshape(
            num_experts,
            num_clients,
        )


class MetaExpertAggregator:
    """
    每轮只更新一次元网络，
    然后重新计算最终专家聚合权重。

    温度退火规则：
    1. 每个联邦轮次开始时计算一次 current_tau；
    2. 同一轮 alpha_before 和 alpha_final 使用相同温度；
    3. 温度不会在同一次元更新前后变化。
    """

    ALLOWED_FEATURES = {
        "loss_z",
        "loss_raw",
        "sample_ratio",
        "expert_freq",
        "expert_count_ratio",
        "expert_loss_z",
        "expert_loss_raw",
        "delta_norm_z",
    }

    ALLOWED_TAU_SCHEDULES = {
        "constant",
        "linear",
        "exponential",
    }

    def __init__(
        self,
        num_experts: int,
        device: torch.device,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        meta_steps: int = 1,
        max_val_batches: Optional[int] = None,
        train_log_path: Optional[str] = None,
        log_fn: Optional[LogFunction] = None,
        input_features: Optional[
            Sequence[str]
        ] = None,
        tau: float = 1.0,
        tau_schedule: str = "constant",
        tau_start: Optional[float] = None,
        tau_end: Optional[float] = None,
        tau_anneal_rounds: Optional[int] = None,
        total_rounds: Optional[int] = None,
        active_mask: bool = False,
        active_threshold: float = 0.0,
        min_active_clients_per_expert: int = 2,
    ):
        if num_experts <= 0:
            raise ValueError(
                "num_experts 必须大于 0"
            )

        if tau <= 0:
            raise ValueError(
                "meta.tau 必须大于 0"
            )

        if int(meta_steps) != 1:
            raise ValueError(
                "当前 full-validation 更新模式要求 "
                "meta.steps: 1；"
                "每个联邦轮次只更新一次元网络。"
            )

        if (
            max_val_batches is not None
            and int(max_val_batches) <= 0
        ):
            raise ValueError(
                "meta.max_val_batches "
                "必须为 null 或正整数"
            )

        if min_active_clients_per_expert < 1:
            raise ValueError(
                "min_active_clients_per_expert "
                "必须 >= 1"
            )

        if (
            log_fn is None
            and train_log_path is None
        ):
            raise ValueError(
                "必须传入 log_fn 或 train_log_path"
            )

        if input_features is None:
            input_features = [
                "loss_z",
                "sample_ratio",
                "expert_freq",
            ]

        if not isinstance(
            input_features,
            (list, tuple),
        ):
            raise TypeError(
                "meta.input_features 必须是 list"
            )

        input_features = list(input_features)

        if not input_features:
            raise ValueError(
                "meta.input_features 不能为空"
            )

        unknown = [
            name
            for name in input_features
            if name not in self.ALLOWED_FEATURES
        ]

        if unknown:
            raise ValueError(
                f"未知 meta input feature: {unknown}; "
                f"支持: {sorted(self.ALLOWED_FEATURES)}"
            )

        self.num_experts = int(num_experts)
        self.device = device
        self.meta_steps = 1

        self.max_val_batches = (
            None
            if max_val_batches is None
            else int(max_val_batches)
        )

        # 旧版固定温度参数。
        # 保留它是为了兼容旧配置。
        self.tau = float(tau)

        # ----------------------------------------------------
        # 温度退火配置
        # ----------------------------------------------------
        self.tau_schedule = (
            str(tau_schedule)
            .strip()
            .lower()
        )

        if (
            self.tau_schedule
            not in self.ALLOWED_TAU_SCHEDULES
        ):
            raise ValueError(
                "meta.tau_schedule 仅支持 "
                "constant / linear / exponential"
            )

        self.tau_start = float(
            self.tau
            if tau_start is None
            else tau_start
        )

        self.tau_end = float(
            self.tau_start
            if tau_end is None
            else tau_end
        )

        if (
            self.tau_start <= 0
            or self.tau_end <= 0
        ):
            raise ValueError(
                "meta.tau_start 和 "
                "meta.tau_end 必须大于 0"
            )

        self.total_rounds = (
            None
            if total_rounds is None
            else int(total_rounds)
        )

        if (
            self.total_rounds is not None
            and self.total_rounds < 1
        ):
            raise ValueError(
                "total_rounds 必须为正整数"
            )

        # tau_anneal_rounds=null 时，
        # 自动使用 train.rounds。
        self.tau_anneal_rounds = (
            self.total_rounds
            if tau_anneal_rounds is None
            else int(tau_anneal_rounds)
        )

        if self.tau_anneal_rounds is None:
            self.tau_anneal_rounds = 1

        if self.tau_anneal_rounds < 1:
            raise ValueError(
                "meta.tau_anneal_rounds "
                "必须为 null 或正整数"
            )

        # 在第一轮 aggregate() 开始时会更新。
        self.current_tau = self.tau_start

        self.active_mask = bool(active_mask)
        self.active_threshold = float(
            active_threshold
        )
        self.min_active_clients_per_expert = int(
            min_active_clients_per_expert
        )

        self.input_feature_names = input_features
        self.train_log_path = train_log_path
        self.log_fn = log_fn
        self.round_id = 0

        self.meta_net = MetaWeightNet(
            input_dim=len(
                self.input_feature_names
            ),
            hidden_dim=int(hidden_dim),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr=float(lr),
        )

    def _tau_progress(
        self,
        round_id: Optional[int] = None,
    ) -> float:
        """
        返回当前轮次的退火进度。

        第1轮进度为0；
        最后一个退火轮次进度为1；
        超出退火轮数后保持1。
        """
        current_round = (
            self.round_id
            if round_id is None
            else int(round_id)
        )

        if self.tau_anneal_rounds <= 1:
            return 1.0

        progress = (
            float(current_round - 1)
            / float(
                self.tau_anneal_rounds - 1
            )
        )

        return min(
            1.0,
            max(
                0.0,
                progress,
            ),
        )

    def scheduled_tau(
        self,
        round_id: Optional[int] = None,
    ) -> float:
        """根据联邦轮次计算当轮温度。"""
        progress = self._tau_progress(
            round_id
        )

        if self.tau_schedule == "constant":
            return self.tau_start

        if self.tau_schedule == "linear":
            return (
                self.tau_start
                + (
                    self.tau_end
                    - self.tau_start
                )
                * progress
            )

        # 指数退火：
        #
        # tau_r =
        # tau_start *
        # (tau_end / tau_start) ^ progress
        #
        # 第1轮严格等于 tau_start；
        # 最后一个退火轮次严格等于 tau_end。
        ratio = (
            self.tau_end
            / self.tau_start
        )

        return (
            self.tau_start
            * (ratio ** progress)
        )

    def _write_log(
        self,
        message: str,
    ) -> None:
        if not message.endswith("\n"):
            message += "\n"

        if self.log_fn is not None:
            self.log_fn(
                message.rstrip("\n")
            )
            return

        if self.train_log_path is None:
            return

        os.makedirs(
            os.path.dirname(
                os.path.abspath(
                    self.train_log_path
                )
            ),
            exist_ok=True,
        )

        with open(
            self.train_log_path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(message)

    def log_meta_inputs(
        self,
        stage: str,
        feature_values: Mapping[
            str,
            torch.Tensor,
        ],
        features: torch.Tensor,
    ) -> None:
        (
            num_experts,
            num_clients,
            _,
        ) = features.shape

        for expert_id in range(
            num_experts
        ):
            for client_id in range(
                num_clients
            ):
                pieces = []

                for feature_name in (
                    self.input_feature_names
                ):
                    value = feature_values[
                        feature_name
                    ][
                        expert_id,
                        client_id,
                    ]

                    pieces.append(
                        f"{feature_name}="
                        f"{float(value):.8f}"
                    )

                self._write_log(
                    f"[META_INPUT_"
                    f"{stage.upper()}] "
                    f"round={self.round_id} "
                    f"expert={expert_id} "
                    f"client={client_id} "
                    + " ".join(pieces)
                )

    def log_alpha(
        self,
        stage: str,
        alpha: torch.Tensor,
    ) -> None:
        alpha_cpu = (
            alpha
            .detach()
            .cpu()
        )

        for expert_id in range(
            alpha_cpu.shape[0]
        ):
            values = ",".join(
                f"{float(value):.8f}"
                for value
                in alpha_cpu[expert_id]
            )

            self._write_log(
                f"[META_ALPHA_"
                f"{stage.upper()}] "
                f"round={self.round_id} "
                f"expert={expert_id} "
                f"alpha=[{values}]"
            )

    def _num_clients(
        self,
        values: Sequence[int],
    ) -> int:
        num_clients = len(values)

        if num_clients <= 0:
            raise ValueError(
                "本轮客户端数量为空"
            )

        return num_clients

    def _as_client_vector(
        self,
        values: Sequence[float],
        num_clients: int,
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)

        if tensor.numel() != num_clients:
            raise ValueError(
                f"{name} 数量不一致: "
                f"{tensor.numel()} "
                f"!= {num_clients}"
            )

        return tensor

    def _as_client_expert_matrix(
        self,
        values: Sequence[
            Sequence[float]
        ],
        num_clients: int,
        name: str,
    ) -> torch.Tensor:
        if values is None:
            raise ValueError(
                f"使用 {name} 时必须传入对应数据"
            )

        tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

        expected = (
            num_clients,
            self.num_experts,
        )

        if (
            tensor.dim() != 2
            or tuple(tensor.shape)
            != expected
        ):
            raise ValueError(
                f"{name} shape 不一致: "
                f"{tuple(tensor.shape)} "
                f"!= {expected}"
            )

        return tensor

    def build_delta_norm_z_feature(
        self,
        client_state_dicts: Sequence[
            Mapping[str, torch.Tensor]
        ],
        global_state_dict: Mapping[
            str,
            torch.Tensor,
        ],
        num_clients: int,
    ) -> torch.Tensor:
        norms = torch.zeros(
            self.num_experts,
            num_clients,
            dtype=torch.float32,
            device=self.device,
        )

        for expert_id in range(
            self.num_experts
        ):
            names = [
                name
                for name
                in global_state_dict.keys()
                if (
                    is_expert_param(name)
                    and get_expert_id_from_name(
                        name
                    ) == expert_id
                    and torch.is_floating_point(
                        global_state_dict[name]
                    )
                )
            ]

            for (
                client_id,
                client_state,
            ) in enumerate(
                client_state_dicts
            ):
                squared_sum = torch.zeros(
                    (),
                    device=self.device,
                )

                for name in names:
                    delta = (
                        client_state[name]
                        .to(self.device)
                        .float()
                        - global_state_dict[name]
                        .to(self.device)
                        .float()
                    )

                    squared_sum = (
                        squared_sum
                        + torch.sum(
                            delta * delta
                        )
                    )

                norms[
                    expert_id,
                    client_id,
                ] = torch.sqrt(
                    squared_sum.clamp_min(
                        0.0
                    )
                )

        mean = norms.mean(
            dim=1,
            keepdim=True,
        )

        std = (
            norms.std(
                dim=1,
                unbiased=False,
                keepdim=True,
            )
            .clamp_min(1e-6)
        )

        return (
            norms - mean
        ) / std

    def build_meta_features(
        self,
        client_losses: Sequence[float],
        client_expert_freqs: Sequence[
            Sequence[float]
        ],
        client_num_samples: Sequence[int],
        client_expert_counts: Sequence[
            Sequence[float]
        ],
        client_expert_losses: Sequence[
            Sequence[float]
        ],
        client_state_dicts: Sequence[
            Mapping[str, torch.Tensor]
        ],
        global_state_dict: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        num_clients = self._num_clients(
            client_num_samples
        )

        losses = self._as_client_vector(
            client_losses,
            num_clients,
            "client_losses",
        )

        loss_mean = losses.mean()

        loss_std = (
            losses.std(
                unbiased=False
            )
            .clamp_min(1e-6)
        )

        loss_z = (
            (
                losses - loss_mean
            )
            / loss_std
        ).unsqueeze(0).expand(
            self.num_experts,
            num_clients,
        )

        loss_raw = (
            losses
            .unsqueeze(0)
            .expand(
                self.num_experts,
                num_clients,
            )
        )

        sample_counts = (
            self._as_client_vector(
                client_num_samples,
                num_clients,
                "client_num_samples",
            )
        )

        sample_ratio = (
            sample_counts
            / sample_counts.sum()
            .clamp_min(1e-6)
        ).unsqueeze(0).expand(
            self.num_experts,
            num_clients,
        )

        expert_freq = (
            self._as_client_expert_matrix(
                client_expert_freqs,
                num_clients,
                "client_expert_freqs",
            )
            .transpose(0, 1)
        )

        expert_counts = (
            self._as_client_expert_matrix(
                client_expert_counts,
                num_clients,
                "client_expert_counts",
            )
            .transpose(0, 1)
        )

        count_denom = (
            expert_counts.sum(
                dim=1,
                keepdim=True,
            )
        )

        uniform = torch.full_like(
            expert_counts,
            1.0 / num_clients,
        )

        expert_count_ratio = (
            torch.where(
                count_denom > 1e-12,
                (
                    expert_counts
                    / count_denom.clamp_min(
                        1e-12
                    )
                ),
                uniform,
            )
        )

        expert_loss_raw = (
            self._as_client_expert_matrix(
                client_expert_losses,
                num_clients,
                "client_expert_losses",
            )
            .transpose(0, 1)
        )

        expert_loss_mean = (
            expert_loss_raw.mean(
                dim=1,
                keepdim=True,
            )
        )

        expert_loss_std = (
            expert_loss_raw.std(
                dim=1,
                unbiased=False,
                keepdim=True,
            )
            .clamp_min(1e-6)
        )

        expert_loss_z = (
            (
                expert_loss_raw
                - expert_loss_mean
            )
            / expert_loss_std
        )

        delta_norm_z = (
            self.build_delta_norm_z_feature(
                client_state_dicts=(
                    client_state_dicts
                ),
                global_state_dict=(
                    global_state_dict
                ),
                num_clients=num_clients,
            )
        )

        feature_values: Dict[
            str,
            torch.Tensor,
        ] = {
            "loss_z": loss_z,
            "loss_raw": loss_raw,
            "sample_ratio": sample_ratio,
            "expert_freq": expert_freq,
            "expert_count_ratio": (
                expert_count_ratio
            ),
            "expert_loss_z": expert_loss_z,
            "expert_loss_raw": (
                expert_loss_raw
            ),
            "delta_norm_z": delta_norm_z,
        }

        features = torch.stack(
            [
                feature_values[name]
                for name
                in self.input_feature_names
            ],
            dim=-1,
        )

        return (
            features,
            feature_values,
        )

    def compute_alpha_from_features(
        self,
        features: torch.Tensor,
        client_expert_freqs: Sequence[
            Sequence[float]
        ],
    ) -> torch.Tensor:
        # 这里使用的是当前联邦轮次固定的温度。
        #
        # 不再使用固定 self.tau：
        # scores = self.meta_net(features) / self.tau
        #
        # 修改为：
        scores = (
            self.meta_net(features)
            / self.current_tau
        )

        if not self.active_mask:
            return torch.softmax(
                scores,
                dim=1,
            )

        num_clients = scores.shape[1]

        expert_freq = (
            self._as_client_expert_matrix(
                client_expert_freqs,
                num_clients,
                "client_expert_freqs",
            )
            .transpose(0, 1)
        )

        mask = (
            expert_freq
            > self.active_threshold
        )

        safe_mask = mask.clone()

        for expert_id in range(
            self.num_experts
        ):
            active_count = int(
                mask[expert_id]
                .sum()
                .item()
            )

            if (
                active_count
                < self.min_active_clients_per_expert
            ):
                safe_mask[expert_id] = True

        masked_scores = scores.masked_fill(
            ~safe_mask,
            float("-inf"),
        )

        return torch.softmax(
            masked_scores,
            dim=1,
        )

    def build_aggregated_state_dict(
        self,
        client_state_dicts: Sequence[
            Mapping[str, torch.Tensor]
        ],
        client_num_samples: Sequence[int],
        non_expert_agg: str,
        alpha: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if not client_state_dicts:
            raise ValueError(
                "client_state_dicts 不能为空"
            )

        num_clients = len(
            client_state_dicts
        )

        expected_shape = (
            self.num_experts,
            num_clients,
        )

        if tuple(alpha.shape) != expected_shape:
            raise ValueError(
                f"alpha shape 不一致: "
                f"{tuple(alpha.shape)} "
                f"!= {expected_shape}"
            )

        non_expert_weights = (
            get_basic_weights(
                non_expert_agg,
                client_num_samples,
            )
        )

        new_state: Dict[
            str,
            torch.Tensor,
        ] = {}

        for name in (
            client_state_dicts[0].keys()
        ):
            first = (
                client_state_dicts[0][name]
            )

            if not torch.is_floating_point(
                first
            ):
                new_state[name] = (
                    first
                    .to(device)
                    .clone()
                )
                continue

            if is_expert_param(name):
                expert_id = (
                    get_expert_id_from_name(
                        name
                    )
                )

                if expert_id is None:
                    raise ValueError(
                        "无法从参数名解析 "
                        f"expert id: {name}"
                    )

                aggregated = torch.zeros_like(
                    first,
                    device=device,
                )

                for (
                    client_id,
                    client_state,
                ) in enumerate(
                    client_state_dicts
                ):
                    aggregated = (
                        aggregated
                        + alpha[
                            expert_id,
                            client_id,
                        ]
                        * client_state[name]
                        .to(device)
                    )

            else:
                aggregated = torch.zeros_like(
                    first,
                    device=device,
                )

                for (
                    client_id,
                    client_state,
                ) in enumerate(
                    client_state_dicts
                ):
                    aggregated = (
                        aggregated
                        + float(
                            non_expert_weights[
                                client_id
                            ]
                        )
                        * client_state[name]
                        .to(device)
                    )

            new_state[name] = aggregated

        return new_state

    def compute_validation_loss(
        self,
        model: nn.Module,
        temp_state_dict: Mapping[
            str,
            torch.Tensor,
        ],
        val_loader: Iterable,
    ) -> Tuple[
        torch.Tensor,
        int,
        int,
    ]:
        """
        先计算全部被选中的验证batch loss，
        再直接求batch均值。

        当 val_size=1000 且
        val_batch_size=250 时：

        Meta loss =
        四个等大小batch loss的算术平均。
        """
        if val_loader is None:
            raise ValueError(
                "meta validation loader 不能为空"
            )

        model.eval()

        criterion = nn.CrossEntropyLoss(
            reduction="mean"
        )

        batch_losses: List[
            torch.Tensor
        ] = []

        total_samples = 0

        for (
            batch_id,
            (images, labels),
        ) in enumerate(val_loader):
            if (
                self.max_val_batches
                is not None
                and batch_id
                >= self.max_val_batches
            ):
                break

            images = images.to(
                self.device,
                non_blocking=(
                    self.device.type
                    == "cuda"
                ),
            )

            labels = labels.to(
                self.device,
                non_blocking=(
                    self.device.type
                    == "cuda"
                ),
            )

            logits = functional_call(
                model,
                temp_state_dict,
                (images,),
            )

            batch_losses.append(
                criterion(
                    logits,
                    labels,
                )
            )

            total_samples += int(
                labels.size(0)
            )

        if not batch_losses:
            raise ValueError(
                "meta validation loader 为空"
            )

        meta_loss = torch.stack(
            batch_losses,
            dim=0,
        ).mean()

        return (
            meta_loss,
            len(batch_losses),
            total_samples,
        )

    def aggregate(
        self,
        model: nn.Module,
        client_state_dicts: Sequence[
            Mapping[str, torch.Tensor]
        ],
        client_num_samples: Sequence[int],
        client_losses: Sequence[float],
        client_expert_freqs: Sequence[
            Sequence[float]
        ],
        client_expert_counts: Sequence[
            Sequence[float]
        ],
        client_expert_losses: Sequence[
            Sequence[float]
        ],
        pre_client_losses: Sequence[float],
        pre_client_expert_freqs: Sequence[
            Sequence[float]
        ],
        pre_client_expert_counts: Sequence[
            Sequence[float]
        ],
        pre_client_expert_losses: Sequence[
            Sequence[float]
        ],
        val_loader: Iterable,
        non_expert_agg: str = (
            "sample_weighted"
        ),
        second_input_same_as_first: bool = True,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, object],
    ]:
        model.to(self.device)

        self.round_id += 1

        # ----------------------------------------------------
        # 每个联邦轮次只计算一次温度。
        #
        # alpha_before 和 alpha_final 都会读取
        # self.current_tau，因此同一轮不会发生温度错位。
        # ----------------------------------------------------
        self.current_tau = (
            self.scheduled_tau(
                self.round_id
            )
        )

        tau_progress = (
            self._tau_progress(
                self.round_id
            )
        )

        self._write_log(
            f"[META_TAU] "
            f"round={self.round_id} "
            f"schedule={self.tau_schedule} "
            f"tau={self.current_tau:.10f} "
            f"progress={tau_progress:.6f} "
            f"anneal_rounds="
            f"{self.tau_anneal_rounds}"
        )

        global_state = {
            name: (
                tensor
                .detach()
                .cpu()
                .clone()
            )
            for (
                name,
                tensor,
            ) in model.state_dict().items()
        }

        # 第一次输入始终使用训练前probe统计。
        (
            first_features,
            first_values,
        ) = self.build_meta_features(
            client_losses=(
                pre_client_losses
            ),
            client_expert_freqs=(
                pre_client_expert_freqs
            ),
            client_num_samples=(
                client_num_samples
            ),
            client_expert_counts=(
                pre_client_expert_counts
            ),
            client_expert_losses=(
                pre_client_expert_losses
            ),
            client_state_dicts=(
                client_state_dicts
            ),
            global_state_dict=(
                global_state
            ),
        )

        self.log_meta_inputs(
            "first_pre",
            first_values,
            first_features,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        alpha_before = (
            self.compute_alpha_from_features(
                features=first_features,
                client_expert_freqs=(
                    pre_client_expert_freqs
                ),
            )
        )

        self.log_alpha(
            "before_update",
            alpha_before,
        )

        temporary_state = (
            self.build_aggregated_state_dict(
                client_state_dicts=(
                    client_state_dicts
                ),
                client_num_samples=(
                    client_num_samples
                ),
                non_expert_agg=(
                    non_expert_agg
                ),
                alpha=alpha_before,
                device=self.device,
            )
        )

        (
            meta_loss,
            val_batches,
            val_samples,
        ) = self.compute_validation_loss(
            model=model,
            temp_state_dict=(
                temporary_state
            ),
            val_loader=val_loader,
        )

        # 完整Meta验证loss只进行一次反向和一次更新。
        meta_loss.backward()

        self.optimizer.step()

        self._write_log(
            f"[META_FULL_VAL] "
            f"round={self.round_id} "
            f"batches={val_batches} "
            f"samples={val_samples} "
            f"batch_loss_mean="
            f"{float(meta_loss.detach()):.10f} "
            f"optimizer_steps=1"
        )

        if second_input_same_as_first:
            final_features = (
                first_features.detach()
            )

            final_values = {
                name: value.detach()
                for (
                    name,
                    value,
                ) in first_values.items()
            }

            final_freqs = (
                pre_client_expert_freqs
            )

            final_mode = (
                "same_as_first_pre"
            )

        else:
            (
                final_features,
                final_values,
            ) = self.build_meta_features(
                client_losses=(
                    client_losses
                ),
                client_expert_freqs=(
                    client_expert_freqs
                ),
                client_num_samples=(
                    client_num_samples
                ),
                client_expert_counts=(
                    client_expert_counts
                ),
                client_expert_losses=(
                    client_expert_losses
                ),
                client_state_dicts=(
                    client_state_dicts
                ),
                global_state_dict=(
                    global_state
                ),
            )

            final_features = (
                final_features.detach()
            )

            final_values = {
                name: value.detach()
                for (
                    name,
                    value,
                ) in final_values.items()
            }

            final_freqs = (
                client_expert_freqs
            )

            final_mode = (
                "local_train_trajectory"
            )

        self._write_log(
            f"[META_SECOND_INPUT_MODE] "
            f"round={self.round_id} "
            f"mode={final_mode}"
        )

        self.log_meta_inputs(
            "second_final",
            final_values,
            final_features,
        )

        # 元网络已更新，必须重新前向计算最终alpha。
        # 此时仍使用本轮固定的 current_tau。
        with torch.no_grad():
            alpha_final = (
                self.compute_alpha_from_features(
                    features=(
                        final_features
                    ),
                    client_expert_freqs=(
                        final_freqs
                    ),
                )
            )

        self.log_alpha(
            "final",
            alpha_final,
        )

        alpha_delta = torch.mean(
            torch.abs(
                alpha_final
                - alpha_before.detach()
            )
        ).item()

        self._write_log(
            f"[META_ALPHA_DELTA] "
            f"round={self.round_id} "
            f"mean_abs_delta="
            f"{alpha_delta:.10f}"
        )

        final_state_device = (
            self.build_aggregated_state_dict(
                client_state_dicts=(
                    client_state_dicts
                ),
                client_num_samples=(
                    client_num_samples
                ),
                non_expert_agg=(
                    non_expert_agg
                ),
                alpha=alpha_final,
                device=self.device,
            )
        )

        final_state = {
            name: (
                tensor
                .detach()
                .cpu()
                .clone()
            )
            for (
                name,
                tensor,
            ) in final_state_device.items()
        }

        return (
            final_state,
            {
                "meta_loss": float(
                    meta_loss
                    .detach()
                    .cpu()
                    .item()
                ),
                "alpha_before": (
                    alpha_before
                    .detach()
                    .cpu()
                ),
                "alpha": (
                    alpha_final
                    .detach()
                    .cpu()
                ),
                "alpha_delta_mean_abs": (
                    float(alpha_delta)
                ),
                "val_batches": int(
                    val_batches
                ),
                "val_samples": int(
                    val_samples
                ),
                "optimizer_steps": 1,
                "second_input_mode": (
                    final_mode
                ),

                # 新增温度诊断信息。
                "tau": float(
                    self.current_tau
                ),
                "tau_progress": float(
                    tau_progress
                ),
                "tau_schedule": (
                    self.tau_schedule
                ),
            },
        )