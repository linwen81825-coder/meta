import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import argparse
import atexit
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)
from torchvision import (
    datasets,
    transforms,
)

from meta_aggregator import (
    MetaExpertAggregator,
    counts_to_frequency,
    get_expert_id_from_name,
    update_expert_counts,
)
from model import (
    ResNet18SwitchMoE,
    is_expert_param,
    print_trainable_param_stats,
)


SOURCE_COMMIT = (
    "a475be9eedd10b93e00fe8f79291df77e8df786e"
)

_ACTIVE_LOGGER = None

SEED_NAMESPACES = {
    "model_init": "model_init_v1",
    "meta_init": "meta_init_v1",
    "client_loader": "client_loader_v1",
    "pre_probe_loader": "pre_probe_loader_v1",
    "local_train": "local_train_v1",
    "local_model_template": "local_model_template_v1",
    "server_val_loader": "server_val_loader_v1",
    "test_loader": "test_loader_v1",
    "meta_round": "meta_round_v1",
    "evaluation": "evaluation_v1",
}


# ------------------------------------------------------------
# 路径与日志
# ------------------------------------------------------------

def get_project_root() -> str:
    return os.path.dirname(
        os.path.abspath(__file__)
    )


def make_abs_path(
    path: str,
    base_dir: Optional[str] = None,
) -> str:
    path = os.path.expanduser(
        str(path)
    )

    if os.path.isabs(path):
        return os.path.abspath(path)

    if base_dir is None:
        base_dir = get_project_root()

    return os.path.abspath(
        os.path.join(
            base_dir,
            path,
        )
    )


def get_fixed_dataset_root() -> str:
    return os.path.join(
        get_project_root(),
        "data",
    )


def get_log_root(
    cfg: Mapping,
) -> str:
    dataset_cfg = cfg.get(
        "dataset",
        {},
    )

    log_root = dataset_cfg.get(
        "data_root",
        "./runs/default",
    )

    log_root = make_abs_path(
        log_root
    )

    os.makedirs(
        log_root,
        exist_ok=True,
    )

    return log_root


def get_log_name_from_root(
    log_root: str,
) -> str:
    run_name = (
        os.path.basename(
            os.path.normpath(
                log_root
            )
        )
        or "train"
    )

    return f"{run_name}.log"


def get_log_path(
    cfg: Mapping,
) -> str:
    log_root = get_log_root(cfg)

    return os.path.join(
        log_root,
        get_log_name_from_root(
            log_root
        ),
    )


def copy_config_to_log_root(
    config_path: Optional[str],
    cfg: Mapping,
) -> Optional[str]:
    if (
        config_path is None
        or not os.path.isfile(
            config_path
        )
    ):
        return None

    destination = os.path.join(
        get_log_root(cfg),
        "config_used.yaml",
    )

    shutil.copy2(
        config_path,
        destination,
    )

    return destination


class TeeLogger:
    """控制台与单个长期日志文件句柄同时输出。"""

    def __init__(
        self,
        terminal,
        log_path: str,
    ):
        self.terminal = terminal
        self.log_path = log_path
        self.log_file = open(
            self.log_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._closed = False

    def write(
        self,
        message: str,
    ) -> None:
        if self._closed:
            return
        self.terminal.write(message)
        self.log_file.write(message)

    def write_log_only(
        self,
        message: str,
    ) -> None:
        if self._closed:
            return
        self.log_file.write(message)

    def flush(self) -> None:
        if self._closed:
            return
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self.log_file.close()
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def isatty(self) -> bool:
        return bool(
            getattr(
                self.terminal,
                "isatty",
                lambda: False,
            )()
        )

    def fileno(self) -> int:
        return self.terminal.fileno()


def close_logging() -> None:
    """进程退出时刷新并关闭长期日志句柄。"""
    global _ACTIVE_LOGGER

    logger = _ACTIVE_LOGGER
    if logger is None:
        return

    if sys.stdout is logger:
        sys.stdout = logger.terminal
    if sys.stderr is logger:
        sys.stderr = logger.terminal

    logger.close()
    _ACTIVE_LOGGER = None


def setup_logging(
    cfg: Mapping,
    config_path: Optional[str] = None,
) -> str:
    global _ACTIVE_LOGGER

    close_logging()

    log_path = get_log_path(cfg)

    # 程序启动时只清空一次。
    with open(
        log_path,
        "w",
        encoding="utf-8",
    ):
        pass

    _ACTIVE_LOGGER = TeeLogger(
        sys.__stdout__,
        log_path,
    )

    sys.stdout = _ACTIVE_LOGGER
    sys.stderr = _ACTIVE_LOGGER
    atexit.register(close_logging)

    copy_config_to_log_root(
        config_path,
        cfg,
    )

    display_path = os.path.relpath(
        log_path,
        get_project_root(),
    )

    if not display_path.startswith("."):
        display_path = (
            f"./{display_path}"
        )

    print("=" * 80)
    print(
        f"日志开始时间: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(
        f"日志保存路径: "
        f"{display_path}"
    )
    print("=" * 80)

    return log_path


def log_only(
    message: str = "",
) -> None:
    if _ACTIVE_LOGGER is None:
        return

    if not message.endswith("\n"):
        message += "\n"

    _ACTIVE_LOGGER.write_log_only(
        message
    )


# ------------------------------------------------------------
# 配置与复现性
# ------------------------------------------------------------

def load_config(
    config_path: str,
) -> Dict:
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise ValueError(
            "配置文件内容必须是 YAML mapping"
        )

    return cfg


def get_reproducibility_cfg(
    cfg: Mapping,
) -> Dict[str, object]:
    user_cfg = cfg.get(
        "reproducibility",
        {},
    )

    if user_cfg is None:
        user_cfg = {}

    if not isinstance(
        user_cfg,
        Mapping,
    ):
        raise TypeError(
            "reproducibility 必须是 YAML mapping"
        )

    return {
        "enabled": bool(
            user_cfg.get(
                "enabled",
                True,
            )
        ),
        "strict_determinism": bool(
            user_cfg.get(
                "strict_determinism",
                True,
            )
        ),
        "force_num_workers_zero": bool(
            user_cfg.get(
                "force_num_workers_zero",
                True,
            )
        ),
        "disable_tf32": bool(
            user_cfg.get(
                "disable_tf32",
                True,
            )
        ),
        "log_environment": bool(
            user_cfg.get(
                "log_environment",
                True,
            )
        ),
        "log_initial_state_hash": bool(
            user_cfg.get(
                "log_initial_state_hash",
                True,
            )
        ),
        "log_first_round_client_hashes": bool(
            user_cfg.get(
                "log_first_round_client_hashes",
                True,
            )
        ),
        "log_global_state_hash_each_round": bool(
            user_cfg.get(
                "log_global_state_hash_each_round",
                True,
            )
        ),
    }


def set_seed(
    seed: int,
    repro_cfg: Mapping,
) -> None:
    seed = int(seed)

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not bool(
        repro_cfg.get(
            "enabled",
            True,
        )
    ):
        return

    strict = bool(
        repro_cfg.get(
            "strict_determinism",
            True,
        )
    )

    if strict:
        torch.use_deterministic_algorithms(
            True,
            warn_only=False,
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:
        torch.use_deterministic_algorithms(
            False
        )
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    if bool(
        repro_cfg.get(
            "disable_tf32",
            True,
        )
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if hasattr(
            torch,
            "set_float32_matmul_precision",
        ):
            torch.set_float32_matmul_precision(
                "highest"
            )


def stable_seed(
    base_seed: int,
    namespace: str,
    *parts: object,
) -> int:
    payload = "|".join(
        [
            str(int(base_seed)),
            str(namespace),
        ]
        + [
            str(part)
            for part in parts
        ]
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return value % (
        2**63 - 1
    )


def get_fork_rng_devices(
    device: torch.device,
) -> List[int]:
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
    ):
        return []

    if device.index is not None:
        return [
            int(device.index)
        ]

    return [
        int(
            torch.cuda.current_device()
        )
    ]


@contextmanager
def isolated_rng(
    seed: int,
    device: torch.device,
):
    devices = get_fork_rng_devices(
        device
    )

    with torch.random.fork_rng(
        devices=devices,
        enabled=True,
    ):
        random_state = (
            random.getstate()
        )

        numpy_state = (
            np.random.get_state()
        )

        try:
            random.seed(
                int(seed)
            )

            np.random.seed(
                int(seed)
                % (2**32)
            )

            torch.manual_seed(
                int(seed)
            )

            if (
                device.type == "cuda"
                and torch.cuda.is_available()
            ):
                torch.cuda.manual_seed(
                    int(seed)
                )
                torch.cuda.manual_seed_all(
                    int(seed)
                )

            yield

        finally:
            random.setstate(
                random_state
            )

            np.random.set_state(
                numpy_state
            )


def seed_dataloader_worker(
    worker_id: int,
) -> None:
    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    random.seed(
        worker_seed
    )

    np.random.seed(
        worker_seed
    )


def canonical_json_hash(
    value: object,
) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def hash_integer_sequences(
    sequences: Sequence[
        Sequence[int]
    ],
) -> str:
    digest = hashlib.sha256()

    for (
        sequence_id,
        sequence,
    ) in enumerate(sequences):
        array = np.asarray(
            sequence,
            dtype=np.int64,
        )

        digest.update(
            str(sequence_id).encode(
                "utf-8"
            )
        )

        digest.update(
            str(array.shape).encode(
                "utf-8"
            )
        )

        digest.update(
            array.tobytes()
        )

    return digest.hexdigest()


def hash_state_dict(
    state_dict: Mapping[
        str,
        torch.Tensor,
    ],
) -> str:
    digest = hashlib.sha256()

    for name in sorted(
        state_dict.keys()
    ):
        tensor = (
            state_dict[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode("utf-8")
        )

        digest.update(
            str(tensor.dtype).encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tuple(tensor.shape)
            ).encode("utf-8")
        )

        byte_tensor = (
            tensor
            .reshape(-1)
            .view(torch.uint8)
        )

        digest.update(
            byte_tensor
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def hash_file(
    path: str,
) -> str:
    if not os.path.isfile(path):
        return "missing"

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def run_command_text(
    command: Sequence[str],
) -> str:
    try:
        return subprocess.check_output(
            list(command),
            cwd=get_project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    except Exception:
        return "unknown"


def log_reproducibility_environment(
    cfg: Mapping,
    config_path: str,
    seed: int,
    device: torch.device,
    repro_cfg: Mapping,
) -> None:
    if not bool(
        repro_cfg.get(
            "log_environment",
            True,
        )
    ):
        return

    git_commit = run_command_text(
        [
            "git",
            "rev-parse",
            "HEAD",
        ]
    )

    git_status = run_command_text(
        [
            "git",
            "status",
            "--porcelain",
        ]
    )

    git_dirty = (
        "unknown"
        if git_status == "unknown"
        else bool(git_status)
    )

    source_files = {
        "train.py": hash_file(
            os.path.join(
                get_project_root(),
                "train.py",
            )
        ),
        "model.py": hash_file(
            os.path.join(
                get_project_root(),
                "model.py",
            )
        ),
        "meta_aggregator.py": hash_file(
            os.path.join(
                get_project_root(),
                "meta_aggregator.py",
            )
        ),
        "config": hash_file(
            make_abs_path(
                config_path,
                os.getcwd(),
            )
        ),
    }

    log_only(
        "========== [REPRO] 环境与配置 =========="
    )
    log_only(
        f"[REPRO] source_commit_reference="
        f"{SOURCE_COMMIT}"
    )
    log_only(
        f"[REPRO] git_commit="
        f"{git_commit}"
    )
    log_only(
        f"[REPRO] git_dirty="
        f"{git_dirty}"
    )
    log_only(
        f"[REPRO] seed="
        f"{int(seed)}"
    )
    log_only(
        f"[REPRO] device="
        f"{device}"
    )
    log_only(
        f"[REPRO] python="
        f"{platform.python_version()}"
    )
    log_only(
        f"[REPRO] platform="
        f"{platform.platform()}"
    )
    log_only(
        f"[REPRO] torch="
        f"{torch.__version__}"
    )
    log_only(
        f"[REPRO] cuda_runtime="
        f"{torch.version.cuda}"
    )
    log_only(
        f"[REPRO] cudnn="
        f"{torch.backends.cudnn.version()}"
    )

    if torch.cuda.is_available():
        log_only(
            f"[REPRO] gpu="
            f"{torch.cuda.get_device_name(device)}"
        )

    log_only(
        "[REPRO] deterministic_algorithms="
        f"{torch.are_deterministic_algorithms_enabled()}"
    )
    log_only(
        "[REPRO] cudnn_deterministic="
        f"{torch.backends.cudnn.deterministic}"
    )
    log_only(
        "[REPRO] cudnn_benchmark="
        f"{torch.backends.cudnn.benchmark}"
    )
    log_only(
        "[REPRO] cuda_matmul_allow_tf32="
        f"{torch.backends.cuda.matmul.allow_tf32}"
    )
    log_only(
        "[REPRO] cudnn_allow_tf32="
        f"{torch.backends.cudnn.allow_tf32}"
    )
    log_only(
        "[REPRO] CUBLAS_WORKSPACE_CONFIG="
        f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}"
    )
    log_only(
        "[REPRO] effective_config_hash="
        f"{canonical_json_hash(cfg)}"
    )

    for (
        name,
        digest,
    ) in source_files.items():
        log_only(
            f"[REPRO] file_sha256[{name}]="
            f"{digest}"
        )

    log_only(
        "[REPRO] options="
        f"{json.dumps(dict(repro_cfg), sort_keys=True)}"
    )
    log_only(
        "=========================================="
    )


# ------------------------------------------------------------
# 模型和数据
# ------------------------------------------------------------

def get_device(
    cfg: Mapping,
) -> torch.device:
    device_name = str(
        cfg.get(
            "device",
            "cuda",
        )
    )

    if (
        device_name == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def build_model(
    cfg: Mapping,
) -> ResNet18SwitchMoE:
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    return ResNet18SwitchMoE(
        num_classes=int(
            dataset_cfg[
                "num_classes"
            ]
        ),
        num_experts=int(
            model_cfg[
                "num_experts"
            ]
        ),
        expert_hidden_dim=int(
            model_cfg[
                "expert_hidden_dim"
            ]
        ),
        top_k=int(
            model_cfg.get(
                "top_k",
                1,
            )
        ),
    )


def build_datasets(
    cfg: Mapping,
):
    dataset_cfg = cfg["dataset"]

    dataset_name = str(
        dataset_cfg.get(
            "name",
            "cifar10",
        )
    ).lower()

    if dataset_name != "cifar10":
        raise ValueError(
            "当前代码只支持 cifar10，"
            f"收到 dataset.name={dataset_name}"
        )

    data_root = (
        get_fixed_dataset_root()
    )

    os.makedirs(
        data_root,
        exist_ok=True,
    )

    normalize = transforms.Normalize(
        mean=(
            0.4914,
            0.4822,
            0.4465,
        ),
        std=(
            0.2023,
            0.1994,
            0.2010,
        ),
    )

    train_transform = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    )

    test_transform = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    )

    train_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    return (
        train_set,
        test_set,
    )


def get_subset_labels(
    labels: Sequence[int],
    selected_indices: Sequence[int],
) -> List[int]:
    labels_array = np.asarray(
        labels
    )

    selected = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    return labels_array[
        selected
    ].tolist()


def compute_long_tail_class_counts(
    original_class_counts: Sequence[int],
    cfg: Mapping,
) -> np.ndarray:
    dataset_cfg = cfg["dataset"]

    long_tail_cfg = (
        dataset_cfg.get(
            "long_tail",
            {},
        )
    )

    num_classes = int(
        dataset_cfg[
            "num_classes"
        ]
    )

    mode = str(
        long_tail_cfg.get(
            "mode",
            "geometric",
        )
    )

    max_samples = (
        long_tail_cfg.get(
            "max_samples_per_class",
            None,
        )
    )

    min_samples = int(
        long_tail_cfg.get(
            "min_samples_per_class",
            1,
        )
    )

    if min_samples < 1:
        raise ValueError(
            "dataset.long_tail."
            "min_samples_per_class 必须 >= 1"
        )

    original = np.asarray(
        original_class_counts,
        dtype=np.int64,
    )

    if original.size != num_classes:
        raise ValueError(
            "original_class_counts 数量"
            "与 num_classes 不一致"
        )

    if max_samples is None:
        n_max = int(
            original.min()
        )
    else:
        n_max = min(
            int(max_samples),
            int(original.min()),
        )

    if n_max <= 0:
        raise ValueError(
            "long-tail 最大类别样本数必须大于 0"
        )

    target_counts: List[int] = []

    if mode == "geometric":
        decay_factor = float(
            long_tail_cfg.get(
                "decay_factor",
                0.9,
            )
        )

        if not (
            0 < decay_factor <= 1
        ):
            raise ValueError(
                "long_tail.decay_factor "
                "必须在 (0, 1] 内"
            )

        for rank in range(
            num_classes
        ):
            target_counts.append(
                max(
                    min_samples,
                    int(
                        round(
                            n_max
                            * decay_factor**rank
                        )
                    ),
                )
            )

    elif mode == "exp":
        imbalance_factor = float(
            long_tail_cfg.get(
                "imbalance_factor",
                0.1,
            )
        )

        if not (
            0 < imbalance_factor <= 1
        ):
            raise ValueError(
                "long_tail.imbalance_factor "
                "必须在 (0, 1] 内"
            )

        for rank in range(
            num_classes
        ):
            exponent = (
                0.0
                if num_classes == 1
                else rank
                / (
                    num_classes - 1
                )
            )

            target_counts.append(
                max(
                    min_samples,
                    int(
                        round(
                            n_max
                            * imbalance_factor
                            ** exponent
                        )
                    ),
                )
            )

    elif mode == "step":
        imbalance_factor = float(
            long_tail_cfg.get(
                "imbalance_factor",
                0.1,
            )
        )

        if not (
            0 < imbalance_factor <= 1
        ):
            raise ValueError(
                "long_tail.imbalance_factor "
                "必须在 (0, 1] 内"
            )

        for rank in range(
            num_classes
        ):
            value = (
                n_max
                if rank
                < num_classes // 2
                else int(
                    round(
                        n_max
                        * imbalance_factor
                    )
                )
            )

            target_counts.append(
                max(
                    min_samples,
                    value,
                )
            )

    else:
        raise ValueError(
            f"未知 long_tail mode={mode}; "
            "支持 geometric / exp / step"
        )

    return np.asarray(
        target_counts,
        dtype=np.int64,
    )


def build_long_tail_train_set(
    train_set,
    cfg: Mapping,
    seed: int,
):
    dataset_cfg = cfg[
        "dataset"
    ]

    long_tail_cfg = (
        dataset_cfg.get(
            "long_tail",
            {},
        )
    )

    enabled = bool(
        long_tail_cfg.get(
            "enabled",
            False,
        )
    )

    labels = np.asarray(
        train_set.targets
    )

    num_classes = int(
        dataset_cfg[
            "num_classes"
        ]
    )

    if not enabled:
        print(
            "========== 训练集 Long-tail 设置 =========="
        )
        print(
            "dataset.long_tail.enabled = false，"
            "不构造长尾训练集"
        )
        print(
            f"train samples: "
            f"{len(train_set)}"
        )
        print(
            "========================================="
        )

        log_only(
            "[REPRO] long_tail_indices_hash="
            f"{hash_integer_sequences([list(range(len(train_set)))])}"
        )

        return (
            train_set,
            labels.tolist(),
        )

    rng = np.random.default_rng(
        int(seed)
    )

    original_counts = [
        int(
            np.sum(
                labels == class_id
            )
        )
        for class_id
        in range(num_classes)
    ]

    target_by_rank = (
        compute_long_tail_class_counts(
            original_counts,
            cfg,
        )
    )

    class_order = str(
        long_tail_cfg.get(
            "class_order",
            "natural",
        )
    )

    if class_order == "natural":
        ordered_classes = list(
            range(num_classes)
        )

    elif class_order == "reverse":
        ordered_classes = list(
            reversed(
                range(num_classes)
            )
        )

    elif class_order == "random":
        ordered_classes = list(
            range(num_classes)
        )

        rng.shuffle(
            ordered_classes
        )

    else:
        raise ValueError(
            "long_tail.class_order 仅支持 "
            "natural/reverse/random"
        )

    class_to_target = {
        class_id: int(
            target_by_rank[rank]
        )
        for (
            rank,
            class_id,
        ) in enumerate(
            ordered_classes
        )
    }

    selected_indices: List[
        int
    ] = []

    selected_counts: List[
        int
    ] = []

    for class_id in range(
        num_classes
    ):
        class_indices = np.where(
            labels == class_id
        )[0]

        rng.shuffle(
            class_indices
        )

        keep_count = min(
            class_to_target[
                class_id
            ],
            len(class_indices),
        )

        selected_indices.extend(
            class_indices[
                :keep_count
            ].tolist()
        )

        selected_counts.append(
            keep_count
        )

    if bool(
        long_tail_cfg.get(
            "shuffle",
            True,
        )
    ):
        rng.shuffle(
            selected_indices
        )

    subset = Subset(
        train_set,
        selected_indices,
    )

    subset_labels = (
        get_subset_labels(
            labels,
            selected_indices,
        )
    )

    print(
        "========== 训练集 Long-tail 设置 =========="
    )
    print(
        "dataset.long_tail.enabled : true"
    )
    print(
        f"mode : "
        f"{long_tail_cfg.get('mode', 'geometric')}"
    )
    print(
        f"class_order : "
        f"{class_order}"
    )
    print(
        f"original train samples : "
        f"{len(train_set)}"
    )
    print(
        f"long-tail train samples : "
        f"{len(subset)}"
    )

    for class_id in range(
        num_classes
    ):
        print(
            f"class {class_id:02d}: "
            f"original="
            f"{original_counts[class_id]}, "
            f"long_tail="
            f"{selected_counts[class_id]}"
        )

    print(
        "========================================="
    )

    log_only(
        "[REPRO] long_tail_indices_hash="
        f"{hash_integer_sequences([selected_indices])}"
    )

    return (
        subset,
        subset_labels,
    )


# ------------------------------------------------------------
# 标签噪声
# ------------------------------------------------------------

def get_dataset_label(
    dataset,
    index: int,
) -> int:
    if isinstance(
        dataset,
        Subset,
    ):
        return get_dataset_label(
            dataset.dataset,
            dataset.indices[index],
        )

    if hasattr(
        dataset,
        "targets",
    ):
        return int(
            dataset.targets[index]
        )

    _, label = dataset[index]

    return int(label)


def validate_client_noise_rates(
    rates: Sequence[float],
) -> None:
    """
    检查每个客户端的目标标签噪声率是否合法。
    """
    for client_id, rate in enumerate(
        rates
    ):
        rate = float(rate)

        if not (
            0.0 <= rate <= 1.0
        ):
            raise ValueError(
                f"client {client_id} 的 noise rate "
                f"必须位于 [0, 1]，当前为 {rate}"
            )


def resolve_client_noise_rates(
    label_noise_cfg: Mapping,
    num_clients: int,
    default_noise_rate: float,
) -> List[float]:
    """
    解析每个客户端使用的目标标签噪声率。

    优先级：
    1. client_noise_rates 非 null：使用手动列表；
    2. rate_mode=constant：所有客户端使用 noise_rate；
    3. rate_mode=auto：按配置自动生成异构噪声率。

    自动模式支持：
    - linspace_shuffle：在 [min_rate, max_rate] 内等间隔生成，
      再使用 rate_seed 确定性打乱；
    - uniform：在 [min_rate, max_rate] 内独立均匀采样。
    """
    num_clients = int(
        num_clients
    )

    if num_clients <= 0:
        raise ValueError(
            "num_clients 必须大于 0"
        )

    manual_rates = label_noise_cfg.get(
        "client_noise_rates",
        None,
    )

    # 手动列表优先，兼容原有配置。
    if manual_rates is not None:
        if not isinstance(
            manual_rates,
            (list, tuple),
        ):
            raise TypeError(
                "client_noise_rates 必须是 list、tuple 或 null"
            )

        rates = [
            float(rate)
            for rate in manual_rates
        ]

        if len(rates) != num_clients:
            raise ValueError(
                "client_noise_rates 长度必须等于 "
                f"num_clients={num_clients}，"
                f"当前长度={len(rates)}"
            )

        validate_client_noise_rates(
            rates
        )

        return rates

    rate_mode = str(
        label_noise_cfg.get(
            "rate_mode",
            "constant",
        )
    ).lower()

    # 默认保持原行为：所有客户端使用同一个 noise_rate。
    if rate_mode == "constant":
        rates = [
            float(default_noise_rate)
            for _ in range(
                num_clients
            )
        ]

        validate_client_noise_rates(
            rates
        )

        return rates

    if rate_mode != "auto":
        raise ValueError(
            "label_noise.rate_mode 仅支持 "
            "constant / auto"
        )

    auto_cfg = label_noise_cfg.get(
        "auto",
        {},
    )

    if auto_cfg is None:
        auto_cfg = {}

    if not isinstance(
        auto_cfg,
        Mapping,
    ):
        raise TypeError(
            "label_noise.auto 必须是 YAML mapping"
        )

    distribution = str(
        auto_cfg.get(
            "distribution",
            "linspace_shuffle",
        )
    ).lower()

    min_rate = float(
        auto_cfg.get(
            "min_rate",
            0.0,
        )
    )

    max_rate = float(
        auto_cfg.get(
            "max_rate",
            default_noise_rate,
        )
    )

    rate_seed = int(
        auto_cfg.get(
            "rate_seed",
            2026,
        )
    )

    if not (
        0.0 <= min_rate <= 1.0
    ):
        raise ValueError(
            "label_noise.auto.min_rate "
            "必须位于 [0, 1]"
        )

    if not (
        0.0 <= max_rate <= 1.0
    ):
        raise ValueError(
            "label_noise.auto.max_rate "
            "必须位于 [0, 1]"
        )

    if min_rate > max_rate:
        raise ValueError(
            "label_noise.auto.min_rate "
            "不能大于 max_rate"
        )

    rng = np.random.default_rng(
        rate_seed
    )

    if distribution == "linspace_shuffle":
        # 固定覆盖整个区间，平均噪声率稳定，适合算法对比。
        rates_array = np.linspace(
            min_rate,
            max_rate,
            num=num_clients,
            dtype=np.float64,
        )

        rng.shuffle(
            rates_array
        )

    elif distribution == "uniform":
        # 每个客户端独立均匀采样。
        rates_array = rng.uniform(
            low=min_rate,
            high=max_rate,
            size=num_clients,
        )

    else:
        raise ValueError(
            "label_noise.auto.distribution "
            "仅支持 linspace_shuffle / uniform"
        )

    rates = [
        float(rate)
        for rate in rates_array.tolist()
    ]

    validate_client_noise_rates(
        rates
    )

    return rates


def get_client_noise_rate(
    label_noise_cfg: Mapping,
    client_id: int,
    num_clients: int,
    default_noise_rate: float,
) -> float:
    """
    返回指定客户端的目标标签噪声率。
    """
    rates = resolve_client_noise_rates(
        label_noise_cfg=(
            label_noise_cfg
        ),
        num_clients=num_clients,
        default_noise_rate=(
            default_noise_rate
        ),
    )

    client_id = int(
        client_id
    )

    if not (
        0 <= client_id < num_clients
    ):
        raise ValueError(
            f"client_id={client_id} 越界，"
            f"num_clients={num_clients}"
        )

    return float(
        rates[client_id]
    )


def corrupt_labels(
    clean_labels: Sequence[int],
    num_classes: int,
    noise_rate: float,
    mode: str,
    seed: int,
) -> Tuple[
    List[int],
    np.ndarray,
]:
    if not (
        0 <= noise_rate <= 1
    ):
        raise ValueError(
            "label noise_rate 必须在 [0, 1] 内"
        )

    if mode not in {
        "symmetric",
        "pairflip",
    }:
        raise ValueError(
            "label_noise.mode 仅支持 "
            "symmetric / pairflip"
        )

    rng = np.random.default_rng(
        int(seed)
    )

    clean = np.asarray(
        clean_labels,
        dtype=np.int64,
    )

    noisy = clean.copy()

    mask = (
        rng.random(clean.size)
        < noise_rate
    )

    positions = np.where(
        mask
    )[0]

    if mode == "symmetric":
        for position in positions:
            old_label = int(
                clean[position]
            )

            candidate = int(
                rng.integers(
                    0,
                    num_classes - 1,
                )
            )

            if candidate >= old_label:
                candidate += 1

            noisy[position] = (
                candidate
            )

    else:
        noisy[positions] = (
            clean[positions] + 1
        ) % num_classes

    return (
        noisy.tolist(),
        mask,
    )


class ClientLabelNoiseDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        num_classes: int,
        noise_rate: float,
        noise_mode: str,
        seed: int,
        client_id: int,
    ):
        self.base_dataset = (
            base_dataset
        )

        self.indices = list(
            indices
        )

        self.num_classes = int(
            num_classes
        )

        self.noise_rate = float(
            noise_rate
        )

        self.noise_mode = str(
            noise_mode
        )

        self.seed = int(seed)
        self.client_id = int(
            client_id
        )

        self.clean_labels = [
            get_dataset_label(
                base_dataset,
                index,
            )
            for index
            in self.indices
        ]

        (
            self.noisy_labels,
            self.noise_mask,
        ) = corrupt_labels(
            clean_labels=(
                self.clean_labels
            ),
            num_classes=(
                self.num_classes
            ),
            noise_rate=(
                self.noise_rate
            ),
            mode=self.noise_mode,
            seed=self.seed,
        )

        self.num_noisy = int(
            np.sum(
                self.noise_mask
            )
        )

        self.actual_noise_rate = (
            self.num_noisy
            / max(
                len(self.indices),
                1,
            )
        )

    def __len__(self) -> int:
        return len(
            self.indices
        )

    def __getitem__(
        self,
        item: int,
    ):
        image, _ = (
            self.base_dataset[
                self.indices[item]
            ]
        )

        return (
            image,
            self.noisy_labels[item],
        )


def build_client_dataset_with_optional_label_noise(
    train_set,
    indices: Sequence[int],
    cfg: Mapping,
    client_id: int,
    seed: int,
):
    dataset_cfg = cfg[
        "dataset"
    ]

    noise_cfg = (
        dataset_cfg.get(
            "label_noise",
            {},
        )
    )

    enabled = bool(
        noise_cfg.get(
            "enabled",
            False,
        )
    )

    if not enabled:
        return (
            Subset(
                train_set,
                indices,
            ),
            {
                "enabled": False,
                "noise_rate": 0.0,
                "actual_noise_rate": 0.0,
                "num_noisy": 0,
                "num_samples": len(
                    indices
                ),
            },
        )

    noise_rate = (
        get_client_noise_rate(
            label_noise_cfg=(
                noise_cfg
            ),
            client_id=client_id,
            num_clients=int(
                dataset_cfg[
                    "num_clients"
                ]
            ),
            default_noise_rate=float(
                noise_cfg.get(
                    "noise_rate",
                    0.0,
                )
            ),
        )
    )

    noise_mode = str(
        noise_cfg.get(
            "mode",
            "symmetric",
        )
    )

    noise_seed = int(
        seed
        + int(
            noise_cfg.get(
                "seed_offset",
                10000,
            )
        )
        + client_id
    )

    client_dataset = (
        ClientLabelNoiseDataset(
            base_dataset=train_set,
            indices=indices,
            num_classes=int(
                dataset_cfg[
                    "num_classes"
                ]
            ),
            noise_rate=noise_rate,
            noise_mode=noise_mode,
            seed=noise_seed,
            client_id=client_id,
        )
    )

    return (
        client_dataset,
        {
            "enabled": True,
            "mode": noise_mode,
            "noise_rate": noise_rate,
            "actual_noise_rate": (
                client_dataset
                .actual_noise_rate
            ),
            "num_noisy": (
                client_dataset
                .num_noisy
            ),
            "num_samples": len(
                client_dataset
            ),
            "noise_seed": noise_seed,
        },
    )


def print_label_noise_summary(
    infos: Sequence[Mapping],
) -> None:
    if not infos:
        return

    print(
        "========== 客户端标签噪声设置 =========="
    )

    if not any(
        bool(
            info.get(
                "enabled",
                False,
            )
        )
        for info in infos
    ):
        print(
            "dataset.label_noise.enabled = false，"
            "不添加标签噪声"
        )

    else:
        for (
            client_id,
            info,
        ) in enumerate(infos):
            print(
                f"client {client_id:02d}: "
                f"mode="
                f"{info.get('mode', 'none')} | "
                f"target_noise_rate="
                f"{info.get('noise_rate', 0.0):.4f} | "
                f"actual_noise_rate="
                f"{info.get('actual_noise_rate', 0.0):.4f} | "
                f"noisy="
                f"{info.get('num_noisy', 0)}/"
                f"{info.get('num_samples', 0)} | "
                f"noise_seed="
                f"{info.get('noise_seed', -1)}"
            )

    print(
        "======================================="
    )


# ------------------------------------------------------------
# Server验证集和测试集
# ------------------------------------------------------------

def split_server_validation_from_test_set(
    test_set,
    cfg: Mapping,
    seed: int,
):
    server_cfg = cfg.get(
        "server",
        {},
    )

    dataset_cfg = cfg[
        "dataset"
    ]

    val_size = int(
        server_cfg.get(
            "val_size",
            1000,
        )
    )

    num_classes = int(
        dataset_cfg[
            "num_classes"
        ]
    )

    total_size = len(
        test_set
    )

    if val_size <= 0:
        server_val_set = None
        final_test_set = test_set

        print(
            "========== Server 验证集划分 =========="
        )
        print(
            "server.val_size <= 0，"
            "不划分 server validation set"
        )
        print(
            f"final test samples: "
            f"{len(final_test_set)}"
        )
        print(
            "======================================"
        )

        log_only(
            "[REPRO] server_val_indices_hash=none"
        )

        log_only(
            "[REPRO] final_test_indices_hash="
            f"{hash_integer_sequences([list(range(total_size))])}"
        )

        return (
            server_val_set,
            final_test_set,
        )

    if val_size >= total_size:
        raise ValueError(
            "server.val_size 不能大于等于测试集大小"
        )

    if (
        val_size % num_classes
        != 0
    ):
        raise ValueError(
            "为了做 class-balanced server validation set，"
            "server.val_size 必须能被 num_classes 整除。"
        )

    samples_per_class = (
        val_size
        // num_classes
    )

    rng = np.random.default_rng(
        int(seed)
    )

    labels = np.asarray(
        test_set.targets
    )

    server_val_indices: List[
        int
    ] = []

    final_test_indices: List[
        int
    ] = []

    for class_id in range(
        num_classes
    ):
        class_indices = np.where(
            labels == class_id
        )[0]

        rng.shuffle(
            class_indices
        )

        server_val_indices.extend(
            class_indices[
                :samples_per_class
            ].tolist()
        )

        final_test_indices.extend(
            class_indices[
                samples_per_class:
            ].tolist()
        )

    rng.shuffle(
        server_val_indices
    )

    rng.shuffle(
        final_test_indices
    )

    server_val_set = Subset(
        test_set,
        server_val_indices,
    )

    final_test_set = Subset(
        test_set,
        final_test_indices,
    )

    print(
        "========== Server 验证集划分 =========="
    )
    print(
        "划分来源: CIFAR10 test set"
    )
    print(
        f"server val samples       : "
        f"{len(server_val_set)}"
    )
    print(
        f"final test samples       : "
        f"{len(final_test_set)}"
    )
    print(
        f"server val per class     : "
        f"{samples_per_class}"
    )
    print(
        "======================================"
    )

    log_only(
        "[REPRO] server_val_indices_hash="
        f"{hash_integer_sequences([server_val_indices])}"
    )

    log_only(
        "[REPRO] final_test_indices_hash="
        f"{hash_integer_sequences([final_test_indices])}"
    )

    return (
        server_val_set,
        final_test_set,
    )


# ------------------------------------------------------------
# Dirichlet划分
# ------------------------------------------------------------

def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
    max_attempts: int = 100,
) -> List[List[int]]:
    if num_clients <= 0:
        raise ValueError(
            "num_clients 必须大于 0"
        )

    if alpha <= 0:
        raise ValueError(
            "Dirichlet alpha 必须大于 0"
        )

    if min_size < 0:
        raise ValueError(
            "min_size 不能为负数"
        )

    if max_attempts <= 0:
        raise ValueError(
            "max_attempts 必须大于 0"
        )

    rng = np.random.default_rng(
        int(seed)
    )

    labels_array = np.asarray(
        labels
    )

    if labels_array.size == 0:
        raise ValueError(
            "labels 不能为空"
        )

    num_classes = (
        int(
            labels_array.max()
        )
        + 1
    )

    last_sizes: Optional[
        List[int]
    ] = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        client_indices: List[
            List[int]
        ] = [
            []
            for _ in range(
                num_clients
            )
        ]

        for class_id in range(
            num_classes
        ):
            class_indices = np.where(
                labels_array
                == class_id
            )[0]

            rng.shuffle(
                class_indices
            )

            proportions = rng.dirichlet(
                alpha
                * np.ones(
                    num_clients
                )
            )

            split_points = (
                np.cumsum(
                    proportions
                )[:-1]
                * len(
                    class_indices
                )
            ).astype(int)

            class_splits = np.split(
                class_indices,
                split_points,
            )

            for (
                client_id,
                split,
            ) in enumerate(
                class_splits
            ):
                client_indices[
                    client_id
                ].extend(
                    split.tolist()
                )

        last_sizes = [
            len(indices)
            for indices
            in client_indices
        ]

        if min(last_sizes) >= min_size:
            for client_id in range(
                num_clients
            ):
                rng.shuffle(
                    client_indices[
                        client_id
                    ]
                )

            log_only(
                f"[REPRO] dirichlet_attempts="
                f"{attempt}"
            )

            log_only(
                "[REPRO] client_partition_hash="
                f"{hash_integer_sequences(client_indices)}"
            )

            return client_indices

    raise RuntimeError(
        "Dirichlet 客户端划分失败："
        f"尝试 {max_attempts} 次后仍有客户端少于 "
        f"{min_size} 个样本；"
        f"最后一次客户端大小={last_sizes}。"
        "请增大 alpha、减小客户端数或调整 long-tail 设置。"
    )


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

def build_client_datasets(
    train_set,
    client_indices: Sequence[
        Sequence[int]
    ],
    cfg: Mapping,
    seed: int,
) -> List[Dataset]:
    client_datasets: List[
        Dataset
    ] = []

    label_noise_infos: List[
        Mapping
    ] = []

    for (
        client_id,
        indices,
    ) in enumerate(
        client_indices
    ):
        (
            client_dataset,
            noise_info,
        ) = (
            build_client_dataset_with_optional_label_noise(
                train_set=train_set,
                indices=indices,
                cfg=cfg,
                client_id=client_id,
                seed=seed,
            )
        )

        client_datasets.append(
            client_dataset
        )

        label_noise_infos.append(
            noise_info
        )

    print_label_noise_summary(
        label_noise_infos
    )

    log_only(
        "[REPRO] label_noise_info_hash="
        f"{canonical_json_hash(label_noise_infos)}"
    )

    return client_datasets


def get_effective_num_workers(
    cfg: Mapping,
    repro_cfg: Mapping,
) -> int:
    requested = int(
        cfg["train"].get(
            "num_workers",
            2,
        )
    )

    if (
        bool(
            repro_cfg.get(
                "enabled",
                True,
            )
        )
        and bool(
            repro_cfg.get(
                "force_num_workers_zero",
                True,
            )
        )
    ):
        return 0

    return requested


def build_client_loader_for_round(
    client_dataset: Dataset,
    cfg: Mapping,
    device: torch.device,
    base_seed: int,
    round_id: int,
    client_id: int,
    repro_cfg: Mapping,
) -> DataLoader:
    train_cfg = cfg[
        "train"
    ]

    loader_seed = stable_seed(
        base_seed,
        SEED_NAMESPACES[
            "client_loader"
        ],
        round_id,
        client_id,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        loader_seed
    )

    return DataLoader(
        client_dataset,
        batch_size=int(
            train_cfg[
                "batch_size"
            ]
        ),
        shuffle=True,
        num_workers=(
            get_effective_num_workers(
                cfg,
                repro_cfg,
            )
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        worker_init_fn=(
            seed_dataloader_worker
        ),
        generator=generator,
        persistent_workers=False,
    )


def build_pre_probe_loader_for_round(
    client_dataset: Dataset,
    cfg: Mapping,
    device: torch.device,
    base_seed: int,
    round_id: int,
    client_id: int,
    repro_cfg: Mapping,
) -> DataLoader:
    """
    训练前probe使用独立DataLoader和独立随机种子，
    不消耗本地训练loader的生成器状态。
    """
    meta_cfg = cfg.get(
        "meta",
        {},
    )

    batch_size = int(
        meta_cfg.get(
            "pre_probe_batch_size",
            cfg["train"][
                "batch_size"
            ],
        )
    )

    if batch_size <= 0:
        raise ValueError(
            "meta.pre_probe_batch_size 必须大于 0"
        )

    probe_seed = stable_seed(
        base_seed,
        SEED_NAMESPACES[
            "pre_probe_loader"
        ],
        round_id,
        client_id,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        probe_seed
    )

    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=(
            get_effective_num_workers(
                cfg,
                repro_cfg,
            )
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        worker_init_fn=(
            seed_dataloader_worker
        ),
        generator=generator,
        persistent_workers=False,
    )


def build_server_val_loader(
    server_val_set,
    cfg: Mapping,
    device: torch.device,
    base_seed: int,
    repro_cfg: Mapping,
) -> Optional[DataLoader]:
    if server_val_set is None:
        return None

    train_cfg = cfg[
        "train"
    ]

    server_cfg = cfg.get(
        "server",
        {},
    )

    batch_size = int(
        server_cfg.get(
            "val_batch_size",
            train_cfg.get(
                "test_batch_size",
                256,
            ),
        )
    )

    if batch_size <= 0:
        raise ValueError(
            "server.val_batch_size 必须大于 0"
        )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        stable_seed(
            base_seed,
            SEED_NAMESPACES[
                "server_val_loader"
            ],
        )
    )

    return DataLoader(
        server_val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(
            get_effective_num_workers(
                cfg,
                repro_cfg,
            )
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        worker_init_fn=(
            seed_dataloader_worker
        ),
        generator=generator,
        persistent_workers=False,
    )


def build_test_loader(
    test_set,
    cfg: Mapping,
    device: torch.device,
    base_seed: int,
    repro_cfg: Mapping,
) -> DataLoader:
    train_cfg = cfg[
        "train"
    ]

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        stable_seed(
            base_seed,
            SEED_NAMESPACES[
                "test_loader"
            ],
        )
    )

    return DataLoader(
        test_set,
        batch_size=int(
            train_cfg.get(
                "test_batch_size",
                256,
            )
        ),
        shuffle=False,
        num_workers=(
            get_effective_num_workers(
                cfg,
                repro_cfg,
            )
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        worker_init_fn=(
            seed_dataloader_worker
        ),
        generator=generator,
        persistent_workers=False,
    )


# ------------------------------------------------------------
# Loss与专家统计
# ------------------------------------------------------------

def compute_router_balance_loss(
    router_probs: torch.Tensor,
) -> torch.Tensor:
    if router_probs.dim() == 3:
        num_experts = (
            router_probs.size(-1)
        )

        router_probs = (
            router_probs.reshape(
                -1,
                num_experts,
            )
        )

    elif router_probs.dim() == 2:
        num_experts = (
            router_probs.size(-1)
        )

    else:
        raise ValueError(
            "router_probs 维度不对，"
            "期望 [B, E] 或 [B, T, E]，"
            f"实际是 {router_probs.shape}"
        )

    mean_probs = (
        router_probs.mean(dim=0)
    )

    target_probs = (
        torch.ones_like(
            mean_probs
        )
        / num_experts
    )

    return torch.sum(
        (
            mean_probs
            - target_probs
        )
        ** 2
    )


def resolve_autocast_dtype(
    dtype_name: str,
) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in aliases:
        raise ValueError(
            "performance.test_autocast_dtype 仅支持 "
            "float16/fp16 或 bfloat16/bf16"
        )
    return aliases[normalized]


@torch.no_grad()
def compute_probe_statistics(
    model: nn.Module,
    probe_loader: DataLoader,
    num_experts: int,
    device: torch.device,
    max_batches: int,
):
    """
    在本地训练开始前前向多个随机batch。

    loss：
    先累计全部样本loss，再除以全部样本数。

    专家激活比例：
    先累计全部batch的Top-K激活次数，再除以全部专家激活次数。
    """
    if max_batches <= 0:
        raise ValueError(
            "meta.pre_probe_batches 必须大于 0"
        )

    previous_training = model.training
    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    total_loss = 0.0
    total_samples = 0
    processed_batches = 0

    expert_counts = torch.zeros(
        num_experts,
        dtype=torch.long,
    )

    try:
        for (
            batch_id,
            (images, labels),
        ) in enumerate(probe_loader):
            if batch_id >= max_batches:
                break

            images = images.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )
            labels = labels.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            logits, info = model(
                images,
                return_info=True,
            )

            if "topk_indices" in info:
                expert_indices = info[
                    "topk_indices"
                ]
            else:
                expert_indices = info[
                    "top1_indices"
                ]

            update_expert_counts(
                expert_counts=expert_counts,
                expert_indices=expert_indices,
            )

            per_sample_loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                per_sample_loss.sum().item()
            )
            total_samples += int(
                labels.size(0)
            )
            processed_batches += 1

    finally:
        model.train(previous_training)

    if (
        processed_batches == 0
        or total_samples == 0
    ):
        raise ValueError(
            "训练前 probe loader 没有产生任何 batch"
        )

    avg_loss = total_loss / total_samples

    expert_freq = (
        counts_to_frequency(expert_counts)
        .numpy()
        .tolist()
    )
    expert_count_values = (
        expert_counts.numpy().tolist()
    )

    return (
        float(avg_loss),
        expert_freq,
        expert_count_values,
        processed_batches,
        total_samples,
    )


# ------------------------------------------------------------
# 本地训练
# ------------------------------------------------------------

def local_train(
    model: nn.Module,
    global_state_dict: Mapping[
        str,
        torch.Tensor,
    ],
    train_loader: DataLoader,
    probe_loader: Optional[DataLoader],
    pre_probe_batches: int,
    cfg: Mapping,
    device: torch.device,
    base_seed: int,
    round_id: int,
    client_id: int,
):
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    num_experts = int(
        model_cfg["num_experts"]
    )
    router_balance_weight = float(
        train_cfg.get(
            "router_balance_weight",
            0.0,
        )
    )

    local_seed = stable_seed(
        base_seed,
        SEED_NAMESPACES["local_train"],
        round_id,
        client_id,
    )

    with isolated_rng(
        local_seed,
        device,
    ):
        # 所有客户端复用同一个模型实例。
        # 每个客户端训练前完整加载本轮全局state_dict，
        # 从而重置参数及BN等注册buffer。
        model.load_state_dict(
            global_state_dict,
            strict=True,
        )
        model.zero_grad(set_to_none=True)
        model.to(device)

        if probe_loader is not None:
            (
                pre_loss,
                pre_expert_freq,
                pre_expert_counts,
                actual_probe_batches,
                actual_probe_samples,
            ) = compute_probe_statistics(
                model=model,
                probe_loader=probe_loader,
                num_experts=num_experts,
                device=device,
                max_batches=pre_probe_batches,
            )
        else:
            pre_loss = 0.0
            pre_expert_freq = [
                0.0
            ] * num_experts
            pre_expert_counts = [
                0
            ] * num_experts
            actual_probe_batches = 0
            actual_probe_samples = 0

        model.train()

        criterion = nn.CrossEntropyLoss(
            reduction="none"
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            momentum=float(
                train_cfg.get(
                    "momentum",
                    0.9,
                )
            ),
            weight_decay=float(
                train_cfg.get(
                    "weight_decay",
                    0.0005,
                )
            ),
        )

        total_loss = 0.0
        total_samples = 0
        expert_counts = torch.zeros(
            num_experts,
            dtype=torch.long,
        )

        local_epochs = int(
            train_cfg["local_epochs"]
        )

        for _ in range(local_epochs):
            for images, labels in train_loader:
                images = images.to(
                    device,
                    non_blocking=(
                        device.type == "cuda"
                    ),
                )
                labels = labels.to(
                    device,
                    non_blocking=(
                        device.type == "cuda"
                    ),
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits, info = model(
                    images,
                    return_info=True,
                )

                if "topk_indices" in info:
                    expert_indices = info[
                        "topk_indices"
                    ]
                else:
                    expert_indices = info[
                        "top1_indices"
                    ]

                update_expert_counts(
                    expert_counts=expert_counts,
                    expert_indices=expert_indices,
                )

                per_sample_ce_loss = criterion(
                    logits,
                    labels,
                )
                ce_loss = (
                    per_sample_ce_loss.mean()
                )

                balance_loss = torch.tensor(
                    0.0,
                    device=device,
                )
                if router_balance_weight > 0:
                    if "router_probs" not in info:
                        raise ValueError(
                            "model(images, return_info=True) "
                            "没有返回 router_probs，请先在 "
                            "model.py 的 info 中加入。"
                        )
                    balance_loss = (
                        compute_router_balance_loss(
                            info["router_probs"]
                        )
                    )

                loss = (
                    ce_loss
                    + router_balance_weight
                    * balance_loss
                )
                loss.backward()
                optimizer.step()

                total_loss += (
                    per_sample_ce_loss
                    .detach()
                    .sum()
                    .item()
                )
                total_samples += int(
                    images.size(0)
                )

        avg_loss = total_loss / max(
            total_samples,
            1,
        )
        expert_freq = (
            counts_to_frequency(expert_counts)
            .numpy()
            .tolist()
        )
        expert_count_values = (
            expert_counts.numpy().tolist()
        )

        local_state_dict = {
            name: (
                tensor.detach().cpu().clone()
            )
            for name, tensor
            in model.state_dict().items()
        }
        num_samples = len(
            train_loader.dataset
        )

        # 复用模型，不在每个客户端结束后调用
        # torch.cuda.empty_cache()。
        del optimizer

    return (
        local_state_dict,
        num_samples,
        avg_loss,
        expert_freq,
        expert_count_values,
        pre_loss,
        pre_expert_freq,
        pre_expert_counts,
        actual_probe_batches,
        actual_probe_samples,
    )


# ------------------------------------------------------------
# 测试
# ------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_autocast: bool = True,
    autocast_dtype: torch.dtype = torch.float16,
) -> Tuple[float, float]:
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        labels = labels.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        autocast_enabled = bool(
            use_autocast
            and device.type == "cuda"
        )

        if autocast_enabled:
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            ):
                logits = model(images)
        else:
            logits = model(images)

        # 前向使用FP16/BF16，loss归约仍转回FP32。
        loss = criterion(
            logits.float(),
            labels,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    if total <= 0:
        raise ValueError(
            "测试 DataLoader 为空"
        )

    return (
        correct
        / total
        * 100.0,
        total_loss
        / total,
    )


# ------------------------------------------------------------
# 普通聚合
# ------------------------------------------------------------

def get_aggregation_weights(
    method: str,
    client_num_samples: Sequence[int],
) -> np.ndarray:
    num_clients = len(
        client_num_samples
    )

    if num_clients <= 0:
        raise ValueError(
            "客户端数量不能为空"
        )

    if method == "uniform":
        return (
            np.ones(
                num_clients,
                dtype=np.float64,
            )
            / num_clients
        )

    if method == "sample_weighted":
        sample_counts = np.asarray(
            client_num_samples,
            dtype=np.float64,
        )

        total = float(
            sample_counts.sum()
        )

        if total <= 0:
            raise ValueError(
                "client_num_samples 总和必须大于 0"
            )

        return (
            sample_counts
            / total
        )

    raise ValueError(
        f"未知聚合方式: {method}"
    )


def get_expert_activation_count_weights(
    client_expert_counts: Sequence[
        Sequence[float]
    ],
    expert_id: int,
    num_clients: int,
) -> np.ndarray:
    if client_expert_counts is None:
        raise ValueError(
            "expert_agg="
            "expert_activation_count_weighted 时，"
            "必须传入 client_expert_counts。"
        )

    count_values = np.asarray(
        client_expert_counts,
        dtype=np.float64,
    )

    if count_values.ndim != 2:
        raise ValueError(
            "client_expert_counts 应该是二维 "
            "[num_clients, num_experts]"
        )

    (
        input_num_clients,
        num_experts,
    ) = count_values.shape

    if (
        input_num_clients
        != num_clients
    ):
        raise ValueError(
            "client_expert_counts 的客户端数量不一致"
        )

    if (
        expert_id < 0
        or expert_id >= num_experts
    ):
        raise ValueError(
            "expert_id 越界"
        )

    weights = np.maximum(
        count_values[
            :,
            expert_id,
        ],
        0.0,
    )

    if weights.sum() <= 1e-12:
        return (
            np.ones(
                num_clients,
                dtype=np.float64,
            )
            / num_clients
        )

    return (
        weights
        / weights.sum()
    )


def aggregate_state_dicts(
    client_state_dicts: Sequence[
        Mapping[str, torch.Tensor]
    ],
    client_num_samples: Sequence[int],
    cfg: Mapping,
    client_expert_counts: Optional[
        Sequence[Sequence[float]]
    ] = None,
) -> Dict[str, torch.Tensor]:
    agg_cfg = cfg[
        "aggregation"
    ]

    non_expert_method = str(
        agg_cfg[
            "non_expert_agg"
        ]
    )

    expert_method = str(
        agg_cfg[
            "expert_agg"
        ]
    )

    if expert_method == "meta_network":
        raise ValueError(
            "meta_network 不应调用 "
            "aggregate_state_dicts"
        )

    activation_methods = {
        "expert_activation_count_weighted",
        "activation_count_weighted",
    }

    num_clients = len(
        client_num_samples
    )

    non_expert_weights = (
        get_aggregation_weights(
            non_expert_method,
            client_num_samples,
        )
    )

    if (
        expert_method
        in activation_methods
    ):
        expert_weights = None
    else:
        expert_weights = (
            get_aggregation_weights(
                expert_method,
                client_num_samples,
            )
        )

    new_state_dict: Dict[
        str,
        torch.Tensor,
    ] = {}

    for name in (
        client_state_dicts[
            0
        ].keys()
    ):
        first_tensor = (
            client_state_dicts[
                0
            ][name]
        )

        if not torch.is_floating_point(
            first_tensor
        ):
            new_state_dict[name] = (
                first_tensor.clone()
            )
            continue

        if is_expert_param(name):
            if (
                expert_method
                in activation_methods
            ):
                expert_id = (
                    get_expert_id_from_name(
                        name
                    )
                )

                if expert_id is None:
                    raise ValueError(
                        "无法从参数名解析 expert id: "
                        f"{name}"
                    )

                weights = (
                    get_expert_activation_count_weights(
                        client_expert_counts=(
                            client_expert_counts
                        ),
                        expert_id=expert_id,
                        num_clients=num_clients,
                    )
                )

            else:
                weights = (
                    expert_weights
                )

        else:
            weights = (
                non_expert_weights
            )

        aggregated = torch.zeros_like(
            first_tensor
        )

        for (
            client_id,
            client_state,
        ) in enumerate(
            client_state_dicts
        ):
            aggregated += (
                client_state[name]
                * float(
                    weights[
                        client_id
                    ]
                )
            )

        new_state_dict[name] = (
            aggregated
        )

    return new_state_dict


def print_partition_summary(
    client_indices: Sequence[
        Sequence[int]
    ],
) -> None:
    print(
        "========== 客户端数据划分 =========="
    )

    for (
        client_id,
        indices,
    ) in enumerate(
        client_indices
    ):
        print(
            f"client {client_id:02d}: "
            f"{len(indices)} samples"
        )

    print(
        "==================================="
    )


# ------------------------------------------------------------
# 主训练流程
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径",
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    log_path = setup_logging(
        cfg=cfg,
        config_path=args.config,
    )

    print(
        f"配置文件路径: "
        f"{args.config}"
    )

    seed = int(
        cfg.get(
            "seed",
            1,
        )
    )

    repro_cfg = (
        get_reproducibility_cfg(
            cfg
        )
    )

    set_seed(
        seed,
        repro_cfg,
    )

    print(
        f"随机种子: {seed}"
    )

    device = get_device(
        cfg
    )

    print(
        f"使用设备: {device}"
    )

    log_reproducibility_environment(
        cfg=cfg,
        config_path=args.config,
        seed=seed,
        device=device,
        repro_cfg=repro_cfg,
    )

    (
        train_set,
        test_set,
    ) = build_datasets(cfg)

    (
        train_set,
        train_labels,
    ) = build_long_tail_train_set(
        train_set=train_set,
        cfg=cfg,
        seed=seed,
    )

    (
        server_val_set,
        final_test_set,
    ) = split_server_validation_from_test_set(
        test_set=test_set,
        cfg=cfg,
        seed=seed,
    )

    dataset_cfg = cfg[
        "dataset"
    ]

    client_indices = (
        dirichlet_partition(
            labels=train_labels,
            num_clients=int(
                dataset_cfg[
                    "num_clients"
                ]
            ),
            alpha=float(
                dataset_cfg[
                    "alpha"
                ]
            ),
            seed=seed,
            min_size=int(
                dataset_cfg.get(
                    "min_samples_per_client",
                    10,
                )
            ),
            max_attempts=int(
                dataset_cfg.get(
                    "partition_max_attempts",
                    100,
                )
            ),
        )
    )

    print_partition_summary(
        client_indices
    )

    client_datasets = (
        build_client_datasets(
            train_set=train_set,
            client_indices=client_indices,
            cfg=cfg,
            seed=seed,
        )
    )

    server_val_loader = (
        build_server_val_loader(
            server_val_set=(
                server_val_set
            ),
            cfg=cfg,
            device=device,
            base_seed=seed,
            repro_cfg=repro_cfg,
        )
    )

    test_loader = build_test_loader(
        test_set=final_test_set,
        cfg=cfg,
        device=device,
        base_seed=seed,
        repro_cfg=repro_cfg,
    )

    model_init_seed = stable_seed(
        seed,
        SEED_NAMESPACES[
            "model_init"
        ],
    )

    with isolated_rng(
        model_init_seed,
        device,
    ):
        global_model = (
            build_model(cfg)
        )

    global_model.to(device)

    print_trainable_param_stats(
        global_model
    )

    if bool(
        repro_cfg.get(
            "log_initial_state_hash",
            True,
        )
    ):
        log_only(
            "[REPRO] initial_model_seed="
            f"{model_init_seed}"
        )

        log_only(
            "[REPRO] initial_model_state_hash="
            f"{hash_state_dict(global_model.state_dict())}"
        )

    train_cfg = cfg[
        "train"
    ]

    model_cfg = cfg[
        "model"
    ]

    agg_cfg = cfg[
        "aggregation"
    ]

    meta_cfg = cfg.get(
        "meta",
        {},
    )

    performance_cfg = cfg.get(
        "performance",
        {},
    ) or {}

    reuse_local_model = bool(
        performance_cfg.get(
            "reuse_local_model",
            True,
        )
    )
    if not reuse_local_model:
        raise ValueError(
            "当前优化版要求 performance.reuse_local_model=true"
        )

    test_autocast = bool(
        performance_cfg.get(
            "test_autocast",
            True,
        )
    )
    test_autocast_dtype_name = str(
        performance_cfg.get(
            "test_autocast_dtype",
            "float16",
        )
    )
    test_autocast_dtype = resolve_autocast_dtype(
        test_autocast_dtype_name
    )

    meta_input_features = list(
        meta_cfg.get(
            "input_features",
            [
                "loss_z",
                "sample_ratio",
                "expert_freq",
            ],
        )
    )

    pre_probe_batches = int(
        meta_cfg.get(
            "pre_probe_batches",
            4,
        )
    )

    if pre_probe_batches <= 0:
        raise ValueError(
            "meta.pre_probe_batches 必须大于 0"
        )

    second_input_same_as_first = bool(
        meta_cfg.get(
            "second_input_same_as_first",
            True,
        )
    )

    num_clients = int(
        dataset_cfg[
            "num_clients"
        ]
    )

    rounds = int(
        train_cfg[
            "rounds"
        ]
    )

    clients_per_round = int(
        train_cfg[
            "clients_per_round"
        ]
    )

    if (
        clients_per_round
        > num_clients
    ):
        raise ValueError(
            "clients_per_round 不能大于 num_clients"
        )

    effective_num_workers = (
        get_effective_num_workers(
            cfg,
            repro_cfg,
        )
    )

    log_only(
        "[REPRO] requested_num_workers="
        f"{int(train_cfg.get('num_workers', 2))}"
    )

    log_only(
        "[REPRO] effective_num_workers="
        f"{effective_num_workers}"
    )

    local_model_seed = stable_seed(
        seed,
        SEED_NAMESPACES[
            "local_model_template"
        ],
    )

    with isolated_rng(
        local_model_seed,
        device,
    ):
        local_model = build_model(cfg)

    local_model.to(device)
    local_model.zero_grad(
        set_to_none=True
    )

    log_only(
        "[REPRO] local_model_template_seed="
        f"{local_model_seed}"
    )

    best_acc = 0.0

    test_acc_history: List[
        float
    ] = []

    expert_agg = str(
        agg_cfg[
            "expert_agg"
        ]
    )


    non_expert_agg = str(
        agg_cfg[
            "non_expert_agg"
        ]
    )

    meta_aggregator: Optional[
        MetaExpertAggregator
    ] = None

    if expert_agg == "meta_network":
        if server_val_loader is None:
            raise ValueError(
                "使用 meta_network 时，"
                "server.val_size 必须大于 0"
            )

        meta_init_seed = stable_seed(
            seed,
            SEED_NAMESPACES[
                "meta_init"
            ],
        )

        with isolated_rng(
            meta_init_seed,
            device,
        ):
            meta_aggregator = (
                MetaExpertAggregator(
                    num_experts=int(
                        model_cfg[
                            "num_experts"
                        ]
                    ),
                    device=device,
                    hidden_dim=int(
                        meta_cfg.get(
                            "hidden_dim",
                            32,
                        )
                    ),
                    lr=float(
                        meta_cfg.get(
                            "lr",
                            1e-3,
                        )
                    ),
                    meta_steps=int(
                        meta_cfg.get(
                            "steps",
                            1,
                        )
                    ),
                    max_val_batches=(
                        meta_cfg.get(
                            "max_val_batches",
                            None,
                        )
                    ),
                    train_log_path=(
                        log_path
                    ),
                    log_fn=log_only,
                    tau=float(
                        meta_cfg.get(
                            "tau",
                            1.0,
                        )
                    ),

                    tau_schedule=str(
                        meta_cfg.get(
                            "tau_schedule",
                            "constant",
                        )
                    ),

                    tau_start=float(
                        meta_cfg.get(
                            "tau_start",
                            meta_cfg.get(
                                "tau",
                                1.0,
                            ),
                        )
                    ),

                    tau_end=float(
                        meta_cfg.get(
                            "tau_end",
                            meta_cfg.get(
                                "tau_start",
                                meta_cfg.get(
                                    "tau",
                                    1.0,
                                ),
                            ),
                        )
                    ),

                    tau_anneal_rounds=(
                        None
                        if meta_cfg.get(
                            "tau_anneal_rounds"
                        ) is None
                        else int(
                            meta_cfg[
                                "tau_anneal_rounds"
                            ]
                        )
                    ),

                    total_rounds=int(
                        train_cfg.get(
                            "rounds",
                            1,
                        )
                    ),
                    active_mask=bool(
                        meta_cfg.get(
                            "active_mask",
                            False,
                        )
                    ),
                    active_threshold=float(
                        meta_cfg.get(
                            "active_threshold",
                            0.0,
                        )
                    ),
                    min_active_clients_per_expert=int(
                        meta_cfg.get(
                            "min_active_clients_per_expert",
                            2,
                        )
                    ),
                    input_features=(
                        meta_input_features
                    ),
                )
            )

        log_only(
            "[REPRO] meta_init_seed="
            f"{meta_init_seed}"
        )

        log_only(
            "[REPRO] meta_initial_state_hash="
            f"{hash_state_dict(meta_aggregator.meta_net.state_dict())}"
        )

        print(
            "已启用 expert_agg = meta_network"
        )

    else:
        print(
            f"使用普通 expert_agg = "
            f"{expert_agg}"
        )

    print(
        "========== 训练配置 =========="
    )
    print(
        f"rounds              : "
        f"{rounds}"
    )
    print(
        f"num_clients         : "
        f"{num_clients}"
    )
    print(
        f"clients_per_round   : "
        f"{clients_per_round}"
    )
    print(
        f"local_epochs        : "
        f"{train_cfg['local_epochs']}"
    )
    print(
        f"batch_size          : "
        f"{train_cfg['batch_size']}"
    )
    print(
        f"lr                  : "
        f"{train_cfg['lr']}"
    )
    print(
        f"momentum            : "
        f"{train_cfg.get('momentum', 0.9)}"
    )
    print(
        f"weight_decay        : "
        f"{train_cfg.get('weight_decay', 0.0005)}"
    )
    print(
        f"router_balance_w    : "
        f"{train_cfg.get('router_balance_weight', 0.0)}"
    )
    print(
        f"model.top_k         : "
        f"{model_cfg.get('top_k', 1)}"
    )
    print(
        f"non_expert_agg      : "
        f"{non_expert_agg}"
    )
    print(
        f"expert_agg          : "
        f"{expert_agg}"
    )

    if meta_aggregator is not None:
        print(
            "---------- Meta 配置 ----------"
        )
        print(
            f"meta.hidden_dim     : "
            f"{meta_cfg.get('hidden_dim', 32)}"
        )
        print(
            f"meta.lr             : "
            f"{meta_cfg.get('lr', 1e-3)}"
        )
        print(
            f"meta.steps          : "
            f"{meta_cfg.get('steps', 1)}"
        )
        resolved_tau_anneal_rounds = (
            int(
                train_cfg.get(
                    "rounds",
                    1,
                )
            )
            if meta_cfg.get(
                "tau_anneal_rounds"
            ) is None
            else int(
                meta_cfg[
                    "tau_anneal_rounds"
                ]
            )
        )

        print(
            f"meta.tau_schedule    : "
            f"{meta_cfg.get('tau_schedule', 'constant')}"
        )

        print(
            f"meta.tau_start       : "
            f"{meta_cfg.get('tau_start', meta_cfg.get('tau', 1.0))}"
        )

        print(
            f"meta.tau_end         : "
            f"{meta_cfg.get('tau_end', meta_cfg.get('tau_start', meta_cfg.get('tau', 1.0)))}"
        )

        print(
            f"meta.tau_anneal_rounds: "
            f"{resolved_tau_anneal_rounds}"
        )
        print(
            f"max_val_batches     : "
            f"{meta_cfg.get('max_val_batches', None)}"
        )
        print(
            f"pre_probe_batches   : "
            f"{pre_probe_batches}"
        )
        print(
            f"pre_probe_batch_size: "
            f"{meta_cfg.get('pre_probe_batch_size', train_cfg['batch_size'])}"
        )
        print(
            f"second_same_as_first: "
            f"{second_input_same_as_first}"
        )
        print(
            f"meta.active_mask    : "
            f"{meta_cfg.get('active_mask', False)}"
        )
        print(
            f"active_threshold    : "
            f"{meta_cfg.get('active_threshold', 0.0)}"
        )
        print(
            f"min_active_clients : "
            f"{meta_cfg.get('min_active_clients_per_expert', 2)}"
        )
        print(
            "meta.input_features : "
            f"[{', '.join(meta_aggregator.input_feature_names)}]"
        )

    print(
        "---------- 性能优化 ----------"
    )
    print(
        f"reuse_local_model   : "
        f"{reuse_local_model}"
    )
    print(
        "empty_cache/client  : False"
    )
    print(
        f"test_autocast       : "
        f"{test_autocast}"
    )
    print(
        f"test_autocast_dtype : "
        f"{test_autocast_dtype_name}"
    )

    print(
        "=============================="
    )

    for round_id in range(
        1,
        rounds + 1,
    ):
        global_state_dict = {
            name: (
                tensor
                .detach()
                .cpu()
                .clone()
            )
            for name, tensor
            in global_model.state_dict().items()
        }

        selected_clients = list(
            range(
                clients_per_round
            )
        )

        client_state_dicts = []
        client_num_samples = []
        client_losses = []
        client_expert_freqs = []
        client_expert_counts = []

        pre_client_losses = []
        pre_client_expert_freqs = []
        pre_client_expert_counts = []

        for client_id in (
            selected_clients
        ):
            client_loader = (
                build_client_loader_for_round(
                    client_dataset=(
                        client_datasets[
                            client_id
                        ]
                    ),
                    cfg=cfg,
                    device=device,
                    base_seed=seed,
                    round_id=round_id,
                    client_id=client_id,
                    repro_cfg=repro_cfg,
                )
            )

            probe_loader = None

            if (
                expert_agg
                == "meta_network"
            ):
                probe_loader = (
                    build_pre_probe_loader_for_round(
                        client_dataset=(
                            client_datasets[
                                client_id
                            ]
                        ),
                        cfg=cfg,
                        device=device,
                        base_seed=seed,
                        round_id=round_id,
                        client_id=client_id,
                        repro_cfg=(
                            repro_cfg
                        ),
                    )
                )

            (
                local_state_dict,
                num_samples,
                avg_loss,
                expert_freq,
                expert_count_values,
                pre_loss,
                pre_expert_freq,
                pre_expert_count_values,
                actual_probe_batches,
                actual_probe_samples,
            ) = local_train(
                model=local_model,
                global_state_dict=(
                    global_state_dict
                ),
                train_loader=(
                    client_loader
                ),
                probe_loader=(
                    probe_loader
                ),
                pre_probe_batches=(
                    pre_probe_batches
                ),
                cfg=cfg,
                device=device,
                base_seed=seed,
                round_id=round_id,
                client_id=client_id,
            )

            if (
                expert_agg
                == "meta_network"
            ):
                log_only(
                    f"[META_PRE_PROBE] "
                    f"round={round_id} "
                    f"client={client_id} "
                    f"batches="
                    f"{actual_probe_batches} "
                    f"samples="
                    f"{actual_probe_samples} "
                    f"loss="
                    f"{pre_loss:.10f} "
                    f"expert_counts="
                    f"{pre_expert_count_values} "
                    f"expert_freq="
                    f"{pre_expert_freq}"
                )

            if (
                round_id == 1
                and bool(
                    repro_cfg.get(
                        "log_first_round_client_hashes",
                        True,
                    )
                )
            ):
                loader_seed = stable_seed(
                    seed,
                    SEED_NAMESPACES[
                        "client_loader"
                    ],
                    round_id,
                    client_id,
                )

                local_seed = stable_seed(
                    seed,
                    SEED_NAMESPACES[
                        "local_train"
                    ],
                    round_id,
                    client_id,
                )

                log_only(
                    f"[REPRO_CLIENT] "
                    f"round={round_id} "
                    f"client={client_id} "
                    f"loader_seed="
                    f"{loader_seed} "
                    f"local_seed="
                    f"{local_seed} "
                    f"loss="
                    f"{avg_loss:.12f} "
                    f"expert_counts="
                    f"{expert_count_values} "
                    f"state_hash="
                    f"{hash_state_dict(local_state_dict)}"
                )

            client_state_dicts.append(
                local_state_dict
            )

            client_num_samples.append(
                num_samples
            )

            client_losses.append(
                avg_loss
            )

            client_expert_freqs.append(
                expert_freq
            )

            client_expert_counts.append(
                expert_count_values
            )


            pre_client_losses.append(
                pre_loss
            )

            pre_client_expert_freqs.append(
                pre_expert_freq
            )

            pre_client_expert_counts.append(
                pre_expert_count_values
            )


        meta_info = None

        if expert_agg == "meta_network":
            if meta_aggregator is None:
                raise RuntimeError(
                    "meta_aggregator 未初始化"
                )

            meta_round_seed = stable_seed(
                seed,
                SEED_NAMESPACES[
                    "meta_round"
                ],
                round_id,
            )

            with isolated_rng(
                meta_round_seed,
                device,
            ):
                (
                    new_global_state_dict,
                    meta_info,
                ) = meta_aggregator.aggregate(
                    model=global_model,
                    client_state_dicts=(
                        client_state_dicts
                    ),
                    client_num_samples=(
                        client_num_samples
                    ),
                    client_losses=(
                        client_losses
                    ),
                    client_expert_freqs=(
                        client_expert_freqs
                    ),
                    client_expert_counts=(
                        client_expert_counts
                    ),
                    pre_client_losses=(
                        pre_client_losses
                    ),
                    pre_client_expert_freqs=(
                        pre_client_expert_freqs
                    ),
                    pre_client_expert_counts=(
                        pre_client_expert_counts
                    ),
                    val_loader=(
                        server_val_loader
                    ),
                    non_expert_agg=(
                        non_expert_agg
                    ),
                    second_input_same_as_first=(
                        second_input_same_as_first
                    ),
                )

            log_only(
                f"[REPRO_META] "
                f"round={round_id} "
                f"seed={meta_round_seed}"
            )

        else:
            new_global_state_dict = (
                aggregate_state_dicts(
                    client_state_dicts=(
                        client_state_dicts
                    ),
                    client_num_samples=(
                        client_num_samples
                    ),
                    cfg=cfg,
                    client_expert_counts=(
                        client_expert_counts
                    ),
                )
            )

        if bool(
            repro_cfg.get(
                "log_global_state_hash_each_round",
                True,
            )
        ):
            log_only(
                f"[REPRO_GLOBAL] "
                f"round={round_id} "
                f"state_hash="
                f"{hash_state_dict(new_global_state_dict)}"
            )

        global_model.load_state_dict(
            new_global_state_dict,
            strict=True,
        )

        evaluation_seed = stable_seed(
            seed,
            SEED_NAMESPACES[
                "evaluation"
            ],
            round_id,
        )

        with isolated_rng(
            evaluation_seed,
            device,
        ):
            (
                test_acc,
                test_loss,
            ) = evaluate(
                model=global_model,
                test_loader=test_loader,
                device=device,
                use_autocast=(
                    test_autocast
                ),
                autocast_dtype=(
                    test_autocast_dtype
                ),
            )

        test_acc_history.append(
            float(test_acc)
        )

        best_acc = max(
            best_acc,
            test_acc,
        )

        avg_client_loss = float(
            np.mean(
                client_losses
            )
        )

        if meta_info is not None:
            meta_loss = float(
                meta_info.get(
                    "meta_loss"
                )
            )

            print(
                f"Round {round_id:03d} | "
                f"client_loss="
                f"{avg_client_loss:.4f} | "
                f"meta_loss="
                f"{meta_loss:.4f} | "
                f"test_loss="
                f"{test_loss:.4f} | "
                f"acc="
                f"{test_acc:.2f}% | "
                f"best="
                f"{best_acc:.2f}%"
            )

        else:
            print(
                f"Round {round_id:03d} | "
                f"client_loss="
                f"{avg_client_loss:.4f} | "
                f"test_loss="
                f"{test_loss:.4f} | "
                f"acc="
                f"{test_acc:.2f}% | "
                f"best="
                f"{best_acc:.2f}%"
            )

    print("=" * 80)
    print(
        f"训练结束时间: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(
        f"最终 best_acc: "
        f"{best_acc:.2f}%"
    )

    if test_acc_history:
        window = min(
            20,
            len(
                test_acc_history
            ),
        )

        start_round = (
            len(
                test_acc_history
            )
            - window
            + 1
        )

        mean_acc = float(
            np.mean(
                test_acc_history[
                    -window:
                ]
            )
        )

        print(
            f"后{window}轮准确率均值: "
            f"{mean_acc:.4f}% "
            f"(Rounds "
            f"{start_round}-"
            f"{len(test_acc_history)})"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
