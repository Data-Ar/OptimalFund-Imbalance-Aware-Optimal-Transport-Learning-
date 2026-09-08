"""Experiment configuration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Config:
    """Hyperparameters and dataset locations for a training sweep."""

    hosp_root: str = ""
    phone_root_clean: str = ""
    phone_root_mms: str = ""
    phone_root: str = ""

    phone_eval_severities: tuple[str, ...] = ("clean", "mild", "moderate", "severe")

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    epochs: int = 30
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    deterministic_training: bool = True

    num_classes: int = 5
    use_phone_weighted_sampler: bool = True

    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    ot_mode: str = "class_sinkhorn"  # none | prototype | sinkhorn | class_sinkhorn
    ot_lambda: float = 0.3
    use_feature_norm: bool = True
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 50

    ref_classes: tuple[int, ...] = (3, 4)
    target_spec: float = 0.90
    steps_multiplier: int = 2
    n_boot: int = 2000
    bootstrap_all_severities: bool = True

    backbones: tuple[str, ...] = (
        "resnet50",
        "mobilevit_s",
        "efficientnet_b0",
        "mobileone_s4",
    )
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    ot_modes: tuple[str, ...] = ("none", "prototype", "sinkhorn", "class_sinkhorn")

    resume: bool = False
    overwrite_existing: bool = False
    save_predictions: bool = True
    persistent_workers: bool = False

    out_dir: str = "./runs_dr_5seeds_resumable"


cfg = Config()


def phone_root_for_severity(config: Config, severity: str) -> str:
    """Return the phone dataset root for a given degradation severity."""
    if severity == "clean":
        return config.phone_root_clean
    return str(Path(config.phone_root_mms) / severity)


def set_seed(seed: int, config: Config | None = None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    config = config or cfg
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = config.deterministic_training
    torch.backends.cudnn.benchmark = not config.deterministic_training


def required_dataset_paths(config: Config) -> list[str]:
    """Return dataset path fields that must be set before training."""
    missing = []
    if not config.hosp_root:
        missing.append("hosp_root")
    if not config.phone_root_clean:
        missing.append("phone_root_clean")
    if not config.phone_root_mms:
        missing.append("phone_root_mms")
    return missing
