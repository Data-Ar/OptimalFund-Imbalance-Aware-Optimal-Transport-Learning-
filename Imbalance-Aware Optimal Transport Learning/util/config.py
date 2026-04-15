from dataclasses import dataclass
from pathlib import Path
import os
import torch


class Config:
    hosp_root: str = "/medailab/medailab/shilab/DR_Detection_45300/split_dataset"

    # CLEAN phone root (original)
    phone_root_clean: str = "/medailab/medailab/shilab/mobile_BRSET/split_dataset"

    # MMS root that contains mild/moderate/severe folders
    phone_root_mms: str = "/medailab/medailab/shilab/mobile_BRSET/split_dataset_mms"

    # Which phone severities to evaluate on
    phone_eval_severities: tuple = ("clean", "mild", "moderate", "severe")

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True

    num_classes: int = 5
    use_phone_weighted_sampler: bool = True

    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    ot_mode: str = "class_sinkhorn"     # none | prototype | sinkhorn | class_sinkhorn
    ot_lambda: float = 0.3
    use_feature_norm: bool = True
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 50

    ref_classes: tuple = (3,4)
    target_spec: float = 0.90
    steps_multiplier: int = 2   
    n_boot: int = 2000
    bootstrap_all_severities: bool = True
        
    backbones: tuple = ("resnet50", "mobilevit_s", "efficientnet_b0", "mobileone_s4")
    seeds: tuple = (42, 43, 44)

    out_dir: str = "./runs_dr_full_varies_boostrap_final_03"


def phone_root_for_severity(cfg: Config, severity: str) -> str:
    """Return phone dataset root for a given severity."""
    if severity == "clean":
        return cfg.phone_root_clean
    return str(Path(cfg.phone_root_mms) / severity)


cfg = Config()
# keep legacy name for existing code that expects cfg.phone_root
cfg.phone_root = cfg.phone_root_clean
os.makedirs(cfg.out_dir, exist_ok=True)
