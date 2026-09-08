"""Datasets, transforms, and dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import Config, cfg, phone_root_for_severity

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_tf(img_size: int, train: bool):
    """ImageNet-normalized train/eval transforms."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class DRFolderDataset(Dataset):
    """Folder dataset with ``{split}/{class_id}/image`` layout."""

    def __init__(
        self, root: str, split: str, transform=None, num_classes: int = 5
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        base = self.root / split
        if not base.exists():
            raise FileNotFoundError(f"Missing split folder: {base}")

        samples = []
        for y in range(num_classes):
            cls_dir = base / str(y)
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    samples.append((str(p), y))

        if len(samples) == 0:
            raise RuntimeError(f"No images found under {base}")

        self.samples = samples
        self.labels = np.array([y for _, y in samples], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(y), path


def make_weighted_sampler(
    labels: np.ndarray,
    num_classes: int,
    generator: torch.Generator | None = None,
):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = (1.0 / counts)[labels]
    return WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def make_loader(
    ds: Dataset,
    train: bool,
    use_weighted_sampler: bool,
    generator: torch.Generator | None = None,
    config: Config | None = None,
):
    config = config or cfg
    sampler = None
    shuffle = False
    if train:
        if use_weighted_sampler:
            sampler = make_weighted_sampler(ds.labels, config.num_classes, generator)
        else:
            shuffle = True

    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=train,
        persistent_workers=(config.persistent_workers and config.num_workers > 0),
        generator=generator,
    )


@dataclass
class DataBundle:
    """Initialized hospital and phone datasets/loaders for one experiment."""

    ds_h_train: DRFolderDataset
    ds_h_val: DRFolderDataset
    ds_h_test: DRFolderDataset
    ds_p_train: DRFolderDataset
    ds_p_val: DRFolderDataset
    ds_p_test: DRFolderDataset
    ld_p_val: DataLoader
    ld_h_test: DataLoader
    ld_p_test: DataLoader
    phone_test_loaders: dict


def initialize_data(config: Config | None = None) -> DataBundle:
    """Load hospital and phone datasets and build evaluation loaders."""
    config = config or cfg
    tf_train = build_tf(config.img_size, train=True)
    tf_eval = build_tf(config.img_size, train=False)

    ds_h_train = DRFolderDataset(
        config.hosp_root, "train", transform=tf_train, num_classes=config.num_classes
    )
    ds_h_val = DRFolderDataset(
        config.hosp_root, "val", transform=tf_eval, num_classes=config.num_classes
    )
    ds_h_test = DRFolderDataset(
        config.hosp_root, "test", transform=tf_eval, num_classes=config.num_classes
    )
    ds_p_train = DRFolderDataset(
        config.phone_root_clean,
        "train",
        transform=tf_train,
        num_classes=config.num_classes,
    )
    ds_p_val = DRFolderDataset(
        config.phone_root_clean,
        "val",
        transform=tf_eval,
        num_classes=config.num_classes,
    )
    ds_p_test = DRFolderDataset(
        config.phone_root_clean,
        "test",
        transform=tf_eval,
        num_classes=config.num_classes,
    )

    ld_p_val = make_loader(ds_p_val, train=False, use_weighted_sampler=False, config=config)
    ld_h_test = make_loader(ds_h_test, train=False, use_weighted_sampler=False, config=config)
    ld_p_test = make_loader(ds_p_test, train=False, use_weighted_sampler=False, config=config)

    phone_test_loaders = {}
    for sev in config.phone_eval_severities:
        sev_root = phone_root_for_severity(config, sev)
        ds_sev = DRFolderDataset(
            sev_root, "test", transform=tf_eval, num_classes=config.num_classes
        )
        ld_sev = make_loader(
            ds_sev, train=False, use_weighted_sampler=False, config=config
        )
        phone_test_loaders[sev] = {
            "root": sev_root,
            "dataset": ds_sev,
            "loader": ld_sev,
        }

    print("Hospital images:", len(ds_h_train), len(ds_h_val), len(ds_h_test))
    print("Phone    images:", len(ds_p_train), len(ds_p_val), len(ds_p_test))
    print(
        "Phone train class counts:",
        np.bincount(ds_p_train.labels, minlength=config.num_classes).tolist(),
    )

    return DataBundle(
        ds_h_train=ds_h_train,
        ds_h_val=ds_h_val,
        ds_h_test=ds_h_test,
        ds_p_train=ds_p_train,
        ds_p_val=ds_p_val,
        ds_p_test=ds_p_test,
        ld_p_val=ld_p_val,
        ld_h_test=ld_h_test,
        ld_p_test=ld_p_test,
        phone_test_loaders=phone_test_loaders,
    )


def build_run_train_loaders(
    data: DataBundle,
    seed: int,
    generator_states: dict[str, torch.Tensor] | None = None,
    config: Config | None = None,
):
    """Build seed-specific training loaders, optionally restoring generator state."""
    config = config or cfg

    hospital_generator = torch.Generator()
    phone_generator = torch.Generator()
    hospital_generator.manual_seed(seed + 10_001)
    phone_generator.manual_seed(seed + 20_001)

    if generator_states is not None:
        if "hospital" in generator_states:
            hospital_generator.set_state(generator_states["hospital"])
        if "phone" in generator_states:
            phone_generator.set_state(generator_states["phone"])

    hospital_loader = make_loader(
        data.ds_h_train,
        train=True,
        use_weighted_sampler=False,
        generator=hospital_generator,
        config=config,
    )
    phone_loader = make_loader(
        data.ds_p_train,
        train=True,
        use_weighted_sampler=config.use_phone_weighted_sampler,
        generator=phone_generator,
        config=config,
    )
    generators = {"hospital": hospital_generator, "phone": phone_generator}
    print("Train batches | hospital:", len(hospital_loader), "phone:", len(phone_loader))
    return hospital_loader, phone_loader, generators
