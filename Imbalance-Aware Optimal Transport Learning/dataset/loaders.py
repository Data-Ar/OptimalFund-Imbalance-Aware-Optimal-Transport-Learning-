import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

from util.transforms import build_tf
from util.config import phone_root_for_severity
from dataset.dr_dataset import DRFolderDataset


def make_weighted_sampler(labels: np.ndarray, num_classes: int):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = (1.0 / counts)[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(labels),
                                 replacement=True)


def make_loader(ds: Dataset, train: bool, use_weighted_sampler: bool, cfg):
    sampler = None
    shuffle = False
    if train:
        if use_weighted_sampler:
            sampler = make_weighted_sampler(ds.labels, cfg.num_classes)
        else:
            shuffle = True

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=train,
        persistent_workers=(cfg.num_workers > 0),
    )


def make_phone_test_loader(phone_root: str, cfg, tf_eval):
    ds = DRFolderDataset(phone_root, "test", transform=tf_eval, num_classes=cfg.num_classes)
    ld = make_loader(ds, train=False, use_weighted_sampler=False, cfg=cfg)
    return ds, ld


def build_data_objects(cfg):
    tf_train = build_tf(cfg.img_size, train=True)
    tf_eval  = build_tf(cfg.img_size, train=False)

    ds_h_train = DRFolderDataset(cfg.hosp_root, 'train', transform=tf_train, num_classes=cfg.num_classes)
    ds_h_val   = DRFolderDataset(cfg.hosp_root, 'val',   transform=tf_eval,  num_classes=cfg.num_classes)
    ds_h_test  = DRFolderDataset(cfg.hosp_root, 'test',  transform=tf_eval,  num_classes=cfg.num_classes)

    ds_p_train = DRFolderDataset(cfg.phone_root_clean, 'train', transform=tf_train, num_classes=cfg.num_classes)
    ds_p_val   = DRFolderDataset(cfg.phone_root_clean, 'val',   transform=tf_eval,  num_classes=cfg.num_classes)
    ds_p_test  = DRFolderDataset(cfg.phone_root_clean, 'test',  transform=tf_eval,  num_classes=cfg.num_classes)

    ld_h_train = make_loader(ds_h_train, train=True,  use_weighted_sampler=False, cfg=cfg)
    ld_p_train = make_loader(ds_p_train, train=True,  use_weighted_sampler=cfg.use_phone_weighted_sampler, cfg=cfg)

    ld_p_val   = make_loader(ds_p_val,   train=False, use_weighted_sampler=False, cfg=cfg)
    ld_h_test  = make_loader(ds_h_test,  train=False, use_weighted_sampler=False, cfg=cfg)
    ld_p_test  = make_loader(ds_p_test,  train=False, use_weighted_sampler=False, cfg=cfg)

    phone_test_loaders = {}
    for sev in cfg.phone_eval_severities:
        sev_root = phone_root_for_severity(cfg, sev)
        ds_sev = DRFolderDataset(sev_root, 'test', transform=tf_eval, num_classes=cfg.num_classes)
        ld_sev = make_loader(ds_sev, train=False, use_weighted_sampler=False, cfg=cfg)
        phone_test_loaders[sev] = {
            'root': sev_root,
            'dataset': ds_sev,
            'loader': ld_sev,
        }

    return {
        'tf_train': tf_train,
        'tf_eval': tf_eval,
        'ds_h_train': ds_h_train,
        'ds_h_val': ds_h_val,
        'ds_h_test': ds_h_test,
        'ds_p_train': ds_p_train,
        'ds_p_val': ds_p_val,
        'ds_p_test': ds_p_test,
        'ld_h_train': ld_h_train,
        'ld_p_train': ld_p_train,
        'ld_p_val': ld_p_val,
        'ld_h_test': ld_h_test,
        'ld_p_test': ld_p_test,
        'phone_test_loaders': phone_test_loaders,
    }
