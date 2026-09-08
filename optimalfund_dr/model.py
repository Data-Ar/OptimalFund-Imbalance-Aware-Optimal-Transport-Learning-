"""Classification backbone and standard losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .config import Config, cfg


class DRModel(nn.Module):
    """timm backbone with a linear classification head."""

    def __init__(self, backbone: str, num_classes: int, config: Config | None = None):
        super().__init__()
        self.config = config or cfg
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_feat: bool = False):
        feat = self.backbone(x)
        if self.config.use_feature_norm:
            feat = F.normalize(feat, p=2, dim=1)
        logits = self.head(feat)
        return (logits, feat) if return_feat else logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def get_criterion(config: Config | None = None):
    config = config or cfg
    return FocalLoss(config.focal_gamma) if config.use_focal_loss else nn.CrossEntropyLoss()
