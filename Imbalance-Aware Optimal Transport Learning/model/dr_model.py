import torch.nn as nn
import torch.nn.functional as F
import timm


class DRModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, use_feature_norm: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)
        self.use_feature_norm = use_feature_norm

    def forward(self, x, return_feat: bool = False):
        feat = self.backbone(x)
        if self.use_feature_norm:
            feat = F.normalize(feat, p=2, dim=1)
        logits = self.head(feat)
        return (logits, feat) if return_feat else logits
