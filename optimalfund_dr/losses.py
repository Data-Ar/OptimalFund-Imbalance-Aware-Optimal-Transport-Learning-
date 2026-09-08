"""Optimal-transport alignment losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import Config, cfg


def prototype_ot_loss(feat_h, y_h, feat_p, y_p, num_classes: int):
    """Align class-wise mean embeddings between hospital and phone batches."""
    loss = feat_h.new_tensor(0.0)
    for c in range(num_classes):
        mh = feat_h[y_h == c]
        mp = feat_p[y_p == c]
        if mh.shape[0] > 0 and mp.shape[0] > 0:
            loss = loss + F.mse_loss(mh.mean(0), mp.mean(0))
    return loss


def pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        (a * a).sum(1, keepdim=True) + (b * b).sum(1, keepdim=True).T - 2 * (a @ b.T)
    ).clamp_min(0.0)


def sinkhorn_log(cost: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
    n, m = cost.shape
    log_a = -math.log(n) * torch.ones(n, device=cost.device)
    log_b = -math.log(m) * torch.ones(m, device=cost.device)
    log_k = -cost / eps

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k.T + u.unsqueeze(0), dim=1)

    log_t = u.unsqueeze(1) + log_k + v.unsqueeze(0)
    transport = torch.exp(log_t)
    return torch.sum(transport * cost)


def sinkhorn_ot_loss(feat_h, feat_p, eps: float, iters: int):
    cost = pairwise_sq_dists(feat_h, feat_p)
    return sinkhorn_log(cost, eps=eps, iters=iters)


def class_cond_sinkhorn_ot_loss(
    feat_h, y_h, feat_p, y_p, num_classes: int, eps: float, iters: int
):
    loss = feat_h.new_tensor(0.0)
    for c in range(num_classes):
        mh = feat_h[y_h == c]
        mp = feat_p[y_p == c]
        if mh.shape[0] > 0 and mp.shape[0] > 0:
            cost = pairwise_sq_dists(mh, mp)
            loss = loss + sinkhorn_log(cost, eps=eps, iters=iters)
    return loss


def compute_ot_loss(feat_h, y_h, feat_p, y_p, config: Config | None = None):
    config = config or cfg
    if config.ot_mode == "none":
        return feat_h.new_tensor(0.0)
    if config.ot_mode == "prototype":
        return prototype_ot_loss(feat_h, y_h, feat_p, y_p, config.num_classes)
    if config.ot_mode == "sinkhorn":
        return sinkhorn_ot_loss(
            feat_h, feat_p, config.sinkhorn_eps, config.sinkhorn_iters
        )
    if config.ot_mode == "class_sinkhorn":
        return class_cond_sinkhorn_ot_loss(
            feat_h,
            y_h,
            feat_p,
            y_p,
            config.num_classes,
            config.sinkhorn_eps,
            config.sinkhorn_iters,
        )
    raise ValueError(f"Unknown ot_mode: {config.ot_mode}")
