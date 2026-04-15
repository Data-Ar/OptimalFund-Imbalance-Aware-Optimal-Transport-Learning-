import math

import torch
import torch.nn.functional as F


def prototype_ot_loss(feat_h, y_h, feat_p, y_p, num_classes: int):
    loss = feat_h.new_tensor(0.0)
    for c in range(num_classes):
        mh = feat_h[y_h == c]
        mp = feat_p[y_p == c]
        if mh.shape[0] > 0 and mp.shape[0] > 0:
            loss = loss + F.mse_loss(mh.mean(0), mp.mean(0))
    return loss


def pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return ((A*A).sum(1, keepdim=True) + (B*B).sum(1, keepdim=True).T - 2*(A @ B.T)).clamp_min(0.0)


def sinkhorn_log(C: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
    n, m = C.shape
    log_a = -math.log(n) * torch.ones(n, device=C.device)
    log_b = -math.log(m) * torch.ones(m, device=C.device)
    logK = -C / eps

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iters):
        u = log_a - torch.logsumexp(logK + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(logK.T + u.unsqueeze(0), dim=1)

    logT = u.unsqueeze(1) + logK + v.unsqueeze(0)
    T = torch.exp(logT)
    return torch.sum(T * C)


def sinkhorn_ot_loss(feat_h, feat_p, eps: float, iters: int):
    C = pairwise_sq_dists(feat_h, feat_p)
    return sinkhorn_log(C, eps=eps, iters=iters)


def class_cond_sinkhorn_ot_loss(feat_h, y_h, feat_p, y_p, num_classes: int, eps: float, iters: int):
    loss = feat_h.new_tensor(0.0)
    for c in range(num_classes):
        mh = feat_h[y_h == c]
        mp = feat_p[y_p == c]
        if mh.shape[0] > 0 and mp.shape[0] > 0:
            C = pairwise_sq_dists(mh, mp)
            loss = loss + sinkhorn_log(C, eps=eps, iters=iters)
    return loss


def compute_ot_loss(feat_h, y_h, feat_p, y_p, cfg):
    if cfg.ot_mode == "none":
        return feat_h.new_tensor(0.0)
    if cfg.ot_mode == "prototype":
        return prototype_ot_loss(feat_h, y_h, feat_p, y_p, cfg.num_classes)
    if cfg.ot_mode == "sinkhorn":
        return sinkhorn_ot_loss(feat_h, feat_p, cfg.sinkhorn_eps, cfg.sinkhorn_iters)
    if cfg.ot_mode == "class_sinkhorn":
        return class_cond_sinkhorn_ot_loss(feat_h, y_h, feat_p, y_p, cfg.num_classes, cfg.sinkhorn_eps, cfg.sinkhorn_iters)
    raise ValueError(f"Unknown ot_mode: {cfg.ot_mode}")
