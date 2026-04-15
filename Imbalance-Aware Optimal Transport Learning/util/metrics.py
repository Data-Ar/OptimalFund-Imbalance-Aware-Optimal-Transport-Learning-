import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


def compute_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, num_classes: int):
    macro_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    per_class_auc = roc_auc_score(y_true, probs, multi_class="ovr", average=None)

    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))

    sens, spec = [], []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens.append(TP / (TP + FN + 1e-8))
        spec.append(TN / (TN + FP + 1e-8))

    return float(macro_auc), per_class_auc.astype(float), np.array(sens, float), np.array(spec, float), cm


def sens_at_spec_ovr(y_true: np.ndarray, probs: np.ndarray, num_classes: int, target_spec: float):
    sens_list, thr_list = [], []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        p_c = probs[:, c]
        fpr, tpr, thr = roc_curve(y_bin, p_c)
        spec = 1.0 - fpr
        valid = np.where(spec >= target_spec)[0]
        if len(valid) == 0:
            sens_list.append(float("nan"))
            thr_list.append(float("nan"))
        else:
            idx = valid[np.argmax(tpr[valid])]
            sens_list.append(float(tpr[idx]))
            thr_list.append(float(thr[idx]))
    return np.array(sens_list, float), np.array(thr_list, float)


def choose_ref_threshold_at_spec(y_true_mc: np.ndarray, probs_mc: np.ndarray, ref_classes, target_spec: float):
    y_ref = np.isin(y_true_mc, list(ref_classes)).astype(int)
    p_ref = probs_mc[:, list(ref_classes)].sum(axis=1)
    fpr, tpr, thr = roc_curve(y_ref, p_ref)
    spec = 1.0 - fpr
    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = valid[np.argmax(tpr[valid])]
    return float(tpr[idx]), float(spec[idx]), float(thr[idx])


def refsens_spec_at_threshold(y_true_mc: np.ndarray, probs_mc: np.ndarray, thr: float, ref_classes):
    y_ref = np.isin(y_true_mc, list(ref_classes)).astype(int)
    p_ref = probs_mc[:, list(ref_classes)].sum(axis=1)
    pred = (p_ref >= thr).astype(int)

    TP = np.sum((pred == 1) & (y_ref == 1))
    FN = np.sum((pred == 0) & (y_ref == 1))
    FP = np.sum((pred == 1) & (y_ref == 0))
    TN = np.sum((pred == 0) & (y_ref == 0))

    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)
    return float(sens), float(spec)


def bootstrap_ci_image_level(
    y_true_mc,
    probs_mc,
    num_classes,
    ref_thr,
    ref_classes,
    n_boot=1000,
    alpha=0.95,
    seed=42,
):
    rng = np.random.default_rng(seed)
    n = len(y_true_mc)

    auc_vals = []
    sens_vals = []
    spec_vals = []

    auc_fail = 0
    sensspec_fail = 0

    lo_pct = (1.0 - alpha) / 2.0 * 100.0
    hi_pct = (1.0 + alpha) / 2.0 * 100.0

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true_mc[idx]
        p_b = probs_mc[idx]

        try:
            auc_b, _, _, _, _ = compute_metrics_from_probs(y_b, p_b, num_classes)
            if np.isfinite(auc_b):
                auc_vals.append(float(auc_b))
            else:
                auc_fail += 1
        except Exception:
            auc_fail += 1

        try:
            sens_b, spec_b = refsens_spec_at_threshold(y_b, p_b, ref_thr, ref_classes)
            if np.isfinite(sens_b):
                sens_vals.append(float(sens_b))
            else:
                sensspec_fail += 1
            if np.isfinite(spec_b):
                spec_vals.append(float(spec_b))
            else:
                sensspec_fail += 1
        except Exception:
            sensspec_fail += 1

    def summarize(vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            float(np.mean(vals)),
            float(np.percentile(vals, lo_pct)),
            float(np.percentile(vals, hi_pct)),
        )

    auc_mean, auc_lo, auc_hi = summarize(auc_vals)
    sens_mean, sens_lo, sens_hi = summarize(sens_vals)
    spec_mean, spec_lo, spec_hi = summarize(spec_vals)

    print(
        f"[BOOTSTRAP] valid_auc={len(auc_vals)}/{n_boot}, "
        f"valid_sens={len(sens_vals)}/{n_boot}, "
        f"valid_spec={len(spec_vals)}/{n_boot}"
    )

    return {
        "auc_boot_mean": auc_mean,
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "ref_sens_boot_mean": sens_mean,
        "ref_sens_ci_low": sens_lo,
        "ref_sens_ci_high": sens_hi,
        "ref_spec_boot_mean": spec_mean,
        "ref_spec_ci_low": spec_lo,
        "ref_spec_ci_high": spec_hi,
        "auc_boot_valid": len(auc_vals),
        "ref_sens_boot_valid": len(sens_vals),
        "ref_spec_boot_valid": len(spec_vals),
        "n_boot": n_boot,
    }


def collect_probs_labels(model: nn.Module, loader: DataLoader):
    model.eval()
    probs_list, labels_list = [], []
    for x, y, _ in loader:
        x = x.to(cfg.device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(y.numpy())
    return np.vstack(probs_list), np.concatenate(labels_list)


def evaluate(model: nn.Module, loader: DataLoader):
    probs, labels = collect_probs_labels(model, loader)

    try:
        macro_auc, per_class_auc, sens, spec, cm = compute_metrics_from_probs(
            labels, probs, cfg.num_classes
        )
    except Exception as e:
        print(f"[WARN] compute_metrics_from_probs failed: {e}")
        macro_auc = float("nan")
        per_class_auc = [float("nan")] * cfg.num_classes
        sens = [float("nan")] * cfg.num_classes
        spec = [float("nan")] * cfg.num_classes
        cm = np.full((cfg.num_classes, cfg.num_classes), np.nan)

    try:
        sens90, thr90 = sens_at_spec_ovr(
            labels, probs, cfg.num_classes, target_spec=cfg.target_spec
        )
    except Exception as e:
        print(f"[WARN] sens_at_spec_ovr failed: {e}")
        sens90, thr90 = float("nan"), float("nan")

    return macro_auc, per_class_auc, sens, spec, cm, sens90, thr90, probs, labels
