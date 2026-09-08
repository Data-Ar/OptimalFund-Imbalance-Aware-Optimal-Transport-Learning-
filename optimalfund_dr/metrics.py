"""Classification metrics, clinical thresholds, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .config import Config, cfg
from .utils import extract_patient_id


@dataclass
class EvalResult:
    macro_auc: float
    per_class_auc: np.ndarray
    sens: np.ndarray
    spec: np.ndarray
    cm: np.ndarray
    sens90: Any
    thr90: Any
    probs: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    extended: dict[str, Any]


def compute_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, num_classes: int):
    macro_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    per_class_auc = roc_auc_score(y_true, probs, multi_class="ovr", average=None)

    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))

    sens, spec = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn + 1e-8))
        spec.append(tn / (tn + fp + 1e-8))

    return (
        float(macro_auc),
        per_class_auc.astype(float),
        np.array(sens, float),
        np.array(spec, float),
        cm,
    )


def compute_extended_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    labels = list(range(num_classes))
    predicted = np.argmax(probs, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predicted, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "qwk": float(
            cohen_kappa_score(y_true, predicted, labels=labels, weights="quadratic")
        ),
        "per_class_precision": precision.astype(float),
        "per_class_recall": recall.astype(float),
        "per_class_f1": f1.astype(float),
        "per_class_support": support.astype(int),
        "confusion_matrix": matrix.astype(int),
    }


def sens_at_spec_ovr(
    y_true: np.ndarray, probs: np.ndarray, num_classes: int, target_spec: float
):
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


def choose_ref_threshold_at_spec(
    y_true_mc: np.ndarray, probs_mc: np.ndarray, ref_classes, target_spec: float
):
    y_ref = np.isin(y_true_mc, list(ref_classes)).astype(int)
    p_ref = probs_mc[:, list(ref_classes)].sum(axis=1)
    fpr, tpr, thr = roc_curve(y_ref, p_ref)
    spec = 1.0 - fpr
    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = valid[np.argmax(tpr[valid])]
    return float(tpr[idx]), float(spec[idx]), float(thr[idx])


def refsens_spec_at_threshold(
    y_true_mc: np.ndarray, probs_mc: np.ndarray, thr: float, ref_classes
):
    y_ref = np.isin(y_true_mc, list(ref_classes)).astype(int)
    p_ref = probs_mc[:, list(ref_classes)].sum(axis=1)
    pred = (p_ref >= thr).astype(int)

    tp = np.sum((pred == 1) & (y_ref == 1))
    fn = np.sum((pred == 0) & (y_ref == 1))
    fp = np.sum((pred == 1) & (y_ref == 0))
    tn = np.sum((pred == 0) & (y_ref == 0))

    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return float(sens), float(spec)


def bootstrap_ci_patient_clustered(
    y_true_mc,
    probs_mc,
    patient_ids,
    num_classes,
    ref_thr,
    ref_classes,
    n_boot=1000,
    alpha=0.95,
    seed=42,
):
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids, dtype=str)
    if len(patient_ids) != len(y_true_mc):
        raise ValueError("patient_ids and y_true_mc must have the same length.")
    unique_patients, inverse = np.unique(patient_ids, return_inverse=True)
    rows_by_patient = [
        np.flatnonzero(inverse == patient_index)
        for patient_index in range(len(unique_patients))
    ]

    auc_vals = []
    sens_vals = []
    spec_vals = []

    lo_pct = (1.0 - alpha) / 2.0 * 100.0
    hi_pct = (1.0 + alpha) / 2.0 * 100.0

    for _ in range(n_boot):
        selected_patients = rng.integers(0, len(unique_patients), size=len(unique_patients))
        idx = np.concatenate(
            [rows_by_patient[patient_index] for patient_index in selected_patients]
        )
        y_b = y_true_mc[idx]
        p_b = probs_mc[idx]

        try:
            auc_b, _, _, _, _ = compute_metrics_from_probs(y_b, p_b, num_classes)
            if np.isfinite(auc_b):
                auc_vals.append(float(auc_b))
        except Exception:
            pass

        try:
            sens_b, spec_b = refsens_spec_at_threshold(y_b, p_b, ref_thr, ref_classes)
            if np.isfinite(sens_b):
                sens_vals.append(float(sens_b))
            if np.isfinite(spec_b):
                spec_vals.append(float(spec_b))
        except Exception:
            pass

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
        f"[PATIENT BOOTSTRAP] patients={len(unique_patients)}, "
        f"valid_auc={len(auc_vals)}/{n_boot}, "
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


@torch.no_grad()
def collect_probs_labels(
    model: nn.Module, loader: DataLoader, config: Config | None = None
):
    config = config or cfg
    model.eval()
    probs_list, labels_list, paths_list = [], [], []
    for x, y, paths in loader:
        x = x.to(config.device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(y.numpy())
        paths_list.extend(str(path) for path in paths)
    return (
        np.vstack(probs_list),
        np.concatenate(labels_list),
        np.asarray(paths_list, dtype=str),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, config: Config | None = None
) -> EvalResult:
    config = config or cfg
    probs, labels, paths = collect_probs_labels(model, loader, config=config)

    try:
        macro_auc, per_class_auc, sens, spec, cm = compute_metrics_from_probs(
            labels, probs, config.num_classes
        )
    except Exception as e:
        print(f"[WARN] compute_metrics_from_probs failed: {e}")
        macro_auc = float("nan")
        per_class_auc = np.full(config.num_classes, np.nan)
        sens = np.full(config.num_classes, np.nan)
        spec = np.full(config.num_classes, np.nan)
        cm = np.full((config.num_classes, config.num_classes), np.nan)

    try:
        sens90, thr90 = sens_at_spec_ovr(
            labels, probs, config.num_classes, target_spec=config.target_spec
        )
    except Exception as e:
        print(f"[WARN] sens_at_spec_ovr failed: {e}")
        sens90, thr90 = float("nan"), float("nan")

    try:
        extended = compute_extended_classification_metrics(
            labels, probs, config.num_classes
        )
    except Exception as e:
        print(f"[WARN] extended classification metrics failed: {e}")
        extended = {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "qwk": float("nan"),
            "per_class_precision": np.full(config.num_classes, np.nan),
            "per_class_recall": np.full(config.num_classes, np.nan),
            "per_class_f1": np.full(config.num_classes, np.nan),
            "per_class_support": np.zeros(config.num_classes, dtype=int),
            "confusion_matrix": np.zeros(
                (config.num_classes, config.num_classes), dtype=int
            ),
        }

    return EvalResult(
        macro_auc=macro_auc,
        per_class_auc=per_class_auc,
        sens=sens,
        spec=spec,
        cm=cm,
        sens90=sens90,
        thr90=thr90,
        probs=probs,
        labels=labels,
        paths=paths,
        extended=extended,
    )


def patient_ids_from_paths(paths: np.ndarray) -> np.ndarray:
    return np.asarray([extract_patient_id(path) for path in paths], dtype=str)
