"""Multi-seed sweeps and CSV summaries."""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
from scipy import stats

from .config import Config, cfg
from .data import DataBundle
from .engine import run_one

RAW_FIELDS = [
    "backbone",
    "seed",
    "ot_mode",
    "ot_lambda",
    "use_focal",
    "use_phone_sampler",
    "best_epoch",
    "epochs_completed",
    "stop_reason",
    "best_checkpoint",
    "last_checkpoint",
    "prediction_csv",
    "hospital_prediction_csv",
    "phone_val_prediction_csv",
    "confusion_matrix_csv",
    "hospital_confusion_matrix_csv",
    "phone_val_confusion_matrix_csv",
    "best_phone_val_macro_auc",
    "val_ref_thr_at_spec90",
    "hosp_test_macro_auc",
    "hosp_test_accuracy",
    "hosp_test_balanced_accuracy",
    "hosp_test_macro_precision",
    "hosp_test_macro_recall",
    "hosp_test_macro_f1",
    "hosp_test_qwk",
    "phone_severity",
    "phone_root",
    "bootstrap_resampling_unit",
    "phone_test_macro_auc",
    "phone_test_accuracy",
    "phone_test_balanced_accuracy",
    "phone_test_macro_precision",
    "phone_test_macro_recall",
    "phone_test_macro_f1",
    "phone_test_qwk",
    "phone_test_macro_auc_boot_mean",
    "phone_test_macro_auc_ci_low",
    "phone_test_macro_auc_ci_high",
    "phone_test_ref_sens_at_valthr",
    "phone_test_ref_sens_at_valthr_boot_mean",
    "phone_test_ref_sens_at_valthr_ci_low",
    "phone_test_ref_sens_at_valthr_ci_high",
    "phone_test_ref_spec_at_valthr",
    "phone_test_ref_spec_at_valthr_boot_mean",
    "phone_test_ref_spec_at_valthr_ci_low",
    "phone_test_ref_spec_at_valthr_ci_high",
    "phone_test_per_class_auc",
    "phone_test_per_class_precision",
    "phone_test_per_class_recall",
    "phone_test_per_class_f1",
    "phone_test_per_class_support",
    "phone_test_confusion_matrix",
    "phone_test_sens",
    "phone_test_spec",
    "phone_test_sens_at_spec90_ovr",
    "phone_test_thr_at_spec90_ovr",
]


def mean_ci(values, alpha=0.95):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(values.mean()), (float("nan"), float("nan"))
    m = values.mean()
    s = values.std(ddof=1)
    ci = stats.t.interval(alpha, n - 1, loc=m, scale=s / np.sqrt(n))
    return float(m), (float(ci[0]), float(ci[1]))


def mean_std(values):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def summarize_group(bb_rows: list[dict[str, Any]], ot_mode: str, backbone: str, severity: str):
    aucs = [r.get("phone_test_macro_auc", np.nan) for r in bb_rows]
    accuracies = [r.get("phone_test_accuracy", np.nan) for r in bb_rows]
    balanced_accuracies = [r.get("phone_test_balanced_accuracy", np.nan) for r in bb_rows]
    macro_precisions = [r.get("phone_test_macro_precision", np.nan) for r in bb_rows]
    macro_recalls = [r.get("phone_test_macro_recall", np.nan) for r in bb_rows]
    macro_f1s = [r.get("phone_test_macro_f1", np.nan) for r in bb_rows]
    qwks = [r.get("phone_test_qwk", np.nan) for r in bb_rows]
    refsens = [r.get("phone_test_ref_sens_at_valthr", np.nan) for r in bb_rows]
    refspec = [r.get("phone_test_ref_spec_at_valthr", np.nan) for r in bb_rows]

    auc_ci_lows = [r.get("phone_test_macro_auc_ci_low", np.nan) for r in bb_rows]
    auc_ci_highs = [r.get("phone_test_macro_auc_ci_high", np.nan) for r in bb_rows]
    rs_ci_lows = [r.get("phone_test_ref_sens_at_valthr_ci_low", np.nan) for r in bb_rows]
    rs_ci_highs = [r.get("phone_test_ref_sens_at_valthr_ci_high", np.nan) for r in bb_rows]
    rp_ci_lows = [r.get("phone_test_ref_spec_at_valthr_ci_low", np.nan) for r in bb_rows]
    rp_ci_highs = [r.get("phone_test_ref_spec_at_valthr_ci_high", np.nan) for r in bb_rows]

    auc_m, auc_s = mean_std(aucs)
    accuracy_m, accuracy_s = mean_std(accuracies)
    balanced_accuracy_m, balanced_accuracy_s = mean_std(balanced_accuracies)
    macro_precision_m, macro_precision_s = mean_std(macro_precisions)
    macro_recall_m, macro_recall_s = mean_std(macro_recalls)
    macro_f1_m, macro_f1_s = mean_std(macro_f1s)
    qwk_m, qwk_s = mean_std(qwks)
    rs_m, rs_s = mean_std(refsens)
    rp_m, rp_s = mean_std(refspec)

    auc_lo_m, _ = mean_std(auc_ci_lows)
    auc_hi_m, _ = mean_std(auc_ci_highs)
    rs_lo_m, _ = mean_std(rs_ci_lows)
    rs_hi_m, _ = mean_std(rs_ci_highs)
    rp_lo_m, _ = mean_std(rp_ci_lows)
    rp_hi_m, _ = mean_std(rp_ci_highs)

    return {
        "ot_mode": ot_mode,
        "backbone": backbone,
        "phone_severity": severity,
        "n_seeds": len(bb_rows),
        "phone_test_macro_auc_mean": auc_m,
        "phone_test_macro_auc_std": auc_s,
        "phone_test_macro_auc_ci_low_mean": auc_lo_m,
        "phone_test_macro_auc_ci_high_mean": auc_hi_m,
        "phone_test_accuracy_mean": accuracy_m,
        "phone_test_accuracy_std": accuracy_s,
        "phone_test_balanced_accuracy_mean": balanced_accuracy_m,
        "phone_test_balanced_accuracy_std": balanced_accuracy_s,
        "phone_test_macro_precision_mean": macro_precision_m,
        "phone_test_macro_precision_std": macro_precision_s,
        "phone_test_macro_recall_mean": macro_recall_m,
        "phone_test_macro_recall_std": macro_recall_s,
        "phone_test_macro_f1_mean": macro_f1_m,
        "phone_test_macro_f1_std": macro_f1_s,
        "phone_test_qwk_mean": qwk_m,
        "phone_test_qwk_std": qwk_s,
        "phone_test_ref_sens_at_valthr_mean": rs_m,
        "phone_test_ref_sens_at_valthr_std": rs_s,
        "phone_test_ref_sens_at_valthr_ci_low_mean": rs_lo_m,
        "phone_test_ref_sens_at_valthr_ci_high_mean": rs_hi_m,
        "phone_test_ref_spec_at_valthr_mean": rp_m,
        "phone_test_ref_spec_at_valthr_std": rp_s,
        "phone_test_ref_spec_at_valthr_ci_low_mean": rp_lo_m,
        "phone_test_ref_spec_at_valthr_ci_high_mean": rp_hi_m,
    }


def write_csv(path: str, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, None) for key in fields})


def build_summary_rows(
    rows: list[dict[str, Any]], ot_mode: str, config: Config | None = None
) -> list[dict[str, Any]]:
    config = config or cfg
    summary_rows = []
    for backbone in config.backbones:
        for severity in config.phone_eval_severities:
            group = [
                row
                for row in rows
                if row.get("backbone") == backbone
                and row.get("ot_mode") == ot_mode
                and row.get("phone_severity") == severity
            ]
            if not group:
                continue
            summary_rows.append(summarize_group(group, ot_mode, backbone, severity))
    return summary_rows


def run_sweep(
    data: DataBundle,
    save_name=None,
    save_summary=True,
    config: Config | None = None,
):
    config = config or cfg
    rows = []
    for backbone in config.backbones:
        vals_auc = []
        for seed in config.seeds:
            rows_sd = run_one(data, backbone, seed, config=config)
            rows.extend(rows_sd)
            for row in rows_sd:
                if row.get("phone_severity") == "clean":
                    vals_auc.append(row.get("phone_test_macro_auc", np.nan))

        m_auc, ci_auc = mean_ci(vals_auc, 0.95)
        print(f"== {backbone} (clean) | AUC mean={m_auc:.4f} 95%CI={ci_auc} ==")

    if save_name is None:
        save_name = f"RAW_results_ot_{config.ot_mode}.csv"
    csv_path = os.path.join(config.out_dir, save_name)
    write_csv(csv_path, rows, RAW_FIELDS)
    print("Saved RAW:", csv_path)

    if save_summary:
        summary_rows = build_summary_rows(rows, config.ot_mode, config)
        if summary_rows:
            summary_path = os.path.join(
                config.out_dir, f"CLINICAL_SUMMARY_ot_{config.ot_mode}.csv"
            )
            write_csv(summary_path, summary_rows)
            print("Saved SUMMARY:", summary_path)

    return rows


def run_all_ot_modes(data: DataBundle, config: Config | None = None) -> None:
    config = config or cfg
    ot_modes = list(config.ot_modes)
    all_summary_rows = []
    base_ot_lambda = float(config.ot_lambda)

    for ot_mode in ot_modes:
        config.ot_mode = ot_mode
        lam = 0.0 if ot_mode == "none" else base_ot_lambda
        config.ot_lambda = lam

        print("\n" + "=" * 100)
        print(f"RUNNING OT MODE: {ot_mode} | lambda={lam}")
        print("=" * 100)

        rows = run_sweep(
            data,
            save_name=f"RAW_results_ot_{ot_mode}.csv",
            save_summary=False,
            config=config,
        )
        summary_rows = build_summary_rows(rows, ot_mode, config)
        all_summary_rows.extend(summary_rows)

        if summary_rows:
            summary_path = os.path.join(
                config.out_dir, f"CLINICAL_SUMMARY_ot_{ot_mode}.csv"
            )
            write_csv(summary_path, summary_rows)
            print("Saved SUMMARY:", summary_path)
        else:
            print(f"[WARN] No summary rows generated for mode={ot_mode}")

    if all_summary_rows:
        all_path = os.path.join(config.out_dir, "CLINICAL_SUMMARY_ALL_OT_MODES.csv")
        write_csv(all_path, all_summary_rows)
        print("Saved COMBINED SUMMARY:", all_path)
    else:
        print("[WARN] No combined summary rows were generated.")

    config.ot_lambda = base_ot_lambda
