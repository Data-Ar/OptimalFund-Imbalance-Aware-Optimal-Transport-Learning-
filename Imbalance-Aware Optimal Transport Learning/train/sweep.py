import os
import csv
import numpy as np
from scipy import stats

from train.train_loop import run_one


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


def run_sweep(cfg, data_objects, save_name=None, save_summary=True):
    rows = []
    for bb in cfg.backbones:
        vals_auc = []
        vals_refsens = []
        vals_refspec = []

        for sd in cfg.seeds:
            rows_sd = run_one(bb, sd, cfg, data_objects)  # list of rows (one per severity)
            rows.extend(rows_sd)

            # collect main clinical metrics (PHONE TEST) on CLEAN only
            for row in rows_sd:
                if row.get("phone_severity") == "clean":
                    vals_auc.append(row.get("phone_test_macro_auc", np.nan))
                    vals_refsens.append(row.get("phone_test_ref_sens_at_valthr", np.nan))
                    vals_refspec.append(row.get("phone_test_ref_spec_at_valthr", np.nan))

        # print backbone-level mean/CI for CLEAN (informational; still t-based over seeds)
        m_auc, ci_auc = mean_ci(vals_auc, 0.95)
        print(f"== {bb} (clean) | AUC mean={m_auc:.4f} 95%CI={ci_auc} ==")

    # ---------- RAW CSV ----------
    if save_name is None:
        save_name = f"RAW_results_ot_{cfg.ot_mode}.csv"
    csv_path = os.path.join(cfg.out_dir, save_name)

    fields = [
        "backbone", "seed", "ot_mode", "ot_lambda", "use_focal", "use_phone_sampler",
        "best_phone_val_macro_auc", "val_ref_thr_at_spec90",
        "hosp_test_macro_auc",
        "phone_severity", "phone_root",

        "phone_test_macro_auc",
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

        "phone_test_per_class_auc", "phone_test_sens", "phone_test_spec",
        "phone_test_sens_at_spec90_ovr", "phone_test_thr_at_spec90_ovr"
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fields})
    print("Saved RAW:", csv_path)

    # ---------- SUMMARY CSV (mean±std over seeds, grouped by backbone + severity) ----------
    if save_summary:
        summary_rows = []
        for bb in cfg.backbones:
            for sev in cfg.phone_eval_severities:
                bb_rows = [
                    r for r in rows
                    if r.get("backbone") == bb and r.get("ot_mode") == cfg.ot_mode and r.get("phone_severity") == sev
                ]
                if len(bb_rows) == 0:
                    continue

                aucs = [r.get("phone_test_macro_auc", np.nan) for r in bb_rows]
                refsens = [r.get("phone_test_ref_sens_at_valthr", np.nan) for r in bb_rows]
                refspec = [r.get("phone_test_ref_spec_at_valthr", np.nan) for r in bb_rows]

                auc_ci_lows = [r.get("phone_test_macro_auc_ci_low", np.nan) for r in bb_rows]
                auc_ci_highs = [r.get("phone_test_macro_auc_ci_high", np.nan) for r in bb_rows]

                rs_ci_lows = [r.get("phone_test_ref_sens_at_valthr_ci_low", np.nan) for r in bb_rows]
                rs_ci_highs = [r.get("phone_test_ref_sens_at_valthr_ci_high", np.nan) for r in bb_rows]

                rp_ci_lows = [r.get("phone_test_ref_spec_at_valthr_ci_low", np.nan) for r in bb_rows]
                rp_ci_highs = [r.get("phone_test_ref_spec_at_valthr_ci_high", np.nan) for r in bb_rows]

                auc_m, auc_s = mean_std(aucs)
                rs_m, rs_s = mean_std(refsens)
                rp_m, rp_s = mean_std(refspec)

                auc_lo_m, _ = mean_std(auc_ci_lows)
                auc_hi_m, _ = mean_std(auc_ci_highs)

                rs_lo_m, _ = mean_std(rs_ci_lows)
                rs_hi_m, _ = mean_std(rs_ci_highs)

                rp_lo_m, _ = mean_std(rp_ci_lows)
                rp_hi_m, _ = mean_std(rp_ci_highs)

                summary_rows.append({
                    "ot_mode": cfg.ot_mode,
                    "backbone": bb,
                    "phone_severity": sev,
                    "n_seeds": len(bb_rows),

                    "phone_test_macro_auc_mean": auc_m,
                    "phone_test_macro_auc_std": auc_s,
                    "phone_test_macro_auc_ci_low_mean": auc_lo_m,
                    "phone_test_macro_auc_ci_high_mean": auc_hi_m,

                    "phone_test_ref_sens_at_valthr_mean": rs_m,
                    "phone_test_ref_sens_at_valthr_std": rs_s,
                    "phone_test_ref_sens_at_valthr_ci_low_mean": rs_lo_m,
                    "phone_test_ref_sens_at_valthr_ci_high_mean": rs_hi_m,

                    "phone_test_ref_spec_at_valthr_mean": rp_m,
                    "phone_test_ref_spec_at_valthr_std": rp_s,
                    "phone_test_ref_spec_at_valthr_ci_low_mean": rp_lo_m,
                    "phone_test_ref_spec_at_valthr_ci_high_mean": rp_hi_m,
                })

        summary_name = f"CLINICAL_SUMMARY_ot_{cfg.ot_mode}.csv"
        summary_path = os.path.join(cfg.out_dir, summary_name)
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print("Saved SUMMARY:", summary_path)

    return rows


def run_all_ot_modes(cfg, data_objects):
    ot_modes = ["class_sinkhorn"]
    #ot_modes = ["class_sinkhorn", "none", "prototype", "sinkhorn"]
    all_summary_rows = []

    for om in ot_modes:
        cfg.ot_mode = om

        # set a single lambda (or 0 for none)
        lam = 0.0 if om == "none" else cfg.ot_lambda
        cfg.ot_lambda = lam

        print("\n" + "=" * 100)
        print(f"RUNNING OT MODE: {om} | lambda={lam}")
        print("=" * 100)

        rows = run_sweep(cfg, data_objects, 
            save_name=f"RAW_results_ot_{om}.csv",
            save_summary=False
        )

        summary_rows = []
        for bb in cfg.backbones:
            for sev in cfg.phone_eval_severities:
                bb_rows = [
                    r for r in rows
                    if r.get("backbone") == bb
                    and r.get("ot_mode") == om
                    and r.get("phone_severity") == sev
                ]

                if len(bb_rows) == 0:
                    continue

                aucs = [r.get("phone_test_macro_auc", np.nan) for r in bb_rows]
                refsens = [r.get("phone_test_ref_sens_at_valthr", np.nan) for r in bb_rows]
                refspec = [r.get("phone_test_ref_spec_at_valthr", np.nan) for r in bb_rows]

                auc_ci_lows = [r.get("phone_test_macro_auc_ci_low", np.nan) for r in bb_rows]
                auc_ci_highs = [r.get("phone_test_macro_auc_ci_high", np.nan) for r in bb_rows]

                rs_ci_lows = [r.get("phone_test_ref_sens_at_valthr_ci_low", np.nan) for r in bb_rows]
                rs_ci_highs = [r.get("phone_test_ref_sens_at_valthr_ci_high", np.nan) for r in bb_rows]

                rp_ci_lows = [r.get("phone_test_ref_spec_at_valthr_ci_low", np.nan) for r in bb_rows]
                rp_ci_highs = [r.get("phone_test_ref_spec_at_valthr_ci_high", np.nan) for r in bb_rows]

                auc_m, auc_s = mean_std(aucs)
                rs_m, rs_s = mean_std(refsens)
                rp_m, rp_s = mean_std(refspec)

                auc_lo_m, _ = mean_std(auc_ci_lows)
                auc_hi_m, _ = mean_std(auc_ci_highs)

                rs_lo_m, _ = mean_std(rs_ci_lows)
                rs_hi_m, _ = mean_std(rs_ci_highs)

                rp_lo_m, _ = mean_std(rp_ci_lows)
                rp_hi_m, _ = mean_std(rp_ci_highs)

                row = {
                    "ot_mode": om,
                    "backbone": bb,
                    "phone_severity": sev,
                    "n_seeds": len(bb_rows),

                    "phone_test_macro_auc_mean": auc_m,
                    "phone_test_macro_auc_std": auc_s,
                    "phone_test_macro_auc_ci_low_mean": auc_lo_m,
                    "phone_test_macro_auc_ci_high_mean": auc_hi_m,

                    "phone_test_ref_sens_at_valthr_mean": rs_m,
                    "phone_test_ref_sens_at_valthr_std": rs_s,
                    "phone_test_ref_sens_at_valthr_ci_low_mean": rs_lo_m,
                    "phone_test_ref_sens_at_valthr_ci_high_mean": rs_hi_m,

                    "phone_test_ref_spec_at_valthr_mean": rp_m,
                    "phone_test_ref_spec_at_valthr_std": rp_s,
                    "phone_test_ref_spec_at_valthr_ci_low_mean": rp_lo_m,
                    "phone_test_ref_spec_at_valthr_ci_high_mean": rp_hi_m,
                }

                summary_rows.append(row)
                all_summary_rows.append(row)

        if len(summary_rows) > 0:
            summary_path = os.path.join(
                cfg.out_dir,
                f"CLINICAL_SUMMARY_ot_{om}.csv"
            )
            with open(summary_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                for r in summary_rows:
                    w.writerow(r)
            print("Saved SUMMARY:", summary_path)
        else:
            print(f"[WARN] No summary rows generated for mode={om}")

    if len(all_summary_rows) > 0:
        all_path = os.path.join(cfg.out_dir, "CLINICAL_SUMMARY_ALL_OT_MODES.csv")
        with open(all_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
            w.writeheader()
            for r in all_summary_rows:
                w.writerow(r)
        print("Saved COMBINED SUMMARY:", all_path)
    else:
        print("[WARN] No combined summary rows were generated.")
