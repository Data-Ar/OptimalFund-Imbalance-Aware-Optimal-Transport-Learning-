"""Training loop and one resumable experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    atomic_json_save,
    atomic_torch_save,
    assert_matching_signature,
    cached_evaluation_rows,
    capture_rng_state,
    evaluation_signature,
    load_torch_checkpoint,
    model_state_cpu,
    optimizer_to_device,
    restore_rng_state,
    run_artifact_paths,
    save_confusion_matrix_csv,
    save_prediction_csv,
    training_signature,
)
from .config import Config, cfg, set_seed
from .data import DataBundle, build_run_train_loaders
from .losses import compute_ot_loss
from .metrics import (
    bootstrap_ci_patient_clustered,
    choose_ref_threshold_at_spec,
    evaluate,
    patient_ids_from_paths,
    refsens_spec_at_threshold,
)
from .model import DRModel, get_criterion


def train_one_epoch(model, loader_h, loader_p, opt, criterion, scaler=None, config: Config | None = None):
    config = config or cfg
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_ot = 0.0

    num_steps = len(loader_p) * config.steps_multiplier
    phone_iter = iter(loader_p)
    hosp_iter = iter(loader_h)

    for _ in range(num_steps):
        try:
            xp, yp, _ = next(phone_iter)
        except StopIteration:
            phone_iter = iter(loader_p)
            xp, yp, _ = next(phone_iter)

        try:
            xh, yh, _ = next(hosp_iter)
        except StopIteration:
            hosp_iter = iter(loader_h)
            xh, yh, _ = next(hosp_iter)

        xp = xp.to(config.device, non_blocking=True)
        yp = yp.to(config.device, non_blocking=True)
        xh = xh.to(config.device, non_blocking=True)
        yh = yh.to(config.device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        use_amp = (
            scaler is not None
            and scaler.is_enabled()
            and config.use_amp
            and str(config.device).startswith("cuda")
        )

        if use_amp:
            with autocast("cuda", enabled=True):
                logits_h, feat_h = model(xh, return_feat=True)
                logits_p, feat_p = model(xp, return_feat=True)

                loss_ce = criterion(logits_h, yh) + criterion(logits_p, yp)
                loss_ot = compute_ot_loss(feat_h, yh, feat_p, yp, config=config)
                loss = loss_ce + config.ot_lambda * loss_ot

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss encountered: {loss.item()}")
                continue

            scaler.scale(loss).backward()

            if config.grad_clip and config.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(opt)
            scaler.update()
        else:
            logits_h, feat_h = model(xh, return_feat=True)
            logits_p, feat_p = model(xp, return_feat=True)

            loss_ce = criterion(logits_h, yh) + criterion(logits_p, yp)
            loss_ot = compute_ot_loss(feat_h, yh, feat_p, yp, config=config)
            loss = loss_ce + config.ot_lambda * loss_ot

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss encountered: {loss.item()}")
                continue

            loss.backward()

            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            opt.step()

        total_loss += float(loss.item())
        total_ce += float(loss_ce.item())
        total_ot += float(loss_ot.item()) if hasattr(loss_ot, "item") else float(loss_ot)

    denom = max(1, num_steps)
    return {
        "loss": total_loss / denom,
        "loss_ce": total_ce / denom,
        "loss_ot": total_ot / denom,
    }


def _empty_bootstrap() -> dict[str, Any]:
    return {
        "auc_boot_mean": float("nan"),
        "auc_ci_low": float("nan"),
        "auc_ci_high": float("nan"),
        "ref_sens_boot_mean": float("nan"),
        "ref_sens_ci_low": float("nan"),
        "ref_sens_ci_high": float("nan"),
        "ref_spec_boot_mean": float("nan"),
        "ref_spec_ci_low": float("nan"),
        "ref_spec_ci_high": float("nan"),
        "auc_boot_valid": 0,
        "ref_sens_boot_valid": 0,
        "ref_spec_boot_valid": 0,
        "n_boot": 0,
    }


def run_one(
    data: DataBundle,
    backbone: str,
    seed: int,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Train or resume one backbone/seed run and return per-severity evaluation rows."""
    config = config or cfg

    set_seed(seed, config)
    signature = training_signature(backbone, seed, config)
    artifacts = run_artifact_paths(backbone, seed, config)
    artifacts["run_dir"].mkdir(parents=True, exist_ok=True)

    existing = [path for key, path in artifacts.items() if key != "run_dir" and path.exists()]
    if existing and not config.resume and not config.overwrite_existing:
        raise FileExistsError(
            f"Existing artifacts found for {config.ot_mode}/{backbone}/seed_{seed}. "
            "Use --resume to continue or --overwrite to retrain this run."
        )

    model = DRModel(backbone, config.num_classes, config=config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = get_criterion(config)
    amp_enabled = config.use_amp and str(config.device).startswith("cuda")
    try:
        scaler = GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    last_epoch = -1
    best_epoch = -1
    best_val_auc = -float("inf")
    best_ref_thr = float("nan")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    training_complete = False
    stop_reason = "not_started"
    generator_states = None

    if config.resume and artifacts["last"].exists():
        checkpoint = load_torch_checkpoint(artifacts["last"], map_location="cpu")
        assert_matching_signature(checkpoint, signature)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to_device(optimizer, config.device)
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        last_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint.get("best_epoch", -1))
        best_val_auc = float(checkpoint.get("best_val_auc", -float("inf")))
        best_ref_thr = float(checkpoint.get("best_ref_thr", float("nan")))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        history = list(checkpoint.get("history", []))
        training_complete = bool(checkpoint.get("training_complete", False))
        stop_reason = str(checkpoint.get("stop_reason", "resumed"))
        generator_states = checkpoint.get("loader_generator_states")
        print(
            f"[RESUME] {config.ot_mode}/{backbone}/seed_{seed} from epoch "
            f"{start_epoch}; complete={training_complete}"
        )
    elif config.resume and artifacts["best"].exists():
        raise RuntimeError(
            f"Found best.pt but no last.pt for {config.ot_mode}/{backbone}/seed_{seed}. "
            "A full optimizer resume is impossible. Use --overwrite to retrain."
        )
    elif config.resume:
        print(
            f"[RESUME] No checkpoint found for {config.ot_mode}/{backbone}/seed_{seed}; starting new."
        )

    ld_h_train_run, ld_p_train_run, loader_generators = build_run_train_loaders(
        data, seed, generator_states, config=config
    )

    if config.resume and artifacts["last"].exists():
        checkpoint = load_torch_checkpoint(artifacts["last"], map_location="cpu")
        restore_rng_state(checkpoint.get("rng_state"))

    if training_complete:
        cached_rows = cached_evaluation_rows(artifacts["rows"], signature, config=config)
        if cached_rows is not None:
            print(
                f"[RESUME] Reusing completed evaluation for {config.ot_mode}/{backbone}/seed_{seed}."
            )
            return cached_rows

    if not training_complete:
        stop_reason = "max_epochs"
        for epoch in range(start_epoch, config.epochs):
            tr_stats = train_one_epoch(
                model,
                ld_h_train_run,
                ld_p_train_run,
                optimizer,
                criterion,
                scaler,
                config=config,
            )

            val = evaluate(model, data.ld_p_val, config=config)
            macro_sens = float(np.mean(val.sens))
            macro_spec = float(np.mean(val.spec))
            ref_sens90, ref_spec90, ref_thr90 = choose_ref_threshold_at_spec(
                val.labels, val.probs, config.ref_classes, config.target_spec
            )

            improved = np.isfinite(val.macro_auc) and (
                val.macro_auc > best_val_auc + config.early_stopping_min_delta
            )
            if improved:
                best_val_auc = float(val.macro_auc)
                best_ref_thr = float(ref_thr90)
                best_epoch = epoch
                epochs_without_improvement = 0
                best_payload = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "training_signature": signature,
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_val_auc": float(best_val_auc),
                    "best_ref_thr": float(best_ref_thr),
                    "model_state": model_state_cpu(model),
                }
                atomic_torch_save(best_payload, artifacts["best"])
                print("Saved best checkpoint:", artifacts["best"])
            else:
                epochs_without_improvement += 1

            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(tr_stats["loss"]),
                    "train_ce": float(tr_stats["loss_ce"]),
                    "train_ot": float(tr_stats["loss_ot"]),
                    "val_macro_auc": float(val.macro_auc),
                    "val_macro_sens": float(macro_sens),
                    "val_macro_spec": float(macro_spec),
                    "val_ref_sens_at_spec90": float(ref_sens90),
                    "val_ref_spec_at_spec90": float(ref_spec90),
                    "val_ref_threshold": float(ref_thr90),
                    "improved": bool(improved),
                }
            )
            last_epoch = epoch

            last_payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "training_signature": signature,
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_val_auc": float(best_val_auc),
                "best_ref_thr": float(best_ref_thr),
                "epochs_without_improvement": int(epochs_without_improvement),
                "model_state": model_state_cpu(model),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "loader_generator_states": {
                    name: generator.get_state()
                    for name, generator in loader_generators.items()
                },
                "rng_state": capture_rng_state(),
                "history": history,
                "training_complete": False,
                "stop_reason": "running",
            }
            atomic_torch_save(last_payload, artifacts["last"])

            print(
                f"Epoch {epoch + 1}/{config.epochs} "
                f"Loss={tr_stats['loss']:.4f} "
                f"CE={tr_stats['loss_ce']:.4f} "
                f"OT={tr_stats['loss_ot']:.4f} "
                f"LR={optimizer.param_groups[0]['lr']:.2e} "
                f"NoImprove={epochs_without_improvement}/{config.early_stopping_patience}"
            )
            print(
                f"PHONE VAL  | AUC={val.macro_auc:.4f}  "
                f"macro_sens={macro_sens:.4f}  macro_spec={macro_spec:.4f}  "
                f"RefSens@Spec0.90={ref_sens90:.4f}"
            )

            if (
                config.early_stopping_patience > 0
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                stop_reason = "early_stopping"
                print(
                    f"[EARLY STOP] No validation macro-AUC improvement for "
                    f"{config.early_stopping_patience} consecutive epochs."
                )
                break

        if best_epoch < 0 or not artifacts["best"].exists():
            raise RuntimeError("Training completed without a valid validation checkpoint.")

        final_payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "training_signature": signature,
            "epoch": int(last_epoch),
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "best_ref_thr": float(best_ref_thr),
            "epochs_without_improvement": int(epochs_without_improvement),
            "model_state": model_state_cpu(model),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loader_generator_states": {
                name: generator.get_state()
                for name, generator in loader_generators.items()
            },
            "rng_state": capture_rng_state(),
            "history": history,
            "training_complete": True,
            "stop_reason": stop_reason,
        }
        atomic_torch_save(final_payload, artifacts["last"])
        print("Saved resumable checkpoint:", artifacts["last"])

    best_checkpoint = load_torch_checkpoint(artifacts["best"], map_location="cpu")
    assert_matching_signature(best_checkpoint, signature)
    model.load_state_dict(best_checkpoint["model_state"])
    best_epoch = int(best_checkpoint["best_epoch"])
    best_val_auc = float(best_checkpoint["best_val_auc"])
    best_ref_thr = float(best_checkpoint["best_ref_thr"])
    model.to(config.device)

    hosp_test = evaluate(model, data.ld_h_test, config=config)
    phone_val_best = evaluate(model, data.ld_p_val, config=config)
    hospital_prediction_path = None
    phone_val_prediction_path = None
    if config.save_predictions:
        hospital_prediction_path = save_prediction_csv(
            backbone,
            seed,
            "hospital_test",
            hosp_test.paths,
            hosp_test.labels,
            hosp_test.probs,
            output_filename="hospital_test.csv",
            config=config,
        )
        phone_val_prediction_path = save_prediction_csv(
            backbone,
            seed,
            "phone_val_clean",
            phone_val_best.paths,
            phone_val_best.labels,
            phone_val_best.probs,
            output_filename="phone_val_clean.csv",
            config=config,
        )
    hospital_confusion_path = save_confusion_matrix_csv(
        backbone,
        seed,
        hosp_test.extended["confusion_matrix"],
        "confusion_matrix_hospital_test.csv",
        config=config,
    )
    phone_val_confusion_path = save_confusion_matrix_csv(
        backbone,
        seed,
        phone_val_best.extended["confusion_matrix"],
        "confusion_matrix_phone_val_clean.csv",
        config=config,
    )

    rows = []
    for sev in config.phone_eval_severities:
        sev_root = data.phone_test_loaders[sev]["root"]
        ld_p_test_sev = data.phone_test_loaders[sev]["loader"]
        phone_test = evaluate(model, ld_p_test_sev, config=config)
        phone_metrics = phone_test.extended

        if config.save_predictions:
            saved_prediction_path = save_prediction_csv(
                backbone,
                seed,
                sev,
                phone_test.paths,
                phone_test.labels,
                phone_test.probs,
                config=config,
            )
        else:
            saved_prediction_path = None
        confusion_matrix_path = save_confusion_matrix_csv(
            backbone,
            seed,
            phone_metrics["confusion_matrix"],
            f"confusion_matrix_phone_test_{sev}.csv",
            config=config,
        )

        test_ref_sens, test_ref_spec = refsens_spec_at_threshold(
            phone_test.labels, phone_test.probs, best_ref_thr, config.ref_classes
        )

        do_boot = (sev == "clean") or config.bootstrap_all_severities
        if do_boot:
            boot = bootstrap_ci_patient_clustered(
                y_true_mc=phone_test.labels,
                probs_mc=phone_test.probs,
                patient_ids=patient_ids_from_paths(phone_test.paths),
                ref_thr=best_ref_thr,
                ref_classes=config.ref_classes,
                num_classes=config.num_classes,
                n_boot=config.n_boot,
                alpha=0.95,
                seed=seed,
            )
        else:
            boot = _empty_bootstrap()

        rows.append(
            {
                "backbone": backbone,
                "seed": seed,
                "ot_mode": config.ot_mode,
                "ot_lambda": config.ot_lambda,
                "use_focal": config.use_focal_loss,
                "use_phone_sampler": config.use_phone_weighted_sampler,
                "best_epoch": best_epoch + 1,
                "epochs_completed": last_epoch + 1,
                "stop_reason": stop_reason,
                "best_checkpoint": str(artifacts["best"]),
                "last_checkpoint": str(artifacts["last"]),
                "prediction_csv": str(saved_prediction_path) if saved_prediction_path else "",
                "hospital_prediction_csv": (
                    str(hospital_prediction_path) if hospital_prediction_path else ""
                ),
                "phone_val_prediction_csv": (
                    str(phone_val_prediction_path) if phone_val_prediction_path else ""
                ),
                "confusion_matrix_csv": str(confusion_matrix_path),
                "hospital_confusion_matrix_csv": str(hospital_confusion_path),
                "phone_val_confusion_matrix_csv": str(phone_val_confusion_path),
                "best_phone_val_macro_auc": float(best_val_auc),
                "val_ref_thr_at_spec90": float(best_ref_thr),
                "hosp_test_macro_auc": float(hosp_test.macro_auc),
                "hosp_test_accuracy": float(hosp_test.extended["accuracy"]),
                "hosp_test_balanced_accuracy": float(
                    hosp_test.extended["balanced_accuracy"]
                ),
                "hosp_test_macro_precision": float(hosp_test.extended["macro_precision"]),
                "hosp_test_macro_recall": float(hosp_test.extended["macro_recall"]),
                "hosp_test_macro_f1": float(hosp_test.extended["macro_f1"]),
                "hosp_test_qwk": float(hosp_test.extended["qwk"]),
                "phone_severity": sev,
                "phone_root": sev_root,
                "bootstrap_resampling_unit": "patient_filename",
                "phone_test_macro_auc": float(phone_test.macro_auc),
                "phone_test_accuracy": float(phone_metrics["accuracy"]),
                "phone_test_balanced_accuracy": float(phone_metrics["balanced_accuracy"]),
                "phone_test_macro_precision": float(phone_metrics["macro_precision"]),
                "phone_test_macro_recall": float(phone_metrics["macro_recall"]),
                "phone_test_macro_f1": float(phone_metrics["macro_f1"]),
                "phone_test_qwk": float(phone_metrics["qwk"]),
                "phone_test_macro_auc_boot_mean": boot["auc_boot_mean"],
                "phone_test_macro_auc_ci_low": boot["auc_ci_low"],
                "phone_test_macro_auc_ci_high": boot["auc_ci_high"],
                "phone_test_ref_sens_at_valthr": float(test_ref_sens),
                "phone_test_ref_sens_at_valthr_boot_mean": boot["ref_sens_boot_mean"],
                "phone_test_ref_sens_at_valthr_ci_low": boot["ref_sens_ci_low"],
                "phone_test_ref_sens_at_valthr_ci_high": boot["ref_sens_ci_high"],
                "phone_test_ref_spec_at_valthr": float(test_ref_spec),
                "phone_test_ref_spec_at_valthr_boot_mean": boot["ref_spec_boot_mean"],
                "phone_test_ref_spec_at_valthr_ci_low": boot["ref_spec_ci_low"],
                "phone_test_ref_spec_at_valthr_ci_high": boot["ref_spec_ci_high"],
                "phone_test_per_class_auc": phone_test.per_class_auc.tolist(),
                "phone_test_per_class_precision": phone_metrics[
                    "per_class_precision"
                ].tolist(),
                "phone_test_per_class_recall": phone_metrics["per_class_recall"].tolist(),
                "phone_test_per_class_f1": phone_metrics["per_class_f1"].tolist(),
                "phone_test_per_class_support": phone_metrics[
                    "per_class_support"
                ].tolist(),
                "phone_test_confusion_matrix": phone_metrics["confusion_matrix"].tolist(),
                "phone_test_sens": phone_test.sens.tolist(),
                "phone_test_spec": phone_test.spec.tolist(),
                "phone_test_sens_at_spec90_ovr": (
                    phone_test.sens90.tolist()
                    if hasattr(phone_test.sens90, "tolist")
                    else phone_test.sens90
                ),
                "phone_test_thr_at_spec90_ovr": (
                    phone_test.thr90.tolist()
                    if hasattr(phone_test.thr90, "tolist")
                    else phone_test.thr90
                ),
            }
        )

    atomic_json_save(
        {
            "training_signature": signature,
            "evaluation_signature": evaluation_signature(config),
            "rows": rows,
        },
        artifacts["rows"],
    )
    return rows
