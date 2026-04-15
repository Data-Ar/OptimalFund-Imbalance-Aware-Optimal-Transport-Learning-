import numpy as np
import torch
from torch.amp import autocast

from model.dr_model import DRModel
from loss.focal import get_criterion
from loss.ot import compute_ot_loss
from util.reproducibility import set_seed
from util.metrics import evaluate, choose_ref_threshold_at_spec, refsens_spec_at_threshold, bootstrap_ci_image_level


def train_one_epoch(model, loader_h, loader_p, opt, criterion, cfg, scaler=None):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_ot = 0.0

    # phone-anchored epoch length
    steps_multiplier = cfg.steps_multiplier
    num_steps = len(loader_p) * steps_multiplier

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

        xp = xp.to(cfg.device, non_blocking=True)
        yp = yp.to(cfg.device, non_blocking=True)
        xh = xh.to(cfg.device, non_blocking=True)
        yh = yh.to(cfg.device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        use_amp = (scaler is not None) and cfg.use_amp and str(cfg.device).startswith("cuda")

        if use_amp:
            with autocast("cuda", enabled=True):
                logits_h, feat_h = model(xh, return_feat=True)
                logits_p, feat_p = model(xp, return_feat=True)

                loss_ce = criterion(logits_h, yh) + criterion(logits_p, yp)
                loss_ot = compute_ot_loss(feat_h, yh, feat_p, yp, cfg)
                loss = loss_ce + cfg.ot_lambda * loss_ot

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss encountered: {loss.item()}")
                continue

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(opt)
            scaler.update()

        else:
            logits_h, feat_h = model(xh, return_feat=True)
            logits_p, feat_p = model(xp, return_feat=True)

            loss_ce = criterion(logits_h, yh) + criterion(logits_p, yp)
            loss_ot = compute_ot_loss(feat_h, yh, feat_p, yp, cfg)
            loss = loss_ce + cfg.ot_lambda * loss_ot

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss encountered: {loss.item()}")
                continue

            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

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


def run_one(backbone: str, seed: int, cfg, data_objects):
    set_seed(seed)
    model = DRModel(backbone, cfg.num_classes, use_feature_norm=cfg.use_feature_norm).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = get_criterion(cfg)

    best_val_auc, best_state, best_ref_thr = -1.0, None, float("nan")

    for epoch in range(cfg.epochs):
        tr_stats = train_one_epoch(model, data_objects["ld_h_train"], data_objects["ld_p_train"], optimizer, criterion, cfg)

        val_macro_auc, _, val_sens, val_spec, _, _, _, val_probs, val_labels = evaluate(model, data_objects["ld_p_val"])
        macro_sens, macro_spec = float(np.mean(val_sens)), float(np.mean(val_spec))

        ref_sens90, ref_spec90, ref_thr90 = choose_ref_threshold_at_spec(
            val_labels, val_probs, cfg.ref_classes, cfg.target_spec
        )

        if val_macro_auc > best_val_auc:
            best_val_auc = val_macro_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_ref_thr = ref_thr90

        print(
            f"Epoch {epoch} "
            f"Loss={tr_stats['loss']:.4f} "
            f"CE={tr_stats['loss_ce']:.4f} "
            f"OT={tr_stats['loss_ot']:.4f} "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )
        print(
            f"PHONE VAL  | AUC={val_macro_auc:.4f}  "
            f"macro_sens={macro_sens:.4f}  macro_spec={macro_spec:.4f}  "
            f"RefSens@Spec0.90={ref_sens90:.4f}"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})

    # Hospital test (single)
    hosp_test = evaluate(model, data_objects["ld_h_test"])

    # Phone test across severities (clean + MMS tiers)
    rows = []
    for sev in cfg.phone_eval_severities:
        sev_root = data_objects["phone_test_loaders"][sev]["root"]
        ld_p_test_sev = data_objects["phone_test_loaders"][sev]["loader"]

        phone_test = evaluate(model, ld_p_test_sev)

        # Apply best CLEAN phone-val threshold to this severity's phone-test referable DR
        p_test_probs, p_test_labels = phone_test[7], phone_test[8]
        test_ref_sens, test_ref_spec = refsens_spec_at_threshold(
            p_test_labels, p_test_probs, best_ref_thr, cfg.ref_classes
        )

        do_boot = (sev == "clean") or cfg.bootstrap_all_severities

        if do_boot:
            boot = bootstrap_ci_image_level(
                y_true_mc=p_test_labels,
                probs_mc=p_test_probs,
                ref_thr=best_ref_thr,
                ref_classes=cfg.ref_classes,
                num_classes=cfg.num_classes,
                n_boot=cfg.n_boot,
                alpha=0.95,
                seed=seed,
            )
        else:
            boot = {
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
        rows.append({
            "backbone": backbone,
            "seed": seed,
            "ot_mode": cfg.ot_mode,
            "ot_lambda": cfg.ot_lambda,
            "use_focal": cfg.use_focal_loss,
            "use_phone_sampler": cfg.use_phone_weighted_sampler,

            "best_phone_val_macro_auc": float(best_val_auc),
            "val_ref_thr_at_spec90": float(best_ref_thr),

            "hosp_test_macro_auc": float(hosp_test[0]),

            "phone_severity": sev,
            "phone_root": sev_root,

            "phone_test_macro_auc": float(phone_test[0]),
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

            "phone_test_per_class_auc": phone_test[1].tolist(),
            "phone_test_sens": phone_test[2].tolist(),
            "phone_test_spec": phone_test[3].tolist(),
            "phone_test_sens_at_spec90_ovr": phone_test[5].tolist(),
            "phone_test_thr_at_spec90_ovr": phone_test[6].tolist(),
        })

    return rows
