"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import cfg, required_dataset_paths
from .data import initialize_data
from .sweep import run_all_ot_modes


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="optimalfund-dr",
        description=(
            "Train DR adaptation methods with resumable checkpoints, early stopping, "
            "and per-image probability exports."
        ),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--hosp-root", default=None, help="Hospital dataset root.")
    parser.add_argument("--phone-root-clean", default=None, help="Clean phone dataset root.")
    parser.add_argument(
        "--phone-root-mms",
        default=None,
        help="Root containing mild/moderate/severe phone splits.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ot-lambda", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["none", "prototype", "sinkhorn", "class_sinkhorn"],
        default=None,
    )
    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete runs and reuse completed evaluations when available.",
    )
    control.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain and overwrite matching run artifacts.",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Disable per-image prediction and probability CSV files.",
    )
    parser.add_argument(
        "--bootstrap-clean-only",
        action="store_true",
        help="Skip per-run bootstrap intervals for degraded test conditions.",
    )
    args = parser.parse_args(argv)

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be at least 1.")
    if args.n_boot is not None and args.n_boot < 1:
        parser.error("--n-boot must be at least 1.")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contains duplicate values.")
    if args.methods is not None and len(set(args.methods)) != len(args.methods):
        parser.error("--methods contains duplicate values.")
    return args


def apply_cli_args(args: argparse.Namespace) -> None:
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.hosp_root is not None:
        cfg.hosp_root = args.hosp_root
    if args.phone_root_clean is not None:
        cfg.phone_root_clean = args.phone_root_clean
    if args.phone_root_mms is not None:
        cfg.phone_root_mms = args.phone_root_mms
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.patience is not None:
        cfg.early_stopping_patience = args.patience
    if args.n_boot is not None:
        cfg.n_boot = args.n_boot
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.ot_lambda is not None:
        cfg.ot_lambda = args.ot_lambda
    if args.seeds is not None:
        cfg.seeds = tuple(args.seeds)
    if args.backbones is not None:
        cfg.backbones = tuple(args.backbones)
    if args.methods is not None:
        cfg.ot_modes = tuple(args.methods)

    cfg.resume = bool(args.resume)
    cfg.overwrite_existing = bool(args.overwrite)
    cfg.save_predictions = not bool(args.no_save_predictions)
    if args.bootstrap_clean_only:
        cfg.bootstrap_all_severities = False
    cfg.phone_root = cfg.phone_root_clean


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_args(argv)
    apply_cli_args(args)

    missing = required_dataset_paths(cfg)
    if missing:
        raise SystemExit(
            "Dataset paths must be provided. Missing: "
            + ", ".join(missing)
            + ". Pass --hosp-root, --phone-root-clean, and --phone-root-mms."
        )

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    print("Effective configuration:", cfg)
    data = initialize_data(cfg)
    run_all_ot_modes(data, cfg)


if __name__ == "__main__":
    main()
