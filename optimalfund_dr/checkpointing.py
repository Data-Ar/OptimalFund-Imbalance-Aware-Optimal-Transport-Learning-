"""Checkpoint, prediction, and confusion-matrix I/O."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import Config, cfg
from .utils import dataset_fingerprint, extract_patient_id

CHECKPOINT_SCHEMA_VERSION = 2


def prediction_method_name(ot_mode: str) -> str:
    return "erm" if ot_mode == "none" else ot_mode


def training_signature(backbone: str, seed: int, config: Config | None = None) -> dict[str, Any]:
    config = config or cfg
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "backbone": backbone,
        "seed": int(seed),
        "ot_mode": config.ot_mode,
        "ot_lambda": float(config.ot_lambda),
        "num_classes": int(config.num_classes),
        "img_size": int(config.img_size),
        "batch_size": int(config.batch_size),
        "steps_multiplier": int(config.steps_multiplier),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "deterministic_training": bool(config.deterministic_training),
        "use_focal_loss": bool(config.use_focal_loss),
        "focal_gamma": float(config.focal_gamma),
        "use_phone_weighted_sampler": bool(config.use_phone_weighted_sampler),
        "use_feature_norm": bool(config.use_feature_norm),
        "sinkhorn_eps": float(config.sinkhorn_eps),
        "sinkhorn_iters": int(config.sinkhorn_iters),
        "early_stopping_patience": int(config.early_stopping_patience),
        "early_stopping_min_delta": float(config.early_stopping_min_delta),
        "hosp_root": str(Path(config.hosp_root).resolve()),
        "phone_root_clean": str(Path(config.phone_root_clean).resolve()),
    }


def evaluation_signature(config: Config | None = None) -> dict[str, Any]:
    config = config or cfg
    return {
        "metrics_schema_version": 2,
        "phone_eval_severities": list(config.phone_eval_severities),
        "ref_classes": list(config.ref_classes),
        "target_spec": float(config.target_spec),
        "n_boot": int(config.n_boot),
        "bootstrap_all_severities": bool(config.bootstrap_all_severities),
        "save_predictions": bool(config.save_predictions),
        "phone_root_mms": str(Path(config.phone_root_mms).resolve()),
    }


def run_artifact_paths(
    backbone: str, seed: int, config: Config | None = None
) -> dict[str, Path]:
    config = config or cfg
    run_dir = (
        Path(config.out_dir)
        / "checkpoints"
        / config.ot_mode
        / backbone
        / f"seed_{seed}"
    )
    return {
        "run_dir": run_dir,
        "last": run_dir / "last.pt",
        "best": run_dir / "best.pt",
        "rows": run_dir / "evaluation_rows.json",
    }


def prediction_run_dir(
    backbone: str, seed: int, config: Config | None = None
) -> Path:
    config = config or cfg
    return (
        Path(config.out_dir)
        / "predictions"
        / prediction_method_name(config.ot_mode)
        / backbone
        / f"seed_{seed}"
    )


def prediction_path(
    backbone: str, seed: int, severity: str, config: Config | None = None
) -> Path:
    return prediction_run_dir(backbone, seed, config=config) / f"phone_test_{severity}.csv"


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def load_torch_checkpoint(path: Path, map_location="cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def model_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def assert_matching_signature(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = checkpoint.get("training_signature")
    if actual != expected:
        raise RuntimeError(
            "Checkpoint configuration does not match the requested experiment. "
            f"Expected {expected}, found {actual}. Use a matching configuration or "
            "start a new run with --overwrite."
        )


def save_prediction_csv(
    backbone: str,
    seed: int,
    severity: str,
    paths: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_filename: str | None = None,
    config: Config | None = None,
) -> Path:
    config = config or cfg
    if not (len(paths) == len(labels) == len(probabilities)):
        raise ValueError("Prediction paths, labels, and probabilities have different lengths.")
    if probabilities.shape[1] != config.num_classes:
        raise ValueError(f"Expected {config.num_classes} probability columns.")

    output_path = (
        prediction_run_dir(backbone, seed, config=config) / output_filename
        if output_filename is not None
        else prediction_path(backbone, seed, severity, config=config)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    fingerprint = dataset_fingerprint(paths, labels)
    probability_columns = [f"prob_{index}" for index in range(config.num_classes)]
    fields = [
        "method",
        "backbone",
        "seed",
        "condition",
        "dataset_fingerprint",
        "image_id",
        "patient_id",
        "y_true",
        "predicted_class",
        *probability_columns,
    ]
    predicted = probabilities.argmax(axis=1)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(labels)):
            row = {
                "method": prediction_method_name(config.ot_mode),
                "backbone": backbone,
                "seed": int(seed),
                "condition": severity,
                "dataset_fingerprint": fingerprint,
                "image_id": str(paths[index]),
                "patient_id": extract_patient_id(str(paths[index])),
                "y_true": int(labels[index]),
                "predicted_class": int(predicted[index]),
            }
            for class_index, column in enumerate(probability_columns):
                row[column] = float(probabilities[index, class_index])
            writer.writerow(row)
    os.replace(temporary, output_path)
    print("Saved predictions:", output_path)
    return output_path


def save_confusion_matrix_csv(
    backbone: str,
    seed: int,
    matrix: np.ndarray,
    output_filename: str,
    config: Config | None = None,
) -> Path:
    config = config or cfg
    matrix = np.asarray(matrix, dtype=int)
    expected_shape = (config.num_classes, config.num_classes)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected confusion matrix shape {expected_shape}, found {matrix.shape}."
        )
    output_path = prediction_run_dir(backbone, seed, config=config) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    fields = ["true_class", *[f"pred_{index}" for index in range(config.num_classes)]]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for true_class in range(config.num_classes):
            row = {"true_class": true_class}
            for predicted_class in range(config.num_classes):
                row[f"pred_{predicted_class}"] = int(matrix[true_class, predicted_class])
            writer.writerow(row)
    os.replace(temporary, output_path)
    print("Saved confusion matrix:", output_path)
    return output_path


def cached_evaluation_rows(
    path: Path,
    expected_training_signature: dict[str, Any],
    config: Config | None = None,
) -> list[dict[str, Any]] | None:
    config = config or cfg
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("training_signature") != expected_training_signature:
        return None
    if payload.get("evaluation_signature") != evaluation_signature(config):
        return None
    run_dir = prediction_run_dir(
        expected_training_signature["backbone"],
        expected_training_signature["seed"],
        config=config,
    )
    if not (run_dir / "confusion_matrix_hospital_test.csv").exists():
        return None
    if not (run_dir / "confusion_matrix_phone_val_clean.csv").exists():
        return None
    for severity in config.phone_eval_severities:
        if not (run_dir / f"confusion_matrix_phone_test_{severity}.csv").exists():
            return None
    if config.save_predictions:
        if not (run_dir / "hospital_test.csv").exists():
            return None
        if not (run_dir / "phone_val_clean.csv").exists():
            return None
        for severity in config.phone_eval_severities:
            if not prediction_path(
                expected_training_signature["backbone"],
                expected_training_signature["seed"],
                severity,
                config=config,
            ).exists():
                return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else None
