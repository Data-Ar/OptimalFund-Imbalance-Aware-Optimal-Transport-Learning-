"""Shared helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def extract_patient_id(image_path: str) -> str:
    """Parse a patient identifier from a fundus image filename."""
    stem = Path(image_path).stem
    pieces = stem.rsplit(".", 1)
    if len(pieces) == 2 and pieces[1].isdigit():
        return pieces[0]
    return stem


def dataset_fingerprint(paths: np.ndarray, labels: np.ndarray) -> str:
    """Stable hash of (path, label) pairs used to tag prediction CSVs."""
    digest = hashlib.sha256()
    for image_path, label in zip(paths, labels):
        digest.update(str(image_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
