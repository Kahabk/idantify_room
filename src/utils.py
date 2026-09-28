"""Small shared utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def l2_normalize(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, eps)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 matrix with every row L2-normalized."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a one- or two-dimensional array")
    return np.ascontiguousarray(l2_normalize(matrix, axis=1), dtype=np.float32)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_label(label: str) -> str:
    return label.replace("_", " ").strip().title()
