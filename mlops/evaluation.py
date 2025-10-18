"""Evaluation utilities for classification tasks."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute standard classification metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
    }
