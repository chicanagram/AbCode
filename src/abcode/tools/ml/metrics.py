from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _safe_corr(corr_fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        value, _ = corr_fn(y_true, y_pred)
        if np.isnan(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_score: np.ndarray | None = None,
) -> Dict[str, Any]:
    task = str(task_type).strip().lower()

    if task == "regression":
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "spearman": _safe_corr(spearmanr, y_true, y_pred),
            "pearson": _safe_corr(pearsonr, y_true, y_pred),
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
        }

    if task == "classification":
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )

        out: Dict[str, Any] = {
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision_weighted": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        if y_score is not None:
            from sklearn.metrics import roc_auc_score

            try:
                out["roc_auc_ovr"] = float(roc_auc_score(y_true, y_score, multi_class="ovr"))
            except Exception:
                pass
        return out

    raise ValueError("task_type must be 'classification' or 'regression'.")
