"""
Evaluation Metrics Module for Predictive Maintenance & Failure Detection System
Implements standard classification metrics and domain-specific prognostic metrics:
- Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- MAE, RMSE, R2, NASA Asymmetric PHM Scoring Function
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Classifier"
) -> Dict[str, Any]:
    """
    Compute full classification suite with focus on industrial safety (Recall & False Negatives).

    Args:
        y_true: Ground truth binary labels (0: healthy, 1: failure).
        y_pred: Predicted class labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (failure).
        model_name: Name of model for reporting.

    Returns:
        Dict[str, Any]: Metrics dictionary including confusion matrix elements.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    return {
        "model_name": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    }


def nasa_asymmetric_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    NASA Prognostics Center of Excellence (PCoE) Asymmetric Scoring Function.
    Penalizes late predictions (d > 0) exponentially harder than early predictions (d < 0).

    Formula:
    d = y_pred - y_true
    s_i = exp(-d / 13) - 1   for d < 0  (early prediction)
    s_i = exp(d / 10) - 1    for d >= 0 (late prediction)

    Args:
        y_true: Ground truth Remaining Useful Life (cycles).
        y_pred: Estimated Remaining Useful Life (cycles).

    Returns:
        Tuple[float, float]: (total_nasa_score, mean_nasa_score)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true

    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    total_score = float(np.sum(scores))
    mean_score = float(np.mean(scores))
    return total_score, mean_score


def evaluate_rul_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "RUL Regressor"
) -> Dict[str, Any]:
    """
    Evaluate Remaining Useful Life regressor performance.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        model_name: Model identifier.

    Returns:
        Dict[str, Any]: Metrics containing MAE, RMSE, R2, and NASA asymmetric scores.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    total_score, mean_score = nasa_asymmetric_scoring(y_true, y_pred)

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "nasa_total_score": total_score,
        "nasa_mean_score": mean_score
    }
