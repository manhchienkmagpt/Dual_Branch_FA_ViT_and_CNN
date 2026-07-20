from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(
    labels,
    probabilities,
    threshold: float = 0.5,
    loss: Optional[float] = None,
) -> Dict[str, object]:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probabilities).astype(float)
    preds_np = (probs_np >= threshold).astype(int)

    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(labels_np, preds_np),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "auc": safe_auc(labels_np, probs_np),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def safe_auc(labels, probabilities) -> float:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probabilities).astype(float)
    if len(np.unique(labels_np)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_np, probs_np))


def binary_confusion_matrix(labels, probabilities, threshold: float = 0.5):
    labels_np = np.asarray(labels).astype(int)
    preds_np = (np.asarray(probabilities).astype(float) >= threshold).astype(int)
    return confusion_matrix(labels_np, preds_np, labels=[0, 1])


def format_metrics(metrics: Dict[str, object]) -> str:
    parts = []
    for key in ["loss", "accuracy", "f1", "precision", "recall", "auc"]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
    return " | ".join(parts)
