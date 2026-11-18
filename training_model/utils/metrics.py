from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def safe_auc(labels, probabilities) -> float:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probabilities).astype(float)
    if len(np.unique(labels_np)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_np, probs_np))


def safe_ap(labels, probabilities) -> float:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probabilities).astype(float)
    if len(np.unique(labels_np)) < 2:
        return float("nan")
    return float(average_precision_score(labels_np, probs_np))


def compute_eer(labels, probabilities) -> float:
    """Compute Equal Error Rate (EER) where FAR == FRR."""
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probabilities).astype(float)
    if len(np.unique(labels_np)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels_np, probs_np, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(0.5 * (fpr[idx] + fnr[idx]))


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
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds_np)),
        "f1": float(f1_score(labels_np, preds_np, zero_division=0)),
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "auc": safe_auc(labels_np, probs_np),
        "ap": safe_ap(labels_np, probs_np),
        "eer": compute_eer(labels_np, probs_np),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def extract_video_id(image_path: str) -> str:
    """Use the full parent directory as the video ID; each folder holds one video."""
    return Path(image_path).parent.as_posix()


def aggregate_video_predictions(
    image_paths: Sequence[str],
    labels: Sequence[float],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> Tuple[Dict[str, object], List[str], List[int], List[float]]:
    """Average frame probabilities within each video folder's full path."""
    grouped_probs: Dict[str, List[float]] = defaultdict(list)
    grouped_labels: Dict[str, List[int]] = defaultdict(list)

    for path, label, prob in zip(image_paths, labels, probabilities):
        vid = extract_video_id(path)
        grouped_probs[vid].append(float(prob))
        grouped_labels[vid].append(int(round(float(label))))

    video_ids: List[str] = []
    video_labels: List[int] = []
    video_probs: List[float] = []
    for vid in sorted(grouped_probs):
        video_ids.append(vid)
        video_labels.append(int(round(float(np.mean(grouped_labels[vid])))))
        video_probs.append(float(np.mean(grouped_probs[vid])))

    metrics = compute_binary_metrics(video_labels, video_probs, threshold=threshold)
    metrics["num_videos"] = len(video_ids)
    return metrics, video_ids, video_labels, video_probs


def binary_confusion_matrix(labels, probabilities, threshold: float = 0.5):
    labels_np = np.asarray(labels).astype(int)
    preds_np = (np.asarray(probabilities).astype(float) >= threshold).astype(int)
    return confusion_matrix(labels_np, preds_np, labels=[0, 1])


def format_metrics(metrics: Dict[str, object]) -> str:
    parts = []
    for key in ["loss", "accuracy", "balanced_accuracy", "f1", "precision", "recall", "auc", "eer", "ap"]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
    return " | ".join(parts)
