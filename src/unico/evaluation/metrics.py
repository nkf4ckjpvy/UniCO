from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    binary_roc_auc: float
    score_roc_auc: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def point_adjust(predictions: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    adjusted = predictions.copy()
    for start, end in segments:
        if start >= len(adjusted):
            continue
        clipped_end = min(end, len(adjusted) - 1)
        if adjusted[start : clipped_end + 1].any():
            adjusted[start : clipped_end + 1] = 1
    return adjusted


def _roc_auc_or_default(labels: np.ndarray, values: np.ndarray, default: float | None) -> float | None:
    if len(np.unique(labels)) < 2:
        return default
    return float(roc_auc_score(labels, values))


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    segments: list[tuple[int, int]] | None = None,
    use_point_adjust: bool = True,
) -> tuple[np.ndarray, ThresholdResult]:
    predictions = (scores > threshold).astype(int)
    if use_point_adjust and segments:
        predictions = point_adjust(predictions, segments)

    binary_roc_auc = _roc_auc_or_default(labels, predictions, 0.0)
    score_roc_auc = _roc_auc_or_default(labels, scores, None)

    result = ThresholdResult(
        threshold=float(threshold),
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        binary_roc_auc=binary_roc_auc,
        score_roc_auc=score_roc_auc,
    )
    return predictions, result


def best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    segments: list[tuple[int, int]] | None = None,
    step: float = 0.001,
    use_point_adjust: bool = True,
) -> tuple[np.ndarray, ThresholdResult]:
    if scores.size == 0:
        raise ValueError("scores cannot be empty")

    lower = float(np.min(scores))
    upper = float(np.max(scores))
    if lower == upper:
        return evaluate_threshold(scores, labels, lower, segments, use_point_adjust)

    thresholds = np.arange(lower, upper + step, step)
    best_predictions, best_result = evaluate_threshold(
        scores, labels, thresholds[0], segments, use_point_adjust
    )
    for threshold in thresholds[1:]:
        predictions, result = evaluate_threshold(scores, labels, float(threshold), segments, use_point_adjust)
        if result.f1 > best_result.f1:
            best_predictions, best_result = predictions, result
    return best_predictions, best_result
