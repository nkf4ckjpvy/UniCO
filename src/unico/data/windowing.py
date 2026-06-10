from __future__ import annotations

import numpy as np


def sliding_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("data must have shape (rows, features)")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(data) < window_size:
        raise ValueError("data is shorter than window_size")

    starts = range(0, len(data) - window_size + 1, stride)
    windows = np.asarray([data[start : start + window_size] for start in starts])
    return np.expand_dims(windows.transpose(0, 2, 1), axis=1)


def sliding_window_labels(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if len(labels) < window_size:
        raise ValueError("labels are shorter than window_size")
    starts = range(0, len(labels) - window_size + 1, stride)
    return np.asarray([int(np.any(labels[start : start + window_size] == 1)) for start in starts])


def attack_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    index = 0
    while index < len(labels):
        if labels[index] == 1:
            start = index
            while index < len(labels) and labels[index] == 1:
                index += 1
            segments.append((start, index - 1))
        else:
            index += 1
    return segments


def slice_array(array: np.ndarray, offset: int = 0, limit: int | None = None) -> np.ndarray:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if offset >= len(array):
        return array[:0]
    sliced = array[offset:]
    return sliced[:limit] if limit is not None else sliced
