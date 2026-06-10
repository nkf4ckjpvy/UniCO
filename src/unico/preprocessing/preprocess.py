from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..data.windowing import sliding_windows


def downsample_csv(input_csv: Path, output_csv: Path, step: int = 10, seed: int = 42) -> None:
    df = pd.read_csv(input_csv)
    sampled = [df.iloc[start : start + step].sample(n=1, random_state=seed) for start in range(0, len(df), step)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(sampled).reset_index(drop=True).to_csv(output_csv, index=False)


def standardize_label_column(input_csv: Path, output_csv: Path, label_column: str, output_label: str = "label") -> None:
    df = pd.read_csv(input_csv)
    if label_column not in df.columns:
        raise ValueError(f"label column not found: {label_column}")
    df = df.rename(columns={label_column: output_label})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def generate_image_windows(
    input_csv: Path,
    output_npy: Path,
    label_column: str = "label",
    method: str = "gaf",
    window_size: int = 16,
    stride: int = 1,
    max_rows: int | None = 10000,
    max_channels: int | None = 24,
    image_size: int | None = None,
) -> None:
    try:
        from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
    except ImportError as exc:
        raise ImportError("Install pyts to generate image windows: pip install pyts") from exc

    df = pd.read_csv(input_csv).fillna(0)
    if label_column in df.columns:
        df = df.drop(columns=[label_column])
    if max_rows is not None:
        df = df.iloc[:max_rows]
    if max_channels is not None:
        df = df.iloc[:, :max_channels]

    values = MinMaxScaler(feature_range=(0, 1)).fit_transform(df.to_numpy(dtype=np.float32))
    windows = sliding_windows(values, window_size, stride).squeeze(1).transpose(0, 2, 1)
    n_windows, _, n_channels = windows.shape
    image_size = image_size or window_size

    if method == "gaf":
        transformer = GramianAngularField(image_size=image_size)
    elif method == "mtf":
        transformer = MarkovTransitionField(image_size=image_size)
    elif method == "rp":
        transformer = RecurrencePlot()
    else:
        raise ValueError("method must be one of: gaf, mtf, rp")

    images = np.empty((n_windows, n_channels, image_size, image_size), dtype=np.float32)
    for channel in range(n_channels):
        images[:, channel] = transformer.fit_transform(windows[:, :, channel]).astype(np.float32)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, images)
