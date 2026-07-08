from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from .windowing import attack_segments, slice_array, sliding_window_labels, sliding_windows


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    attack_segments: list[tuple[int, int]]
    num_features: int
    image_channels: int


class MultiModalWindowDataset(Dataset):
    def __init__(
        self,
        table_windows: torch.Tensor,
        labels: torch.Tensor,
        image_windows: torch.Tensor,
        use_images: bool,
    ):
        self.table_windows = table_windows
        self.labels = labels
        self.image_windows = image_windows
        self.use_images = use_images

    def __len__(self) -> int:
        return self.table_windows.shape[0]

    def __getitem__(self, index: int):
        table_sample = self.table_windows[index]
        label = self.labels[index]
        if not self.use_images:
            return table_sample, label
        image_sample = self.image_windows[index % self.image_windows.shape[0]]
        return table_sample, image_sample, label


def _load_image_windows(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image window file not found: {image_path}")
    return torch.tensor(np.load(image_path), dtype=torch.float32)


def _tabular_bundle(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    image_path: Path,
    window_size: int,
    stride: int,
    batch_size: int,
    train_window_offset: int = 0,
    test_window_offset: int = 0,
    max_train_windows: int | None = None,
    max_test_windows: int | None = None,
    load_images: bool = True,
) -> LoaderBundle:
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    train_windows = sliding_windows(train_features, window_size, stride)
    train_window_labels = sliding_window_labels(train_labels, window_size, stride)
    test_windows = sliding_windows(test_features, window_size, stride)
    test_window_labels = sliding_window_labels(test_labels, window_size, stride)

    train_windows = slice_array(train_windows, train_window_offset, max_train_windows)
    train_window_labels = slice_array(train_window_labels, train_window_offset, max_train_windows)
    test_windows = slice_array(test_windows, test_window_offset, max_test_windows)
    test_window_labels = slice_array(test_window_labels, test_window_offset, max_test_windows)

    train_table = torch.tensor(train_windows, dtype=torch.float32)
    train_y = torch.tensor(train_window_labels, dtype=torch.long)
    test_table = torch.tensor(test_windows, dtype=torch.float32)
    test_y = torch.tensor(test_window_labels, dtype=torch.long)
    image_windows = _load_image_windows(image_path) if load_images else torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    train_dataset = MultiModalWindowDataset(train_table, train_y, image_windows, use_images=True)
    test_dataset = MultiModalWindowDataset(test_table, test_y, image_windows, use_images=False)

    return LoaderBundle(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        attack_segments=attack_segments(test_window_labels),
        num_features=train_table.shape[2],
        image_channels=image_windows.shape[1],
    )


def load_swat(
    data_dir: Path,
    normal_csv: str,
    attack_csv: str,
    image_npy: str,
    window_size: int,
    stride: int,
    batch_size: int,
    train_window_offset: int = 0,
    test_window_offset: int = 0,
    max_train_windows: int | None = None,
    max_test_windows: int | None = None,
    load_images: bool = True,
) -> LoaderBundle:
    normal_df = pd.read_csv(data_dir / normal_csv).fillna(0)
    attack_df = pd.read_csv(data_dir / attack_csv).fillna(0)
    return _tabular_bundle(
        normal_df.iloc[:, :-1].to_numpy(),
        normal_df.iloc[:, -1].to_numpy(),
        attack_df.iloc[:, :-1].to_numpy(),
        attack_df.iloc[:, -1].to_numpy(),
        data_dir / image_npy,
        window_size,
        stride,
        batch_size,
        train_window_offset,
        test_window_offset,
        max_train_windows,
        max_test_windows,
        load_images,
    )


def load_wadi(
    data_dir: Path,
    normal_csv: str,
    attack_csv: str,
    image_npy: str,
    window_size: int,
    stride: int,
    batch_size: int,
    attack_skip_rows: int = 8000,
    train_window_offset: int = 0,
    test_window_offset: int = 0,
    max_train_windows: int | None = None,
    max_test_windows: int | None = None,
    load_images: bool = True,
) -> LoaderBundle:
    normal_df = pd.read_csv(data_dir / normal_csv).fillna(0)
    attack_df = pd.read_csv(data_dir / attack_csv).fillna(0).iloc[attack_skip_rows:]
    return _tabular_bundle(
        normal_df.iloc[:, :-1].to_numpy(),
        normal_df.iloc[:, -1].to_numpy(),
        attack_df.iloc[:, :-1].to_numpy(),
        attack_df.iloc[:, -1].to_numpy(),
        data_dir / image_npy,
        window_size,
        stride,
        batch_size,
        train_window_offset,
        test_window_offset,
        max_train_windows,
        max_test_windows,
        load_images,
    )


def _kdd_columns() -> list[str]:
    return [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
    ]


def _load_kdd_xy(data_path: Path, anomaly_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path, header=None, names=_kdd_columns())
    labels = df["label"].copy()
    labels[labels == "normal."] = 0
    labels[labels != 0] = 1
    df["label"] = labels.astype(int)
    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag", "land", "logged_in", "is_host_login", "is_guest_login"])

    train_df = df.sample(frac=0.5, random_state=seed)
    test_df = df.loc[~df.index.isin(train_df.index)]
    feature_cols = [column for column in df.columns if column != "label"]

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=int)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=int)

    x_train = x_train[y_train == 0]
    y_train = y_train[y_train == 0]

    rng = np.random.default_rng(seed)
    normal_x = x_test[y_test == 0]
    normal_y = y_test[y_test == 0]
    anomaly_x = x_test[y_test == 1]
    anomaly_y = y_test[y_test == 1]
    n_anomaly = min(int(len(normal_x) * anomaly_ratio / (1 - anomaly_ratio)), len(anomaly_x))
    anomaly_idx = rng.permutation(len(anomaly_x))[:n_anomaly]

    x_test = np.concatenate([normal_x, anomaly_x[anomaly_idx]], axis=0)
    y_test = np.concatenate([normal_y, anomaly_y[anomaly_idx]], axis=0)
    order = rng.permutation(len(x_test))
    return x_train, y_train, x_test[order], y_test[order]


def load_kdd(
    data_dir: Path,
    data_file: str,
    image_npy: str,
    window_size: int,
    stride: int,
    batch_size: int,
    seed: int,
    anomaly_ratio: float = 0.2,
    load_images: bool = True,
) -> LoaderBundle:
    x_train, y_train, x_test, y_test = _load_kdd_xy(data_dir / data_file, anomaly_ratio, seed)
    return _tabular_bundle(
        x_train,
        y_train,
        x_test,
        y_test,
        data_dir / image_npy,
        window_size,
        stride,
        batch_size,
        load_images=load_images,
    )
