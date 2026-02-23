from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _read_csv_resolved(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if path.exists():
        return pd.read_csv(path)

    project_path = Path("project") / path
    if project_path.exists():
        return pd.read_csv(project_path)

    raise FileNotFoundError(f"Dataset file not found: {filepath}")


def _find_target_column(columns: pd.Index, candidates: list[str]) -> str:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return str(normalized[candidate.lower()])
    raise ValueError(f"Target column not found. Tried: {candidates}")


def _encode_and_scale(df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = df.dropna(axis=0).copy()
    if data.empty:
        raise ValueError(
            "No rows remain after dropna(). Check missing values and target column quality."
        )

    feature_df = data.drop(columns=[target_column])
    if feature_df.shape[1] == 0:
        raise ValueError("No feature columns available after removing the target column.")

    y = data[target_column].astype(np.float64).to_numpy()
    if y.size == 0:
        raise ValueError("Target array is empty after preprocessing.")

    for col in feature_df.columns:
        if not is_numeric_dtype(feature_df[col]):
            feature_df[col] = pd.Categorical(feature_df[col]).codes

    X = feature_df.astype(np.float64).to_numpy()

    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds_safe = np.where(stds == 0.0, 1.0, stds)
    X = (X - means) / stds_safe

    feature_names = [str(name) for name in feature_df.columns]
    return X.astype(np.float64), y.astype(np.float64), feature_names


def load_used_cars(
    filepath: str = "data/used_cars.csv",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = _read_csv_resolved(filepath)
    target_column = _find_target_column(df.columns, ["price", "selling_price", "car_price"])

    # Keep target, remove extremely sparse feature columns, then row-drop remaining NaNs.
    missing_ratio = df.isna().mean()
    keep_columns = [target_column]
    keep_columns.extend(
        [
            col
            for col in df.columns
            if col != target_column and float(missing_ratio[col]) <= 0.40
        ]
    )
    df = df.loc[:, keep_columns]

    return _encode_and_scale(df, target_column)


def load_california_housing(
    filepath: str = "data/california_housing.csv",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = _read_csv_resolved(filepath)
    target_column = _find_target_column(
        df.columns,
        ["median_house_value", "medhouseval"],
    )
    return _encode_and_scale(df, target_column)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_samples = len(y)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to perform a train/test split.")

    n_test = int(np.floor(n_samples * test_size))
    n_test = max(1, n_test)
    if n_test >= n_samples:
        raise ValueError("test_size leaves no training samples. Use a smaller test_size.")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
