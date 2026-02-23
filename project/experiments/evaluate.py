from __future__ import annotations

from typing import Any

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def empirical_bias_variance(
    model_class: type,
    model_kwargs: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> tuple[float, float]:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    n_train = len(y_train)
    rng = np.random.RandomState(seed)

    all_predictions: list[np.ndarray] = []
    base_seed = int(model_kwargs.get("random_state", 0))

    for i in range(n_bootstrap):
        bootstrap_indices = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train[bootstrap_indices]
        y_boot = y_train[bootstrap_indices]

        iter_kwargs = dict(model_kwargs)
        iter_kwargs["random_state"] = base_seed + i

        model = model_class(**iter_kwargs)
        model.fit(X_boot, y_boot)
        preds = model.predict(X_test)
        all_predictions.append(np.asarray(preds, dtype=np.float64))

    pred_matrix = np.vstack(all_predictions)
    mean_predictions = np.mean(pred_matrix, axis=0)

    bias_squared = float(np.mean((mean_predictions - y_test) ** 2))
    variance = float(np.mean(np.var(pred_matrix, axis=0)))
    return bias_squared, variance
