from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from project.data.load_datasets import load_california_housing, load_used_cars, split_data
from project.experiments.evaluate import r_squared, rmse
from project.models.random_forest import RandomForestRegressorScratch

DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, list[str]]]] = {
    "UsedCars": load_used_cars,
    "CaliforniaHousing": load_california_housing,
}


def _calibration_worker(
    n_estimators: int,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | int]:
    t0 = perf_counter()
    model = RandomForestRegressorScratch(
        n_estimators=n_estimators,
        use_bootstrap=True,
        use_feature_subsampling=True,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - t0
    y_pred = model.predict(X_test)

    return {
        "seed": int(seed),
        "fit_seconds": float(fit_seconds),
        "seconds_per_tree": float(fit_seconds / max(1, n_estimators)),
        "r2": float(r_squared(y_test, y_pred)),
        "rmse": float(rmse(y_test, y_pred)),
    }


def run_calibration(
    output_csv_path: str = "project/results/runtime_calibration.csv",
    sample_sizes: list[int] | None = None,
    estimators_grid: list[int] | None = None,
    seeds: list[int] | None = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    active_sample_sizes = sample_sizes if sample_sizes is not None else [20000, 40000, 80000]
    active_estimators = estimators_grid if estimators_grid is not None else [25, 50, 100]
    active_seeds = seeds if seeds is not None else [42, 123]

    rows: list[dict[str, float | int | str]] = []

    if verbose:
        print(
            "[calibration] Starting "
            f"datasets={len(DATASETS)} sample_sizes={active_sample_sizes} "
            f"estimators={active_estimators} seeds={active_seeds}"
        )

    for dataset_name, load_fn in DATASETS.items():
        X, y, _ = load_fn()
        for n in active_sample_sizes:
            n_eff = min(n, len(y))
            sample_rng = np.random.RandomState(0)
            sample_idx = sample_rng.choice(len(y), size=n_eff, replace=False)
            X_sub = X[sample_idx]
            y_sub = y[sample_idx]

            X_train, X_test, y_train, y_test = split_data(X_sub, y_sub, test_size=0.2, seed=0)
            if verbose:
                print(
                    f"[calibration] {dataset_name} n={n_eff} train={X_train.shape[0]} test={X_test.shape[0]}"
                )

            for n_estimators in active_estimators:
                if n_jobs <= 1:
                    for seed in active_seeds:
                        result = _calibration_worker(
                            n_estimators,
                            seed,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                        )
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "n_samples": n_eff,
                                "n_train": int(X_train.shape[0]),
                                "n_test": int(X_test.shape[0]),
                                "n_estimators": n_estimators,
                                "seed": int(result["seed"]),
                                "fit_seconds": float(result["fit_seconds"]),
                                "seconds_per_tree": float(result["seconds_per_tree"]),
                                "r2": float(result["r2"]),
                                "rmse": float(result["rmse"]),
                            }
                        )
                        if verbose:
                            print(
                                f"  [done] est={n_estimators} seed={seed} "
                                f"fit={result['fit_seconds']:.2f}s r2={result['r2']:.4f} rmse={result['rmse']:.4f}"
                            )
                else:
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        futures = {
                            executor.submit(
                                _calibration_worker,
                                n_estimators,
                                seed,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                            ): seed
                            for seed in active_seeds
                        }
                        for future in as_completed(futures):
                            result = future.result()
                            rows.append(
                                {
                                    "dataset": dataset_name,
                                    "n_samples": n_eff,
                                    "n_train": int(X_train.shape[0]),
                                    "n_test": int(X_test.shape[0]),
                                    "n_estimators": n_estimators,
                                    "seed": int(result["seed"]),
                                    "fit_seconds": float(result["fit_seconds"]),
                                    "seconds_per_tree": float(result["seconds_per_tree"]),
                                    "r2": float(result["r2"]),
                                    "rmse": float(result["rmse"]),
                                }
                            )
                            if verbose:
                                print(
                                    f"  [done] est={n_estimators} seed={result['seed']} "
                                    f"fit={result['fit_seconds']:.2f}s r2={result['r2']:.4f} rmse={result['rmse']:.4f}"
                                )

    df = pd.DataFrame(rows)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = (
        df.groupby(["dataset", "n_samples", "n_estimators"], as_index=False)
        .agg(
            mean_fit_seconds=("fit_seconds", "mean"),
            std_fit_seconds=("fit_seconds", "std"),
            mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"),
        )
        .sort_values(["dataset", "n_samples", "n_estimators"])
    )
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    if verbose:
        print(f"[calibration] Saved detailed results -> {output_path}")
        print(f"[calibration] Saved summary results -> {summary_path}")
    return df
