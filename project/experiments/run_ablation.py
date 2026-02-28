from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from project.data.load_datasets import (
    load_california_housing,
    load_used_cars,
    split_data,
)
from project.experiments.evaluate import empirical_bias_variance, r_squared, rmse
from project.experiments.profiles import get_profile
from project.models.random_forest import RandomForestRegressorScratch

CONFIGS: dict[str, dict[str, object]] = {
    "A_SingleTree": {
        "n_estimators": 1,
        "use_bootstrap": False,
        "use_feature_subsampling": False,
    },
    "B_BaggingOnly": {
        "n_estimators": 100,
        "use_bootstrap": True,
        "use_feature_subsampling": False,
    },
    "C_FeatureRandOnly": {
        "n_estimators": 100,
        "use_bootstrap": False,
        "use_feature_subsampling": True,
    },
    "D_FullRandomForest": {
        "n_estimators": 100,
        "use_bootstrap": True,
        "use_feature_subsampling": True,
    },
}

SEEDS: list[int] = [42, 123, 7, 999, 2024, 31, 55, 88, 200, 314]

DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, list[str]]]] = {
    "UsedCars": load_used_cars,
    "CaliforniaHousing": load_california_housing,
}

FAST_CONFIGS: dict[str, dict[str, object]] = get_profile("fast").configs
FAST_SEEDS: list[int] = get_profile("fast").seeds


def _checkpoint_results(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _fit_seed_worker(
    config: dict[str, object],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | int]:
    start = perf_counter()
    model = RandomForestRegressorScratch(
        **config,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    end = perf_counter()

    y_pred = model.predict(X_test)
    return {
        "seed": int(seed),
        "r2": float(r_squared(y_test, y_pred)),
        "rmse": float(rmse(y_test, y_pred)),
        "train_time": float(end - start),
    }


def run_ablation(
    results_csv_path: str = "project/results/ablation_results.csv",
    configs: dict[str, dict[str, object]] | None = None,
    seeds: list[int] | None = None,
    datasets: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, list[str]]]] | None = None,
    n_bias_bootstrap: int = 30,
    verbose: bool = True,
    checkpoint_every_config: bool = True,
    max_samples_per_dataset: int | None = None,
    seed_results_csv_path: str | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    active_configs = configs if configs is not None else CONFIGS
    active_seeds = seeds if seeds is not None else SEEDS
    active_datasets = datasets if datasets is not None else DATASETS

    rows: list[dict[str, float | str]] = []
    seed_rows: list[dict[str, float | int | str]] = []
    output_path = Path(results_csv_path)
    total_dataset_configs = len(active_datasets) * len(active_configs)
    completed_dataset_configs = 0
    run_start = perf_counter()

    if verbose:
        print(
            "[run_ablation] Starting "
            f"{len(active_datasets)} dataset(s) x {len(active_configs)} config(s) x "
            f"{len(active_seeds)} seed(s), bias_bootstrap={n_bias_bootstrap}"
        )

    for dataset_name, load_fn in active_datasets.items():
        dataset_start = perf_counter()
        X, y, _ = load_fn()
        if max_samples_per_dataset is not None and len(y) > max_samples_per_dataset:
            sample_rng = np.random.RandomState(0)
            sample_indices = sample_rng.choice(len(y), size=max_samples_per_dataset, replace=False)
            X = X[sample_indices]
            y = y[sample_indices]

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=0)
        if verbose:
            print(
                f"[dataset] {dataset_name}: X={X.shape}, train={X_train.shape}, test={X_test.shape}"
            )

        for config_name, config in active_configs.items():
            config_start = perf_counter()
            results_per_seed: list[dict[str, float | int]] = []
            if verbose:
                print(f"[config-start] {dataset_name} | {config_name}")

            if n_jobs <= 1:
                for seed_idx, seed in enumerate(active_seeds, start=1):
                    result = _fit_seed_worker(config, seed, X_train, y_train, X_test, y_test)
                    results_per_seed.append(result)
                    seed_rows.append(
                        {
                            "dataset": dataset_name,
                            "config": config_name,
                            "seed": int(seed),
                            "r2": float(result["r2"]),
                            "rmse": float(result["rmse"]),
                            "train_time": float(result["train_time"]),
                            "n_estimators": int(config.get("n_estimators", 0)),
                            "use_bootstrap": bool(config.get("use_bootstrap", False)),
                            "use_feature_subsampling": bool(config.get("use_feature_subsampling", False)),
                        }
                    )
                    if verbose:
                        elapsed = perf_counter() - config_start
                        print(
                            f"  [seed {seed_idx}/{len(active_seeds)}] seed={seed} "
                            f"r2={result['r2']:.4f} "
                            f"rmse={result['rmse']:.4f} "
                            f"train={result['train_time']:.2f}s "
                            f"elapsed={elapsed:.1f}s"
                        )
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_seed = {
                        executor.submit(
                            _fit_seed_worker,
                            config,
                            seed,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                        ): seed
                        for seed in active_seeds
                    }

                    completed = 0
                    for future in as_completed(future_to_seed):
                        seed = future_to_seed[future]
                        result = future.result()
                        results_per_seed.append(result)
                        seed_rows.append(
                            {
                                "dataset": dataset_name,
                                "config": config_name,
                                "seed": int(seed),
                                "r2": float(result["r2"]),
                                "rmse": float(result["rmse"]),
                                "train_time": float(result["train_time"]),
                                "n_estimators": int(config.get("n_estimators", 0)),
                                "use_bootstrap": bool(config.get("use_bootstrap", False)),
                                "use_feature_subsampling": bool(config.get("use_feature_subsampling", False)),
                            }
                        )
                        completed += 1
                        if verbose:
                            elapsed = perf_counter() - config_start
                            print(
                                f"  [seed {completed}/{len(active_seeds)}] seed={seed} "
                                f"r2={result['r2']:.4f} "
                                f"rmse={result['rmse']:.4f} "
                                f"train={result['train_time']:.2f}s "
                                f"elapsed={elapsed:.1f}s"
                            )

            if verbose:
                print(
                    f"  [bias-variance] {dataset_name} | {config_name} "
                    f"bootstrap_models={n_bias_bootstrap}"
                )

            bias_squared, variance = empirical_bias_variance(
                model_class=RandomForestRegressorScratch,
                model_kwargs={**config, "random_state": 42},
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_bootstrap=n_bias_bootstrap,
                seed=42,
                n_jobs=n_jobs,
            )

            r2_values = np.array([float(r["r2"]) for r in results_per_seed], dtype=np.float64)
            rmse_values = np.array([float(r["rmse"]) for r in results_per_seed], dtype=np.float64)
            time_values = np.array([float(r["train_time"]) for r in results_per_seed], dtype=np.float64)

            rows.append(
                {
                    "dataset": dataset_name,
                    "config": config_name,
                    "mean_r2": float(np.mean(r2_values)),
                    "std_r2": float(np.std(r2_values)),
                    "mean_rmse": float(np.mean(rmse_values)),
                    "std_rmse": float(np.std(rmse_values)),
                    "mean_train_time": float(np.mean(time_values)),
                    "std_train_time": float(np.std(time_values)),
                    "bias_squared": float(bias_squared),
                    "variance": float(variance),
                }
            )

            completed_dataset_configs += 1
            config_elapsed = perf_counter() - config_start
            total_elapsed = perf_counter() - run_start
            if verbose:
                print(
                    f"[config-done] {dataset_name} | {config_name} in {config_elapsed:.1f}s "
                    f"({completed_dataset_configs}/{total_dataset_configs}) total_elapsed={total_elapsed:.1f}s"
                )

            if checkpoint_every_config:
                _checkpoint_results(rows, output_path)
                if seed_results_csv_path is not None:
                    seed_output_path = Path(seed_results_csv_path)
                    seed_output_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(seed_rows).to_csv(seed_output_path, index=False)
                if verbose:
                    print(f"[checkpoint] saved partial results -> {output_path}")

        if verbose:
            print(f"[dataset-done] {dataset_name} in {(perf_counter() - dataset_start):.1f}s")

    results_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    if seed_results_csv_path is not None:
        seed_output_path = Path(seed_results_csv_path)
        seed_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(seed_rows).to_csv(seed_output_path, index=False)
    if verbose:
        print(f"[run_ablation] Done in {(perf_counter() - run_start):.1f}s. Results -> {output_path}")
    return results_df
