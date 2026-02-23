from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_ORDER = [
    "A_SingleTree",
    "B_BaggingOnly",
    "C_FeatureRandOnly",
    "D_FullRandomForest",
]


def _prepare(results_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    datasets = list(dict.fromkeys(results_df["dataset"].tolist()))
    configs = [cfg for cfg in CONFIG_ORDER if cfg in set(results_df["config"].tolist())]
    return datasets, configs


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bar_r2(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        means = [float(subset.loc[cfg, "mean_r2"]) for cfg in configs]
        stds = [float(subset.loc[cfg, "std_r2"]) for cfg in configs]
        ax.bar(x + idx * width - 0.4 + width / 2, means, width=width, yerr=stds, capsize=4, label=dataset)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15)
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Mean R²")
    ax.set_title("Mean R² by Model Configuration")
    ax.legend()

    out = Path(output_dir) / "bar_r2.png"
    _save(fig, out)
    return out


def plot_bar_rmse(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        means = [float(subset.loc[cfg, "mean_rmse"]) for cfg in configs]
        stds = [float(subset.loc[cfg, "std_rmse"]) for cfg in configs]
        ax.bar(x + idx * width - 0.4 + width / 2, means, width=width, yerr=stds, capsize=4, label=dataset)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15)
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Mean RMSE by Model Configuration")
    ax.legend()

    out = Path(output_dir) / "bar_rmse.png"
    _save(fig, out)
    return out


def plot_bias_variance_decomposition(
    results_df: pd.DataFrame,
    output_dir: str = "project/results",
) -> Path:
    datasets, configs = _prepare(results_df)
    n_datasets = max(1, len(datasets))

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        bias_vals = np.array([float(subset.loc[cfg, "bias_squared"]) for cfg in configs], dtype=np.float64)
        var_vals = np.array([float(subset.loc[cfg, "variance"]) for cfg in configs], dtype=np.float64)
        x = np.arange(len(configs))

        ax.bar(x, bias_vals, label="Bias²")
        ax.bar(x, var_vals, bottom=bias_vals, label="Variance")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=15)
        ax.set_xlabel("Model Configuration")
        ax.set_ylabel("Error Component")
        ax.set_title(f"Bias–Variance: {dataset}")
        ax.legend()

    fig.suptitle("Bias-Variance Decomposition by Configuration")
    out = Path(output_dir) / "bias_variance_decomposition.png"
    _save(fig, out)
    return out


def plot_training_time(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        means = [float(subset.loc[cfg, "mean_train_time"]) for cfg in configs]
        stds = [float(subset.loc[cfg, "std_train_time"]) for cfg in configs]
        ax.bar(x + idx * width - 0.4 + width / 2, means, width=width, yerr=stds, capsize=4, label=dataset)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15)
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Mean Training Time (s)")
    ax.set_title("Mean Training Time by Model Configuration")
    ax.legend()

    out = Path(output_dir) / "training_time.png"
    _save(fig, out)
    return out


def generate_all_plots(
    results_df: pd.DataFrame | None = None,
    results_csv_path: str = "project/results/ablation_results.csv",
    output_dir: str = "project/results",
) -> list[Path]:
    if results_df is None:
        results_df = pd.read_csv(results_csv_path)

    return [
        plot_bar_r2(results_df, output_dir=output_dir),
        plot_bar_rmse(results_df, output_dir=output_dir),
        plot_bias_variance_decomposition(results_df, output_dir=output_dir),
        plot_training_time(results_df, output_dir=output_dir),
    ]
