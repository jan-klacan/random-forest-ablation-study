from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_ORDER = [
    "A_SingleTree",
    "B_BaggingOnly",
    "C_FeatureRandOnly",
    "D_FullRandomForest",
]

CONFIG_LABELS = {
    "A_SingleTree": "A\nSingle",
    "B_BaggingOnly": "B\nBagging",
    "C_FeatureRandOnly": "C\nFeatRand",
    "D_FullRandomForest": "D\nFullRF",
}

DATASET_COLORS = {
    "UsedCars": "#4C78A8",
    "CaliforniaHousing": "#F58518",
}


def _apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#C7C7C7",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.facecolor": "white",
        }
    )


def _prepare(results_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    datasets = list(dict.fromkeys(results_df["dataset"].tolist()))
    configs = [cfg for cfg in CONFIG_ORDER if cfg in set(results_df["config"].tolist())]
    return datasets, configs


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", rotation: int = 90) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            (bar.get_x() + bar.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
            color="#333333",
        )


def plot_bar_r2(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    _apply_style()
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        means = [float(subset.loc[cfg, "mean_r2"]) for cfg in configs]
        stds = [float(subset.loc[cfg, "std_r2"]) for cfg in configs]
        bars = ax.bar(
            x + idx * width - 0.4 + width / 2,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=dataset,
            color=DATASET_COLORS.get(dataset, None),
            edgecolor="#444444",
            linewidth=0.5,
        )
        _annotate_bars(ax, bars, fmt="{:.3f}", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Mean R²")
    ax.set_title("Ablation Performance: Mean R² (±1 SD across seeds)")
    ax.set_ylim(0.55, max(0.80, float(results_df["mean_r2"].max()) + 0.03))
    ax.legend()

    out = Path(output_dir) / "bar_r2.png"
    _save(fig, out)
    return out


def plot_bar_rmse(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    _apply_style()
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        means = [float(subset.loc[cfg, "mean_rmse"]) for cfg in configs]
        stds = [float(subset.loc[cfg, "std_rmse"]) for cfg in configs]
        bars = ax.bar(
            x + idx * width - 0.4 + width / 2,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=dataset,
            color=DATASET_COLORS.get(dataset, None),
            edgecolor="#444444",
            linewidth=0.5,
        )
        _annotate_bars(ax, bars, fmt="{:.0f}", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Ablation Performance: Mean RMSE (±1 SD across seeds)")
    ax.legend()

    out = Path(output_dir) / "bar_rmse.png"
    _save(fig, out)
    return out


def plot_bias_variance_decomposition(
    results_df: pd.DataFrame,
    output_dir: str = "project/results",
) -> Path:
    _apply_style()
    datasets, configs = _prepare(results_df)
    n_datasets = max(1, len(datasets))

    fig, axes = plt.subplots(1, n_datasets, figsize=(6.8 * n_datasets, 5.5), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        bias_vals = np.array([float(subset.loc[cfg, "bias_squared"]) for cfg in configs], dtype=np.float64)
        var_vals = np.array([float(subset.loc[cfg, "variance"]) for cfg in configs], dtype=np.float64)
        x = np.arange(len(configs))

        ax.bar(x, bias_vals, label="Bias²", color="#4C78A8", edgecolor="#444444", linewidth=0.5)
        ax.bar(x, var_vals, bottom=bias_vals, label="Variance", color="#54A24B", edgecolor="#444444", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
        ax.set_xlabel("Model Configuration")
        ax.set_ylabel("Error Component (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"Bias–Variance: {dataset}")
        ax.legend()

    fig.suptitle("Bias-Variance Decomposition by Configuration")
    out = Path(output_dir) / "bias_variance_decomposition.png"
    _save(fig, out)
    return out


def plot_bias_variance_clarity(
    results_df: pd.DataFrame,
    output_dir: str = "project/results",
) -> Path:
    _apply_style()
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.36

    fig, axes = plt.subplots(2, len(datasets), figsize=(6.6 * len(datasets), 8), squeeze=False)

    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        bias_vals = np.array([float(subset.loc[cfg, "bias_squared"]) for cfg in configs], dtype=np.float64)
        var_vals = np.array([float(subset.loc[cfg, "variance"]) for cfg in configs], dtype=np.float64)

        ax_bias = axes[0, idx]
        bars_b = ax_bias.bar(
            x,
            bias_vals,
            color="#4C78A8",
            edgecolor="#444444",
            linewidth=0.5,
        )
        ax_bias.set_xticks(x)
        ax_bias.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
        ax_bias.set_title(f"Bias² (linear): {dataset}")
        ax_bias.set_ylabel("Bias²")
        ax_bias.set_ylim(0, max(1.05 * float(np.max(bias_vals)), 1.0))

        ax_var = axes[1, idx]
        bars_v = ax_var.bar(
            x,
            var_vals,
            color="#54A24B",
            edgecolor="#444444",
            linewidth=0.5,
        )
        ax_var.set_xticks(x)
        ax_var.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
        ax_var.set_title(f"Variance (linear): {dataset}")
        ax_var.set_ylabel("Variance")
        ax_var.set_ylim(0, max(1.10 * float(np.max(var_vals)), 1.0))

        if len(configs) <= 4:
            _annotate_bars(ax_bias, bars_b, fmt="{:.2e}", rotation=90)
            _annotate_bars(ax_var, bars_v, fmt="{:.2e}", rotation=90)

    fig.suptitle("Bias-Variance Components (Unstacked, Linear Scale)", fontsize=15, fontweight="bold")
    out = Path(output_dir) / "bias_variance_clarity.png"
    _save(fig, out)
    return out


def plot_bias_variance_relative(
    results_df: pd.DataFrame,
    output_dir: str = "project/results",
) -> Path:
    _apply_style()
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.36

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.8 * len(datasets), 5.2), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        bias_vals = np.array([float(subset.loc[cfg, "bias_squared"]) for cfg in configs], dtype=np.float64)
        var_vals = np.array([float(subset.loc[cfg, "variance"]) for cfg in configs], dtype=np.float64)

        baseline_bias = bias_vals[0] if bias_vals[0] != 0 else 1.0
        baseline_var = var_vals[0] if var_vals[0] != 0 else 1.0
        bias_rel = bias_vals / baseline_bias
        var_rel = var_vals / baseline_var

        bars1 = ax.bar(x - width / 2, bias_rel, width=width, label="Bias² / A", color="#4C78A8", edgecolor="#444444", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, var_rel, width=width, label="Variance / A", color="#54A24B", edgecolor="#444444", linewidth=0.5)

        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
        ax.set_ylabel("Relative to A (A = 1.0)")
        ax.set_title(f"Relative Bias-Variance: {dataset}")
        ax.legend()

        if len(configs) <= 4:
            _annotate_bars(ax, bars1, fmt="{:.2f}", rotation=90)
            _annotate_bars(ax, bars2, fmt="{:.2f}", rotation=90)

    fig.suptitle("Bias-Variance Relative to Single Tree Baseline", fontsize=15, fontweight="bold")
    out = Path(output_dir) / "bias_variance_relative_to_A.png"
    _save(fig, out)
    return out


def plot_training_time(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    _apply_style()
    datasets, configs = _prepare(results_df)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")
        means = [float(subset.loc[cfg, "mean_train_time"]) for cfg in configs]
        stds = [float(subset.loc[cfg, "std_train_time"]) for cfg in configs]
        bars = ax.bar(
            x + idx * width - 0.4 + width / 2,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=dataset,
            color=DATASET_COLORS.get(dataset, None),
            edgecolor="#444444",
            linewidth=0.5,
        )
        _annotate_bars(ax, bars, fmt="{:.0f}", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Mean Training Time (s, log scale)")
    ax.set_title("Runtime Cost by Configuration")
    ax.set_yscale("log")
    ax.legend()

    out = Path(output_dir) / "training_time.png"
    _save(fig, out)
    return out


def plot_seed_stability(seed_results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    _apply_style()
    datasets = list(dict.fromkeys(seed_results_df["dataset"].tolist()))
    configs = [cfg for cfg in CONFIG_ORDER if cfg in set(seed_results_df["config"].tolist())]

    fig, axes = plt.subplots(2, len(datasets), figsize=(6.5 * len(datasets), 8), squeeze=False)
    for col, dataset in enumerate(datasets):
        subset = seed_results_df[seed_results_df["dataset"] == dataset]

        r2_data = [subset[subset["config"] == cfg]["r2"].to_numpy() for cfg in configs]
        rmse_data = [subset[subset["config"] == cfg]["rmse"].to_numpy() for cfg in configs]

        axes[0, col].boxplot(r2_data, tick_labels=[CONFIG_LABELS.get(c, c) for c in configs], showfliers=False)
        axes[0, col].set_title(f"Seed Stability (R²): {dataset}")
        axes[0, col].set_ylabel("R²")

        axes[1, col].boxplot(rmse_data, tick_labels=[CONFIG_LABELS.get(c, c) for c in configs], showfliers=False)
        axes[1, col].set_title(f"Seed Stability (RMSE): {dataset}")
        axes[1, col].set_ylabel("RMSE")

    fig.suptitle("Across-Seed Stability by Configuration", fontsize=15, fontweight="bold")
    out = Path(output_dir) / "seed_stability.png"
    _save(fig, out)
    return out


def plot_efficiency_frontier(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    _apply_style()
    datasets = list(dict.fromkeys(results_df["dataset"].tolist()))

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.2), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        subset = results_df[results_df["dataset"] == dataset]

        for _, row in subset.iterrows():
            cfg = str(row["config"])
            ax.scatter(float(row["mean_train_time"]), float(row["mean_r2"]), s=110)
            ax.annotate(CONFIG_LABELS.get(cfg, cfg).replace("\n", " "), (float(row["mean_train_time"]), float(row["mean_r2"])), textcoords="offset points", xytext=(6, 5), fontsize=9)

        ax.set_xscale("log")
        ax.set_xlabel("Mean train time (s, log)")
        ax.set_ylabel("Mean R²")
        ax.set_title(f"Accuracy vs Runtime: {dataset}")

    fig.suptitle("Efficiency Frontier (Higher R², Lower Time)", fontsize=15, fontweight="bold")
    out = Path(output_dir) / "efficiency_frontier.png"
    _save(fig, out)
    return out


def plot_normalized_summary(results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    _apply_style()
    datasets, configs = _prepare(results_df)
    n_datasets = len(datasets)
    x = np.arange(len(configs))
    width = 0.36

    fig, axes = plt.subplots(2, n_datasets, figsize=(7 * n_datasets, 9), squeeze=False)

    for idx, dataset in enumerate(datasets):
        subset = results_df[results_df["dataset"] == dataset].set_index("config")

        r2 = np.array([float(subset.loc[cfg, "mean_r2"]) for cfg in configs], dtype=np.float64)
        rmse = np.array([float(subset.loc[cfg, "mean_rmse"]) for cfg in configs], dtype=np.float64)
        t = np.array([float(subset.loc[cfg, "mean_train_time"]) for cfg in configs], dtype=np.float64)

        # Quality scores (higher is better): relative to best quality.
        r2_norm = r2 / max(1e-12, np.max(r2))
        rmse_good = np.min(rmse) / np.maximum(rmse, 1e-12)

        # Runtime cost (lower is better): relative to fastest, where 1.0 is fastest.
        time_cost = t / max(1e-12, np.min(t))

        ax_q = axes[0, idx]
        b1 = ax_q.bar(x - width / 2, r2_norm, width=width, label="R² / best", color="#4C78A8")
        b2 = ax_q.bar(x + width / 2, rmse_good, width=width, label="(best RMSE) / RMSE", color="#54A24B")

        if len(configs) <= 4:
            _annotate_bars(ax_q, b1, fmt="{:.2f}", rotation=90)
            _annotate_bars(ax_q, b2, fmt="{:.2f}", rotation=90)

        ax_q.set_xticks(x)
        ax_q.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
        ax_q.set_ylim(0.0, 1.08)
        ax_q.set_ylabel("Quality score [0,1] (higher better)")
        ax_q.set_title(f"Predictive Quality (relative): {dataset}")
        ax_q.legend(loc="lower left", fontsize=9)

        ax_t = axes[1, idx]
        b3 = ax_t.bar(x, time_cost, width=0.55, label="Train time / fastest", color="#F58518")

        if len(configs) <= 4:
            _annotate_bars(ax_t, b3, fmt="{:.2f}x", rotation=90)

        ax_t.set_xticks(x)
        ax_t.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
        ax_t.set_ylabel("Relative runtime cost (1.0 = fastest)")
        ax_t.set_title(f"Training Time Cost (relative): {dataset}")
        ax_t.legend(loc="upper left", fontsize=9)
        ax_t.set_yscale("log")
        ax_t.grid(True, which="both", axis="y", alpha=0.25)

    fig.suptitle("Interpretability View: Quality vs Runtime Cost", fontsize=15, fontweight="bold")
    out = Path(output_dir) / "normalized_summary.png"
    _save(fig, out)
    return out


def plot_seedwise_delta_vs_bagging(seed_results_df: pd.DataFrame, output_dir: str = "project/results") -> Path:
    _apply_style()
    datasets = list(dict.fromkeys(seed_results_df["dataset"].tolist()))
    target_configs = ["A_SingleTree", "C_FeatureRandOnly", "D_FullRandomForest"]
    labels = [CONFIG_LABELS[c].replace("\n", " ") for c in target_configs]

    fig, axes = plt.subplots(2, len(datasets), figsize=(6.8 * len(datasets), 8), squeeze=False)

    for idx, dataset in enumerate(datasets):
        subset = seed_results_df[seed_results_df["dataset"] == dataset]
        pivot_r2 = subset.pivot(index="seed", columns="config", values="r2")
        pivot_rmse = subset.pivot(index="seed", columns="config", values="rmse")

        r2_deltas = [
            (pivot_r2[cfg] - pivot_r2["B_BaggingOnly"]).dropna().to_numpy()
            for cfg in target_configs
        ]
        rmse_deltas = [
            (pivot_rmse[cfg] - pivot_rmse["B_BaggingOnly"]).dropna().to_numpy()
            for cfg in target_configs
        ]

        ax1 = axes[0, idx]
        ax1.boxplot(r2_deltas, tick_labels=labels, showfliers=False)
        ax1.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
        ax1.set_title(f"ΔR² vs B (seed-wise): {dataset}")
        ax1.set_ylabel("R²(config) - R²(B)")

        ax2 = axes[1, idx]
        ax2.boxplot(rmse_deltas, tick_labels=labels, showfliers=False)
        ax2.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
        ax2.set_title(f"ΔRMSE vs B (seed-wise): {dataset}")
        ax2.set_ylabel("RMSE(config) - RMSE(B)")

    fig.suptitle("Seed-wise Differences Relative to Bagging Baseline", fontsize=15, fontweight="bold")
    out = Path(output_dir) / "seedwise_delta_vs_bagging.png"
    _save(fig, out)
    return out


def generate_all_plots(
    results_df: pd.DataFrame | None = None,
    seed_results_df: Optional[pd.DataFrame] = None,
    results_csv_path: str = "project/results/ablation_results.csv",
    seed_results_csv_path: str | None = None,
    output_dir: str = "project/results",
) -> list[Path]:
    if results_df is None:
        results_df = pd.read_csv(results_csv_path)

    if seed_results_df is None and seed_results_csv_path is not None:
        seed_path = Path(seed_results_csv_path)
        if seed_path.exists():
            seed_results_df = pd.read_csv(seed_path)

    outputs = [
        plot_bar_r2(results_df, output_dir=output_dir),
        plot_bar_rmse(results_df, output_dir=output_dir),
        plot_bias_variance_decomposition(results_df, output_dir=output_dir),
        plot_bias_variance_clarity(results_df, output_dir=output_dir),
        plot_bias_variance_relative(results_df, output_dir=output_dir),
        plot_training_time(results_df, output_dir=output_dir),
        plot_efficiency_frontier(results_df, output_dir=output_dir),
        plot_normalized_summary(results_df, output_dir=output_dir),
    ]

    if seed_results_df is not None and not seed_results_df.empty:
        outputs.append(plot_seed_stability(seed_results_df, output_dir=output_dir))
        outputs.append(plot_seedwise_delta_vs_bagging(seed_results_df, output_dir=output_dir))

    return outputs
