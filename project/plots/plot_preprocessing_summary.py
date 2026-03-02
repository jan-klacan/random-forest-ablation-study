from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _resolve_csv(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    alt = Path("project") / path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find dataset at: {path_str}")


def _prepare_used_cars(df: pd.DataFrame, target: str = "price", missing_threshold: float = 0.40) -> dict[str, object]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in used cars CSV.")

    raw_rows, raw_cols = df.shape

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    keep_columns = [target] + [
        col for col in df.columns if col != target and float(missing_ratio[col]) <= missing_threshold
    ]
    filtered_df = df.loc[:, keep_columns]
    filtered_rows, filtered_cols = filtered_df.shape

    cleaned_df = filtered_df.dropna(axis=0).copy()
    clean_rows, clean_cols = cleaned_df.shape

    feature_df = cleaned_df.drop(columns=[target])
    for col in feature_df.columns:
        if not is_numeric_dtype(feature_df[col]):
            feature_df[col] = pd.Categorical(feature_df[col]).codes

    X = feature_df.astype(np.float64).to_numpy()
    means_before = X.mean(axis=0)
    stds_before = X.std(axis=0)
    stds_safe = np.where(stds_before == 0.0, 1.0, stds_before)
    X_scaled = (X - means_before) / stds_safe

    means_after = X_scaled.mean(axis=0)
    stds_after = X_scaled.std(axis=0)

    return {
        "missing_ratio": missing_ratio,
        "missing_threshold": missing_threshold,
        "kept_column_count": len(keep_columns),
        "dropped_column_count": int(raw_cols - len(keep_columns)),
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "filtered_rows": filtered_rows,
        "filtered_cols": filtered_cols,
        "clean_rows": clean_rows,
        "clean_cols": clean_cols,
        "means_before": means_before,
        "stds_before": stds_before,
        "means_after": means_after,
        "stds_after": stds_after,
        "target_series": cleaned_df[target].astype(np.float64),
        "feature_count": feature_df.shape[1],
    }


def plot_missingness(missing_ratio: pd.Series, output_path: Path, threshold: float = 0.40) -> None:
    ordered = missing_ratio.sort_values(ascending=False)
    colors = ["tab:red" if value > threshold else "tab:blue" for value in ordered.values]

    n_features = len(ordered)
    fig_height = max(6, 0.35 * n_features)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    y_pos = np.arange(n_features)
    ax.barh(y_pos, ordered.values, color=colors)
    ax.axvline(threshold, linestyle="--", color="red", label=f"{int(threshold * 100)}% threshold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered.index)
    ax.invert_yaxis()
    ax.set_xlabel("Missing ratio")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, 1.02)
    ax.set_title("Used Cars: Missingness by Feature (All Columns)")

    dropped = int((ordered > threshold).sum())
    kept = int((ordered <= threshold).sum())
    ax.text(
        0.01,
        0.02,
        f"Kept: {kept} features   |   Dropped: {dropped} features",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_funnel(stats: dict[str, object], output_path: Path) -> None:
    stages = ["Raw rows", "After sparse\ncolumn filter", "After dropna"]
    row_values = [
        int(stats["raw_rows"]),
        int(stats["filtered_rows"]),
        int(stats["clean_rows"]),
    ]

    col_stages = ["Raw cols", "Kept cols"]
    col_values = [
        int(stats["raw_cols"]),
        int(stats["filtered_cols"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(stages, row_values)
    axes[0].set_title("Row Retention Through Cleaning")
    axes[0].set_ylabel("Number of rows")

    axes[1].bar(col_stages, col_values)
    axes[1].set_title("Column Retention (Used Cars)")
    axes[1].set_ylabel("Number of columns")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scaling_summary(stats: dict[str, object], output_path: Path) -> None:
    means_before = np.asarray(stats["means_before"])
    stds_before = np.asarray(stats["stds_before"])
    means_after = np.asarray(stats["means_after"])
    stds_after = np.asarray(stats["stds_after"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].boxplot([means_before, means_after], tick_labels=["Before", "After"], showfliers=False)
    axes[0].set_title("Feature Means Distribution")
    axes[0].set_ylabel("Mean value")
    axes[0].set_yscale("symlog", linthresh=1.0)

    axes[1].boxplot([stds_before, stds_after], tick_labels=["Before", "After"], showfliers=False)
    axes[1].set_title("Feature Std Distribution")
    axes[1].set_ylabel("Standard deviation")
    axes[1].set_yscale("log")

    fig.suptitle("Scaling Check: Before vs After Standardization")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_preprocessing_plots(
    used_cars_csv: str = "project/data/used_cars.csv",
    output_dir: str = "project/results",
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    used_cars_path = _resolve_csv(used_cars_csv)
    used_df = pd.read_csv(used_cars_path)

    stats = _prepare_used_cars(used_df)

    file_missing = output_path / "preprocess_usedcars_missingness.png"
    file_funnel = output_path / "preprocess_usedcars_funnel.png"
    file_scaling = output_path / "preprocess_usedcars_scaling.png"

    plot_missingness(stats["missing_ratio"], file_missing)
    plot_funnel(stats, file_funnel)
    plot_scaling_summary(stats, file_scaling)

    return [file_missing, file_funnel, file_scaling]


if __name__ == "__main__":
    outputs = generate_preprocessing_plots()
    for path in outputs:
        print(path)
