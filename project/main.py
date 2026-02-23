from __future__ import annotations

import argparse
import pandas as pd

from project.experiments.run_ablation import FAST_CONFIGS, FAST_SEEDS, run_ablation
from project.plots.plot_results import generate_all_plots


def _format_summary_table(results_df: pd.DataFrame) -> str:
    lines = []
    header = (
        f"{'dataset':<18} {'config':<22} {'mean_r2±std':<18} "
        f"{'mean_rmse±std':<20} {'bias²':<12} {'variance':<12} {'time(s)':<14}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['dataset']:<18} "
            f"{row['config']:<22} "
            f"{row['mean_r2']:.4f}±{row['std_r2']:.4f}   "
            f"{row['mean_rmse']:.4f}±{row['std_rmse']:.4f}   "
            f"{row['bias_squared']:.6f} "
            f"{row['variance']:.6f} "
            f"{row['mean_train_time']:.4f}±{row['std_train_time']:.4f}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random forest ablation study.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a quick mode with fewer seeds and smaller RF config.",
    )
    args = parser.parse_args()

    if args.fast:
        print("[main] FAST mode enabled")
        results_df = run_ablation(
            results_csv_path="project/results/ablation_results_fast.csv",
            configs=FAST_CONFIGS,
            seeds=FAST_SEEDS,
            n_bias_bootstrap=8,
            verbose=True,
            checkpoint_every_config=True,
            max_samples_per_dataset=10000,
        )
    else:
        print("[main] FULL mode enabled")
        results_df = run_ablation(
            results_csv_path="project/results/ablation_results.csv",
            verbose=True,
            checkpoint_every_config=True,
        )

    print(_format_summary_table(results_df))

    generated_paths = generate_all_plots(results_df=results_df, output_dir="project/results")
    for path in generated_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
