from __future__ import annotations

import argparse
import pandas as pd

from project.experiments.profiles import ABLATION_PROFILES, get_profile
from project.experiments.run_ablation import run_ablation
from project.experiments.run_calibration import run_calibration
from project.plots.plot_results import generate_all_plots


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


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
        "--profile",
        type=str,
        default="fast",
        choices=list(ABLATION_PROFILES.keys()),
        help="Ablation profile to run.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run runtime calibration matrix and exit.",
    )
    parser.add_argument(
        "--cal-sizes",
        type=str,
        default=None,
        help="Calibration sample sizes, comma-separated (e.g., 20000,40000,80000).",
    )
    parser.add_argument(
        "--cal-estimators",
        type=str,
        default=None,
        help="Calibration n_estimators grid, comma-separated (e.g., 25,50,100).",
    )
    parser.add_argument(
        "--cal-seeds",
        type=str,
        default=None,
        help="Calibration seeds, comma-separated (e.g., 42,123).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for ablation seed and bootstrap jobs.",
    )
    args = parser.parse_args()

    if args.calibrate:
        print("[main] CALIBRATION mode enabled")
        run_calibration(
            output_csv_path="project/results/runtime_calibration.csv",
            sample_sizes=_parse_int_list(args.cal_sizes),
            estimators_grid=_parse_int_list(args.cal_estimators),
            seeds=_parse_int_list(args.cal_seeds),
            n_jobs=max(1, int(args.n_jobs)),
            verbose=True,
        )
        return

    profile = get_profile(args.profile)
    print(f"[main] ABLATION profile='{profile.name}'")
    results_df = run_ablation(
        results_csv_path=f"project/results/ablation_results_{profile.name}.csv",
        seed_results_csv_path=f"project/results/ablation_seed_results_{profile.name}.csv",
        configs=profile.configs,
        seeds=profile.seeds,
        n_bias_bootstrap=profile.n_bias_bootstrap,
        verbose=True,
        checkpoint_every_config=True,
        max_samples_per_dataset=profile.max_samples_per_dataset,
        n_jobs=max(1, int(args.n_jobs)),
    )

    print(_format_summary_table(results_df))

    generated_paths = generate_all_plots(results_df=results_df, output_dir="project/results")
    for path in generated_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
