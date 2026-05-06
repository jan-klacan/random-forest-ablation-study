# A Random Forest Ablation Study

![Language](https://img.shields.io/badge/language-python-blue)
![Notebook](https://img.shields.io/badge/notebook-ipynb-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
This project runs an ablation study for a scratch-built random forest regressor. It compares:

- Single regression tree
- Bagging only
- Feature subsampling only
- Full random forest (bagging + feature subsampling)

Metrics include R^2, RMSE, bias^2, variance, and training time across datasets and seeds. Results are written to CSV and can be summarized with plots.

## Data inputs
The dataset loader expects CSVs at the following paths (either location is accepted):

- data/used_cars.csv or project/data/used_cars.csv
	- Target column candidates: price, selling_price, car_price
- data/california_housing.csv or project/data/california_housing.csv
	- Target column candidates: median_house_value, medhouseval

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run experiments
Run an ablation profile (generates CSV results and plots):

```bash
python -m project.main --profile fast
```

Run runtime calibration:

```bash
python -m project.main --calibrate \
	--cal-sizes 20000,40000,80000 \
	--cal-estimators 25,50,100 \
	--cal-seeds 42,123 \
	--n-jobs 4
```

### Available profiles
Defined in project/experiments/profiles.py:

- fast
- pilot
- standard
- overnight
- overnight10h
- cd_test
- full

## Outputs
Generated files are written under project/results/ and include:

- ablation_results_<profile>.csv
- ablation_seed_results_<profile>.csv
- runtime_calibration.csv and runtime_calibration_summary.csv
- Plots such as bar_r2.png, bar_rmse.png, bias_variance_decomposition.png, and more

## Plotting utilities
- Ablation runs via project/main.py call generate_all_plots to produce summary figures.
- Preprocessing plots for the used cars dataset can be generated with:

```bash
python -m project.plots.plot_preprocessing_summary
```

## License

MIT