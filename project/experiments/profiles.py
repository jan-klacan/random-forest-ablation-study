from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationProfile:
    name: str
    configs: dict[str, dict[str, Any]]
    seeds: list[int]
    n_bias_bootstrap: int
    max_samples_per_dataset: int | None


def _build_configs(n_estimators: int) -> dict[str, dict[str, Any]]:
    base_tree_params: dict[str, Any] = {
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    }
    return {
        "A_SingleTree": {
            "n_estimators": 1,
            "use_bootstrap": False,
            "use_feature_subsampling": False,
            **base_tree_params,
        },
        "B_BaggingOnly": {
            "n_estimators": n_estimators,
            "use_bootstrap": True,
            "use_feature_subsampling": False,
            **base_tree_params,
        },
        "C_FeatureRandOnly": {
            "n_estimators": n_estimators,
            "use_bootstrap": False,
            "use_feature_subsampling": True,
            **base_tree_params,
        },
        "D_FullRandomForest": {
            "n_estimators": n_estimators,
            "use_bootstrap": True,
            "use_feature_subsampling": True,
            **base_tree_params,
        },
    }


def _build_runtime_balanced_configs(n_estimators: int) -> dict[str, dict[str, Any]]:
    base_tree_params: dict[str, Any] = {
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
    }
    return {
        "A_SingleTree": {
            "n_estimators": 1,
            "use_bootstrap": False,
            "use_feature_subsampling": False,
            **base_tree_params,
        },
        "B_BaggingOnly": {
            "n_estimators": n_estimators,
            "use_bootstrap": True,
            "use_feature_subsampling": False,
            **base_tree_params,
        },
        "C_FeatureRandOnly": {
            "n_estimators": n_estimators,
            "use_bootstrap": False,
            "use_feature_subsampling": True,
            **base_tree_params,
        },
        "D_FullRandomForest": {
            "n_estimators": n_estimators,
            "use_bootstrap": True,
            "use_feature_subsampling": True,
            **base_tree_params,
        },
    }


ABLATION_PROFILES: dict[str, AblationProfile] = {
    "fast": AblationProfile(
        name="fast",
        configs={
            "A_SingleTree": {
                "n_estimators": 1,
                "use_bootstrap": False,
                "use_feature_subsampling": False,
            },
            "D_FullRandomForest": {
                "n_estimators": 25,
                "use_bootstrap": True,
                "use_feature_subsampling": True,
            },
        },
        seeds=[42, 123, 7],
        n_bias_bootstrap=8,
        max_samples_per_dataset=10000,
    ),
    "pilot": AblationProfile(
        name="pilot",
        configs=_build_configs(n_estimators=75),
        seeds=[42, 123, 7, 999, 2024],
        n_bias_bootstrap=15,
        max_samples_per_dataset=50000,
    ),
    "standard": AblationProfile(
        name="standard",
        configs=_build_configs(n_estimators=100),
        seeds=[42, 123, 7, 999, 2024, 31, 55, 88],
        n_bias_bootstrap=20,
        max_samples_per_dataset=90000,
    ),
    "overnight": AblationProfile(
        name="overnight",
        configs=_build_configs(n_estimators=100),
        seeds=[42, 123, 7, 999, 2024, 31, 55, 88, 200, 314],
        n_bias_bootstrap=30,
        max_samples_per_dataset=90000,
    ),
    "overnight10h": AblationProfile(
        name="overnight10h",
        configs=_build_runtime_balanced_configs(n_estimators=50),
        seeds=[42, 123, 7, 999, 2024, 31],
        n_bias_bootstrap=12,
        max_samples_per_dataset=60000,
    ),
    "cd_test": AblationProfile(
        name="cd_test",
        configs={
            key: value
            for key, value in _build_runtime_balanced_configs(n_estimators=50).items()
            if key in {"C_FeatureRandOnly", "D_FullRandomForest"}
        },
        seeds=[42, 123, 7],
        n_bias_bootstrap=6,
        max_samples_per_dataset=40000,
    ),    
    "full": AblationProfile(
        name="full",
        configs=_build_configs(n_estimators=150),
        seeds=[42, 123, 7, 999, 2024, 31, 55, 88, 200, 314],
        n_bias_bootstrap=30,
        max_samples_per_dataset=None,
    ),
}


def get_profile(name: str) -> AblationProfile:
    key = name.strip().lower()
    if key not in ABLATION_PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(ABLATION_PROFILES.keys())}")
    return ABLATION_PROFILES[key]
