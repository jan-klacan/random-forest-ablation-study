from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .regression_tree import RegressionTree


class RandomForestRegressorScratch:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: Union[str, int] = "sqrt",
        use_bootstrap: bool = True,
        use_feature_subsampling: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.use_bootstrap = use_bootstrap
        self.use_feature_subsampling = use_feature_subsampling
        self.random_state = random_state

        self.trees: list[RegressionTree] = []
        self.rng = np.random.RandomState(random_state)

    def _resolve_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.floor(np.sqrt(n_features))))
            if self.max_features == "all":
                return n_features
            raise ValueError("max_features string must be 'sqrt' or 'all'.")

        max_features_int = int(self.max_features)
        return int(min(max(max_features_int, 1), n_features))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        max_features_int = self._resolve_max_features(n_features)

        self.trees = []
        self.rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            tree_seed = int(self.rng.randint(0, 1_000_000))
            bootstrap_rng = np.random.RandomState(tree_seed)

            if self.use_bootstrap:
                indices = bootstrap_rng.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)

            X_sample = X[indices]
            y_sample = y[indices]

            max_feat_for_tree: Optional[int]
            if self.use_feature_subsampling:
                max_feat_for_tree = max_features_int
            else:
                max_feat_for_tree = None

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feat_for_tree,
                random_state=tree_seed,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("The model has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        all_predictions = np.array([tree.predict(X) for tree in self.trees], dtype=np.float64)
        return np.mean(all_predictions, axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)
