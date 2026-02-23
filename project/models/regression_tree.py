from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None


class RegressionTree:
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.var(y) == 0.0
        ):
            return TreeNode(value=float(np.mean(y)))

        n_features = X.shape[1]
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            max_features = int(min(max(self.max_features, 1), n_features))
            feature_indices = self.rng.choice(n_features, size=max_features, replace=False)

        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_loss = np.inf

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                n_total = len(y)
                left_var = np.var(y[left_mask])
                right_var = np.var(y[right_mask])
                loss = (n_left / n_total) * left_var + (n_right / n_total) * right_var

                if loss < best_loss:
                    best_loss = loss
                    best_feature = int(feature_index)
                    best_threshold = float(threshold)

        if best_feature is None or best_threshold is None:
            return TreeNode(value=float(np.mean(y)))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
            value=None,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("The tree has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_sample(self.root, x) for x in X], dtype=np.float64)

    def _predict_sample(self, node: TreeNode, x: np.ndarray) -> float:
        if node.value is not None:
            return float(node.value)

        if node.feature_index is None or node.threshold is None:
            raise ValueError("Invalid tree node encountered during prediction.")

        if x[node.feature_index] <= node.threshold:
            if node.left is None:
                raise ValueError("Invalid left branch in tree.")
            return self._predict_sample(node.left, x)

        if node.right is None:
            raise ValueError("Invalid right branch in tree.")
        return self._predict_sample(node.right, x)
