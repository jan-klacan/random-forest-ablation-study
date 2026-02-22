import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Node:
    """A node in the regression tree."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    """Minimal regression tree supporting feature subsampling."""
    def __init__(self, min_samples_split=2, max_depth=10, max_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if n_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._get_best_split(X, y, n_features)
            if best_split:
                left_child = self._build_tree(best_split['X_left'], best_split['y_left'], depth + 1)
                right_child = self._build_tree(best_split['X_right'], best_split['y_right'], depth + 1)
                return Node(best_split['feature_index'], best_split['threshold'], left_child, right_child)
                
        # Create leaf node with mean value for regression
        leaf_value = np.mean(y)
        return Node(value=leaf_value)

    def _get_best_split(self, X, y, n_features):
        best_split = {}
        max_var_reduction = -float("inf")
        
        # Point 1: Feature subsampling at each split
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    var_reduction = self._variance_reduction(y, y_left, y_right)
                    if var_reduction > max_var_reduction:
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'X_left': X_left,
                            'y_left': y_left,
                            'X_right': X_right,
                            'y_right': y_right
                        }
                        max_var_reduction = var_reduction
                        
        return best_split if max_var_reduction > 0 else None

    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForestRegressor:
    """Minimal ensemble supporting configurable bootstrapping and feature subsampling."""
    def __init__(self, n_estimators=30, min_samples_split=2, max_depth=10, bootstrap=True, max_features=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            # Point 1: Configurable bootstrapping
            if self.bootstrap:
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y
                
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

if __name__ == "__main__":
    # --- Data Preparation ---
    X, y = make_regression(n_samples=500, n_features=10, noise=15.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_features = X_train.shape[1]
    feature_subset_size = max(1, n_features // 3)
    n_trees = 30 

    # --- Point 2: Ablation Study Variants ---
    models = {
        "A: Single Tree": RandomForestRegressor(
            n_estimators=1, bootstrap=False, max_features=n_features
        ),
        "B: Bagging Only": RandomForestRegressor(
            n_estimators=n_trees, bootstrap=True, max_features=n_features
        ),
        "C: Feature Randomness Only": RandomForestRegressor(
            n_estimators=n_trees, bootstrap=False, max_features=feature_subset_size
        ),
        "D: Full Random Forest": RandomForestRegressor(
            n_estimators=n_trees, bootstrap=True, max_features=feature_subset_size
        )
    }

    # --- Execution & Evaluation ---
    print("--Ablation Study Results (MSE)--")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name:30} | MSE: {mse:.2f}")