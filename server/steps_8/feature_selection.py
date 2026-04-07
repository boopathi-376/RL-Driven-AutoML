"""
feature_selection.py

Advanced Feature Selection Module
Supports:
- Variance thresholding
- Correlation pruning
- Model-based selection
- Top-K feature selection

Works with dense + sparse data
RL-compatible design
"""

import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import sparse


# =========================
# CONFIG
# =========================

@dataclass
class FeatureSelectionConfig:
    method: str = "variance"   # variance / correlation / model / kbest / none
    threshold: float = 0.01    # for variance / correlation
    k: int = 50                # for kbest
    task: str = "auto"         # classification / regression / auto


# =========================
# MAIN CLASS
# =========================

class FeatureSelector:
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        self.selector = None
        self.selected_indices = None
        self.fitted = False
        self.report = {}

    # =========================
    # ENTRY
    # =========================
    def fit(self, X: Any, y: Optional[np.ndarray] = None):
        X = self._to_numpy(X)

        # auto detect task
        if self.config.task == "auto" and y is not None:
            self.config.task = "classification" if len(np.unique(y)) < 20 else "regression"

        if self.config.method == "variance":
            self.selector = VarianceThreshold(threshold=self.config.threshold)
            self.selector.fit(X)
            self.selected_indices = self.selector.get_support(indices=True)

        elif self.config.method == "correlation":
            self.selected_indices = self._correlation_selection(X)

        elif self.config.method == "kbest":
            score_func = f_classif if self.config.task == "classification" else f_regression
            self.selector = SelectKBest(score_func=score_func, k=min(self.config.k, X.shape[1]))
            self.selector.fit(X, y)
            self.selected_indices = self.selector.get_support(indices=True)

        elif self.config.method == "model":
            self.selected_indices = self._model_based_selection(X, y)

        elif self.config.method == "none":
            self.selected_indices = np.arange(X.shape[1])

        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        self._log("selected_features", len(self.selected_indices))
        self._log("original_features", X.shape[1])

        self.fitted = True
        return self

    def transform(self, X: Any):
        if not self.fitted:
            raise RuntimeError("FeatureSelector must be fitted first")

        X_np = self._to_numpy(X)

        if sparse.issparse(X_np):
            return X_np[:, self.selected_indices]

        return X_np[:, self.selected_indices]

    def fit_transform(self, X: Any, y: Optional[np.ndarray] = None):
        return self.fit(X, y).transform(X)

    # =========================
    # METHODS
    # =========================
    def _correlation_selection(self, X):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.config.threshold)
        ]

        keep = [i for i in range(X.shape[1]) if i not in df.columns.get_indexer(to_drop)]
        return np.array(keep)

    def _model_based_selection(self, X, y):
        if y is None:
            raise ValueError("Target required for model-based selection")

        if self.config.task == "classification":
            model = RandomForestClassifier(n_estimators=100)
        else:
            model = RandomForestRegressor(n_estimators=100)

        model.fit(X, y)
        importances = model.feature_importances_

        threshold = np.mean(importances)
        return np.where(importances >= threshold)[0]

    # =========================
    # UTIL
    # =========================
    def _to_numpy(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values

        if sparse.issparse(X):
            return X

        if isinstance(X, list):
            return np.array(X)

        if isinstance(X, np.ndarray):
            return X

        raise TypeError(f"Unsupported type: {type(X)}")

    # =========================
    # LOGGER
    # =========================
    def _log(self, key, value):
        self.report[key] = value

    def get_report(self):
        return self.report