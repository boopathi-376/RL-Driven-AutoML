"""
ensemble.py

Advanced Ensemble Module
Supports:
- Voting (classification)
- Averaging (regression)
- Stacking (meta model)
- Weighted ensembles

Works with any sklearn-compatible models
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


# =========================
# CONFIG
# =========================

@dataclass
class EnsembleConfig:
    method: str = "voting"   # voting / stacking / weighted / none
    task: str = "auto"
    weights: Optional[List[float]] = None


# =========================
# MAIN CLASS
# =========================

class EnsembleBuilder:
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.ensemble_model = None
        self.task = None
        self.report = {}

    # =========================
    # ENTRY
    # =========================
    def build(self, models: List, X, y):
        if len(X)!=len(y):
            minlen = min(len(X),len(y))
            X = X[:minlen]
            y = y[:minlen]
        self.task = self._detect_task(y)

        if self.config.method == "none":
            self.ensemble_model = models[0]
            return self.ensemble_model

        if self.config.method == "voting":
            self.ensemble_model = self._build_voting(models)

        elif self.config.method == "stacking":
            self.ensemble_model = self._build_stacking(models)

        elif self.config.method == "weighted":
            self.ensemble_model = self._build_weighted(models)

        else:
            raise ValueError("Unknown ensemble method")

        self.ensemble_model.fit(X, y)

        self._log("method", self.config.method)
        self._log("n_models", len(models))

        return self.ensemble_model

    def predict(self, X):
        return self.ensemble_model.predict(X)

    # =========================
    # METHODS
    # =========================
    def _build_voting(self, models):
        named_models = [(f"m{i}", m) for i, m in enumerate(models)]

        if self.task == "classification":
            return VotingClassifier(
                estimators=named_models,
                voting="soft"
            )
        else:
            return VotingRegressor(
                estimators=named_models
            )

    def _build_stacking(self, models):
        named_models = [(f"m{i}", m) for i, m in enumerate(models)]

        if self.task == "classification":
            return StackingClassifier(
                estimators=named_models,
                final_estimator=LogisticRegression()
            )
        else:
            return StackingRegressor(
                estimators=named_models,
                final_estimator=LinearRegression()
            )

    def _build_weighted(self, models):
        if self.config.weights is None:
            self.config.weights = [1] * len(models)

        class WeightedModel:
            def __init__(self, models, weights, task):
                self.models = models
                self.weights = weights
                self.task = task

            def fit(self, X, y):
                for m in self.models:
                    m.fit(X, y)

            def predict(self, X):
                preds = np.array([m.predict(X) for m in self.models])

                if self.task == "classification":
                    return np.round(np.average(preds, axis=0, weights=self.weights))
                else:
                    return np.average(preds, axis=0, weights=self.weights)

        return WeightedModel(models, self.config.weights, self.task)

    # =========================
    # UTIL
    # =========================
    def _detect_task(self, y):
        return "classification" if len(np.unique(y)) < 20 else "regression"

    def _log(self, key, value):
        self.report[key] = value

    def get_report(self):
        return self.report