"""
hyperparameter_tuning.py

Fast Hyperparameter Tuning Module
Supports:
- Random Search (fast, RL-friendly)
- Model-specific parameter spaces
- Classification & Regression
- Sparse + Dense safe

Avoids slow GridSearch → optimized for hackathons
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score
)


# =========================
# CONFIG
# =========================

@dataclass
class TuningConfig:
    n_trials: int = 10
    task: str = "auto"
    metric: str = "auto"
    test_size: float = 0.2
    random_state: int = 42


# =========================
# MAIN CLASS
# =========================

class HyperparameterTuner:
    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()
        self.best_model = None
        self.best_score = -np.inf
        self.report = {}

    # =========================
    # ENTRY
    # =========================
    def tune(self, model, X, y):
        if len(X)!=len(y):
            minlen = min(len(X),len(y))
            X = X[:minlen]
            y = y[:minlen]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        self.task = self._detect_task(y)
        param_space = self._get_param_space(model)

        for i in range(self.config.n_trials):
            params = self._sample_params(param_space)

            try:
                model.set_params(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = self._evaluate(y_test, preds)

                self._log(f"trial_{i}", {"params": params, "score": score})

                if score > self.best_score:
                    self.best_score = score
                    self.best_model = self._clone_model(model)

            except Exception as e:
                self._log(f"trial_{i}_error", str(e))

        self._log("best_score", self.best_score)
        self._log("best_model", type(self.best_model).__name__)

        return self.best_model

    # =========================
    # PARAM SPACES
    # =========================
    def _get_param_space(self, model) -> Dict[str, Any]:
        name = type(model).__name__

        return {
            "LogisticRegression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"]
            },
            "RandomForestClassifier": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]
            },
            "RandomForestRegressor": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]
            },
            "SGDClassifier": {
                "alpha": [1e-4, 1e-3, 1e-2]
            },
            "SGDRegressor": {
                "alpha": [1e-4, 1e-3, 1e-2]
            },
            "GradientBoostingClassifier": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1]
            },
            "GradientBoostingRegressor": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1]
            }
        }.get(name, {})

    def _sample_params(self, param_space):
        return {k: random.choice(v) for k, v in param_space.items()}

    # =========================
    # METRIC
    # =========================
    def _evaluate(self, y_true, y_pred):
        if self.task == "classification":
            if self.config.metric == "f1":
                return f1_score(y_true, y_pred, average="weighted")
            return accuracy_score(y_true, y_pred)

        else:
            if self.config.metric == "r2":
                return r2_score(y_true, y_pred)
            return -np.sqrt(mean_squared_error(y_true, y_pred))

    def _detect_task(self, y):
        return "classification" if len(np.unique(y)) < 20 else "regression"

    # =========================
    # UTIL
    # =========================
    def _clone_model(self, model):
        import copy
        return copy.deepcopy(model)

    # =========================
    # LOGGER
    # =========================
    def _log(self, key, value):
        self.report[key] = value

    def get_report(self):
        return self.report