"""
model_selection.py

Production-Grade Smart Model Selector
- Rule-based (expert logic)
- Supports text + tabular + mixed data
- Safe, explainable, extensible
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Dict, Any
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


# =========================
# CONFIG
# =========================

@dataclass
class ModelSelectionConfig:
    test_size: float = 0.2
    random_state: int = 42
    metric: str = "auto"   # accuracy / f1 / r2 / rmse


# =========================
# MAIN CLASS
# =========================

class SmartModelSelector:
    def __init__(self, config: Optional[ModelSelectionConfig] = None):
        self.config = config or ModelSelectionConfig()

        self.model = None
        self.model_name = None
        self.task = None

        self.score = None
        self.report: Dict[str, Any] = {}

    # =========================
    # MAIN ENTRY
    # =========================
    def fit(self, X, y):
        X = self._validate_X(X)
        y = self._validate_y(y)

        # If X is a DataFrame with text (object) columns, vectorize with TF-IDF.
        # Raw strings cannot be passed to sklearn estimators directly.
        X = self._vectorize_text_if_needed(X)

        self.task = self._detect_task(y)

        meta = self._analyze_data(X)
        self.model, self.model_name = self._select_model(meta)
        self._train_and_evaluate(X, y)

        return self

    def _vectorize_text_if_needed(self, X):
        """If X contains any string (object) column, join all text columns and
        return a TF-IDF sparse matrix so sklearn models can handle it."""
        if not isinstance(X, pd.DataFrame):
            return X
        text_cols = X.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            return X
        # Combine all text columns into one string per row
        combined = X[text_cols].fillna("").apply(lambda row: " ".join(row.values), axis=1)
        self._tfidf = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
        X_tfidf = self._tfidf.fit_transform(combined)
        self._log("text_vectorized", True)
        self._log("tfidf_features", X_tfidf.shape[1])
        return X_tfidf

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model

    def get_report(self):
        return self.report

    # =========================
    # DATA ANALYSIS
    # =========================
    def _analyze_data(self, X):

        is_sparse = sparse.issparse(X)

        if is_sparse:
            n_samples, n_features = X.shape
        else:
            X_np = np.array(X)
            n_samples, n_features = X_np.shape

        density = None
        if is_sparse:
            density = X.nnz / (n_samples * n_features)

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "is_sparse": is_sparse,
            "density": density
        }

    # =========================
    # MODEL DECISION LOGIC
    # =========================
    def _select_model(self, meta):

        n_samples = meta["n_samples"]
        n_features = meta["n_features"]
        is_sparse = meta["is_sparse"]

        # TEXT / SPARSE DATA
        if is_sparse:
            if self.task == "classification":
                model = SGDClassifier(loss="log_loss", max_iter=1000)
                name = "SGDClassifier (sparse/text optimized)"
            else:
                model = SGDRegressor(max_iter=1000)
                name = "SGDRegressor (sparse)"

        # HIGH DIMENSIONAL
        elif n_features > 500:
            if self.task == "classification":
                model = SGDClassifier(max_iter=1000)
                name = "SGDClassifier (high-dim)"
            else:
                model = SGDRegressor(max_iter=1000)
                name = "SGDRegressor"

        # SMALL DATA
        elif n_samples < 1000:
            if self.task == "classification":
                model = LogisticRegression(max_iter=1000)
                name = "LogisticRegression (small data)"
            else:
                model = LinearRegression()
                name = "LinearRegression"

        # MEDIUM DATA
        elif n_samples < 10000:
            if self.task == "classification":
                model = DecisionTreeClassifier(max_depth=10)
                name = "DecisionTree (medium data)"
            else:
                model = RandomForestRegressor(n_estimators=100)
                name = "RandomForestRegressor"

        # LARGE TABULAR
        else:
            if self.task == "classification":
                model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
                name = "RandomForest (large data)"
            else:
                model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
                name = "RandomForestRegressor (large data)"

        self._log("decision_meta", meta)
        self._log("selected_model", name)

        return model, name

    # =========================
    # TRAIN + EVALUATE
    # =========================
    def _train_and_evaluate(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        try:
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            self.score = self._evaluate(y_test, preds)
            self._log("score", self.score)

        except Exception as e:
            self._log("error", str(e))
            raise RuntimeError(f"Model training failed: {e}")

    # =========================
    # METRIC
    # =========================
    def _evaluate(self, y_true, y_pred):

        if self.task == "text_processing":
            return 0.5  # neutral score (no supervision)

        if self.task == "classification":
            if self.config.metric == "f1":
                return f1_score(y_true, y_pred, average="weighted")
            return accuracy_score(y_true, y_pred)

        else:
            if self.config.metric == "r2":
                return r2_score(y_true, y_pred)

            return np.sqrt(mean_squared_error(y_true, y_pred))
        

    # =========================
    # TASK DETECTION
    # =========================
    def _detect_task(self, y):
        unique_vals = np.unique(y)
        return "classification" if len(unique_vals) < 20 else "regression"

    # =========================
    # VALIDATION
    # =========================
    def _validate_X(self, X):
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return X
        if sparse.issparse(X):
            return X
        raise TypeError(f"Unsupported X type: {type(X)}")

    def _validate_y(self, y):
        if isinstance(y, (pd.Series, list)):
            return np.array(y)
        if isinstance(y, np.ndarray):
            return y
        raise TypeError(f"Unsupported y type: {type(y)}")

    # =========================
    # LOGGER
    # =========================
    def _log(self, key, value):
        self.report[key] = value