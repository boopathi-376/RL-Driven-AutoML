"""
encoding.py

Universal Encoding Module
Supports:
- Categorical encoding (one-hot, ordinal, target)
- Text vectorization (TF-IDF, hashing)
- Mixed DataFrames
- Raw text / list of text

RL-compatible, leakage-aware design
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer


# =========================
# CONFIG
# =========================

@dataclass
class EncodingConfig:
    # categorical
    categorical_method: str = "onehot"   # onehot / ordinal / target

    # text
    text_method: str = "tfidf"           # tfidf / hashing
    max_features: int = 5000
    ngram_range: tuple = (1, 2)

    # behavior
    handle_unknown: str = "ignore"
    drop_first: bool = False


# =========================
# MAIN ENCODER
# =========================

class Encoder:
    def __init__(self, config: Optional[EncodingConfig] = None):
        self.config = config or EncodingConfig()

        # models
        self.cat_encoder = None
        self.text_vectorizer = None

        # metadata
        self.fitted = False
        self.report = {}

    # =========================
    # UNIVERSAL ENTRY
    # =========================
    def fit(self, data: Any, target: Optional[pd.Series] = None):
        if isinstance(data, pd.DataFrame):
            self._fit_dataframe(data, target)

        elif isinstance(data, list):
            self._fit_text(data)

        elif isinstance(data, str):
            self._fit_text([data])

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.fitted = True
        return self

    def transform(self, data: Any) -> Any:
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted first")

        if isinstance(data, pd.DataFrame):
            return self._transform_dataframe(data)

        elif isinstance(data, list):
            return self._transform_text(data)

        elif isinstance(data, str):
            return self._transform_text([data])

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def fit_transform(self, data: Any, target: Optional[pd.Series] = None):
        return self.fit(data, target).transform(data)

    # =========================
    # DATAFRAME HANDLING
    # =========================
    def _fit_dataframe(self, df: pd.DataFrame, target=None):
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns
        self.num_cols = df.select_dtypes(include=np.number).columns

        # ---- CATEGORICAL ----
        if len(self.cat_cols) > 0:
            if self.config.categorical_method == "onehot":
                self.cat_encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown=self.config.handle_unknown,
                    drop="first" if self.config.drop_first else None
                )
                self.cat_encoder.fit(df[self.cat_cols])

            elif self.config.categorical_method == "ordinal":
                self.cat_encoder = OrdinalEncoder()
                self.cat_encoder.fit(df[self.cat_cols])

            elif self.config.categorical_method == "target":
                if target is None:
                    raise ValueError("Target required for target encoding")
                self._fit_target_encoding(df, target)

        # ---- TEXT ----
        text_cols = self._detect_text_columns(df)

        if len(text_cols) > 0:
            self.text_cols = text_cols
            combined_text = df[text_cols].fillna("").agg(" ".join, axis=1)
            self._fit_text(combined_text.tolist())

        self._log("categorical_cols", list(self.cat_cols))
        self._log("text_cols", list(getattr(self, "text_cols", [])))

    def _transform_dataframe(self, df):
        result_parts = []

        # numeric
        if len(self.num_cols) > 0:
            result_parts.append(df[self.num_cols].reset_index(drop=True))

        # categorical
        if self.cat_encoder is not None:
            cat_encoded = self.cat_encoder.transform(df[self.cat_cols])
            cat_df = pd.DataFrame(cat_encoded)
            result_parts.append(cat_df)

        # target encoding
        if self.config.categorical_method == "target":
            result_parts.append(self._transform_target_encoding(df))

        # text
        if hasattr(self, "text_cols"):
            combined_text = df[self.text_cols].fillna("").agg(" ".join, axis=1)
            text_features = self._transform_text(combined_text.tolist())
            text_df = pd.DataFrame(text_features.toarray())
            result_parts.append(text_df)

        final_df = pd.concat(result_parts, axis=1)

        self._log("output_shape", final_df.shape)

        return final_df

    # =========================
    # TEXT HANDLING
    # =========================
    def _fit_text(self, texts: List[str]):
        if self.config.text_method == "tfidf":
            self.text_vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range
            )

        elif self.config.text_method == "hashing":
            self.text_vectorizer = HashingVectorizer(
                n_features=self.config.max_features
            )

        self.text_vectorizer.fit(texts)

    def _transform_text(self, texts: List[str]):
        return self.text_vectorizer.transform(texts)

    # =========================
    # TARGET ENCODING
    # =========================
    def _fit_target_encoding(self, df, target):
        self.target_maps = {}
        for col in self.cat_cols:
            means = target.groupby(df[col]).mean()
            self.target_maps[col] = means

    def _transform_target_encoding(self, df):
        result = pd.DataFrame()
        for col in self.cat_cols:
            result[col] = df[col].map(self.target_maps[col]).fillna(0)
        return result

    # =========================
    # TEXT COLUMN DETECTION
    # =========================
    def _detect_text_columns(self, df):
        text_cols = []
        for col in df.select_dtypes(include=["object"]).columns:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 20:  # heuristic
                text_cols.append(col)
        return text_cols

    # =========================
    # LOGGER
    # =========================
    def _log(self, key, value):
        self.report[key] = value

    def get_report(self):
        return self.report