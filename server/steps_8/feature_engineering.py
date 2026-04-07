"""
feature_engineering.py

Advanced Feature Engineering Module
Supports:
- Numeric feature interactions
- Polynomial features
- Datetime decomposition
- Text-derived features
- Frequency encoding (lightweight categorical intelligence)

RL-compatible design
"""

import pandas as pd
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass

from sklearn.preprocessing import PolynomialFeatures


# =========================
# CONFIG
# =========================

@dataclass
class FeatureEngineeringConfig:
    polynomial_degree: int = 1
    interaction_only: bool = False

    create_interactions: bool = True
    create_datetime_features: bool = True
    create_text_features: bool = True
    frequency_encoding: bool = True


# =========================
# MAIN CLASS
# =========================

class FeatureEngineer:
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self.poly = None
        self.fitted = False
        self.report = {}

    # =========================
    # ENTRY
    # =========================
    def fit(self, df: pd.DataFrame):
        df = df.copy()
        self.num_cols = df.select_dtypes(include=np.number).columns
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns

        # Polynomial setup
        if self.config.polynomial_degree > 1:
            self.poly = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                interaction_only=self.config.interaction_only,
                include_bias=False
            )
            self.poly.fit(df[self.num_cols])

        # Frequency encoding map
        if self.config.frequency_encoding:
            self.freq_maps = {
                col: df[col].value_counts(normalize=True)
                for col in self.cat_cols
            }

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fitted first")

        df = df.copy()

        # Encoding can produce duplicate column names (e.g. two columns both named "0").
        # When that happens df[col_name] returns a DataFrame instead of a Series,
        # which breaks the interaction multiplication below.
        if df.columns.duplicated().any():
            cols = []
            seen = {}
            for c in df.columns:
                if c in seen:
                    seen[c] += 1
                    cols.append(f"{c}.{seen[c]}")
                else:
                    seen[c] = 0
                    cols.append(c)
            df.columns = cols

        features = []

        # =========================
        # NUMERIC BASE
        # =========================
        if len(self.num_cols) > 0:
            features.append(df[self.num_cols].reset_index(drop=True))

        # =========================
        # POLYNOMIAL FEATURES
        # =========================
        if self.poly is not None:
            poly_features = self.poly.transform(df[self.num_cols])
            poly_df = pd.DataFrame(poly_features)
            features.append(poly_df)

        # =========================
        # INTERACTIONS (manual)
        # =========================
        if self.config.create_interactions and len(self.num_cols) > 1:
            cols = list(self.num_cols)
            # Build all columns at once into a dict, then construct DataFrame in one shot.
            # Avoids pandas PerformanceWarning about highly fragmented DataFrames.
            inter_data = {
                f"{cols[i]}_x_{cols[j]}": (df[cols[i]] * df[cols[j]]).values
                for i in range(len(cols))
                for j in range(i + 1, len(cols))
            }
            inter_df = pd.DataFrame(inter_data, index=df.index)
            features.append(inter_df)

        # =========================
        # DATETIME FEATURES
        # =========================
        if self.config.create_datetime_features:
            dt_df = self._extract_datetime(df)
            if not dt_df.empty:
                features.append(dt_df)

        # =========================
        # TEXT FEATURES
        # =========================
        if self.config.create_text_features:
            text_df = self._extract_text_features(df)
            if not text_df.empty:
                features.append(text_df)

        # =========================
        # FREQUENCY ENCODING
        # =========================
        if self.config.frequency_encoding:
            freq_df = pd.DataFrame()
            for col in self.cat_cols:
                freq_df[col + "_freq"] = df[col].map(self.freq_maps[col]).fillna(0)

            if not freq_df.empty:
                features.append(freq_df)

        final_df = pd.concat(features, axis=1)

        self._log("output_shape", final_df.shape)

        return final_df

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    # =========================
    # TEXT FEATURES
    # =========================
    def _extract_text_features(self, df):
        text_cols = df.select_dtypes(include=["object"]).columns
        result = pd.DataFrame()

        for col in text_cols:
            col_data = df[col].astype(str)

            result[col + "_len"] = col_data.str.len()
            result[col + "_word_count"] = col_data.str.split().apply(len)
            result[col + "_unique_words"] = col_data.apply(lambda x: len(set(x.split())))
            result[col + "_uppercase_ratio"] = col_data.apply(
                lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
            )

        return result

    # =========================
    # DATETIME FEATURES
    # =========================
    def _extract_datetime(self, df):
        result = pd.DataFrame()

        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                result[col + "_year"] = df[col].dt.year
                result[col + "_month"] = df[col].dt.month
                result[col + "_day"] = df[col].dt.day
                result[col + "_weekday"] = df[col].dt.weekday

        return result

    # =========================
    # LOGGER
    # =========================
    def _log(self, key, value):
        self.report[key] = value

    def get_report(self):
        return self.report