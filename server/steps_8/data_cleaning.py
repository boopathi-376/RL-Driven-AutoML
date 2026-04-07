"""
Universal Data Cleaning Module
Supports:
- Tabular (DataFrame)
- Raw Text (str)
- List of Text
- Mixed inputs (dict)

Production-ready for ML + NLP pipelines
"""

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass


# =========================
# CONFIG
# =========================

@dataclass
class CleaningConfig:
    # Tabular
    missing_threshold: float = 0.4
    numeric_strategy: str = "median"
    categorical_strategy: str = "mode"
    constant_fill_value: str = "unknown"
    outlier_method: str = "iqr"
    drop_duplicates: bool = True

    # Text
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_special_chars: bool = True
    remove_extra_spaces: bool = True
    normalize_unicode: bool = True


# =========================
# MAIN CLEANER
# =========================

class DataCleaner:
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.report = {}

    # =========================
    # UNIVERSAL ENTRY
    # =========================
    def clean(self, data: Any) -> Any:
        if isinstance(data, pd.DataFrame):
            return self._clean_dataframe(data)

        elif isinstance(data, str):
            return self._clean_text(data)

        elif isinstance(data, list):
            return self._clean_list(data)

        elif isinstance(data, dict):
            return self._clean_dict(data)

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    # =========================
    # TEXT CLEANING
    # =========================
    def _clean_text(self, text: str) -> str:
        original = text

        if not isinstance(text, str):
            return text

        if self.config.normalize_unicode:
            text = text.encode("utf-8", "ignore").decode("utf-8")

        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_html:
            text = re.sub(r"<.*?>", " ", text)

        if self.config.remove_urls:
            text = re.sub(r"http\S+|www\S+", " ", text)

        if self.config.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        if self.config.remove_extra_spaces:
            text = re.sub(r"\s+", " ", text).strip()

        self._log("text_cleaned", True)

        return text

    def _clean_list(self, data: List[Any]) -> List[Any]:
        return [self.clean(item) for item in data]

    def _clean_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self.clean(v) for k, v in data.items()}

    # =========================
    # TABULAR CLEANING
    # =========================
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        self._log("initial_shape", df.shape)

        df.columns = self._standardize_columns(df.columns)
        df = self._drop_high_missing_columns(df)
        df = self._infer_types(df)
        df = self._handle_missing(df)
        df = self._handle_outliers(df)

        # APPLY TEXT CLEANING INSIDE DF
        df = self._clean_text_columns(df)

        if self.config.drop_duplicates:
            df = self._drop_duplicates(df)

        self._log("final_shape", df.shape)

        return df

    def _standardize_columns(self, cols):
        standardized = (
            pd.Series(cols)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w_]", "", regex=True)  # Keep underscores, remove other special chars
        )
        
        # Handle duplicate column names
        if standardized.duplicated().any():
            # Add suffixes to duplicates
            standardized = standardized.to_list()
            seen = {}
            for i, col in enumerate(standardized):
                if col in seen:
                    seen[col] += 1
                    standardized[i] = f"{col}_{seen[col]}"
                else:
                    seen[col] = 0
        
        return standardized

    def _drop_high_missing_columns(self, df):
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > self.config.missing_threshold].index

        self._log("dropped_columns_missing", list(cols_to_drop))
        return df.drop(columns=cols_to_drop)

    def _infer_types(self, df):
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    # errors='coerce' turns unparseable values to NaN;
                    # only adopt the conversion if at least half the non-null
                    # values successfully parsed (avoids wiping string columns).
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if converted.notna().sum() >= df[col].notna().sum() * 0.5:
                        df[col] = converted
                except Exception:
                    pass
        return df

    def _handle_missing(self, df):
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = self._fill_numeric(df[col])
            else:
                df[col] = self._fill_categorical(df[col])

        return df

    def _fill_numeric(self, series):
        if self.config.numeric_strategy == "mean":
            return series.fillna(series.mean())
        elif self.config.numeric_strategy == "median":
            return series.fillna(series.median())
        elif self.config.numeric_strategy == "zero":
            return series.fillna(0)
        else:
            raise ValueError("Invalid numeric strategy")

    def _fill_categorical(self, series):
        if self.config.categorical_strategy == "mode":
            return series.fillna(series.mode().iloc[0] if not series.mode().empty else "unknown")
        elif self.config.categorical_strategy == "constant":
            return series.fillna(self.config.constant_fill_value)
        else:
            raise ValueError("Invalid categorical strategy")

    def _handle_outliers(self, df):
        if self.config.outlier_method == "none":
            return df

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df[col] = df[col].clip(lower, upper)

        return df

    def _clean_text_columns(self, df):
        text_cols = df.select_dtypes(include=["object"]).columns

        for col in text_cols:
            df[col] = df[col].apply(lambda x: self._clean_text(x) if isinstance(x, str) else x)

        return df

    def _drop_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        self._log("duplicates_removed", before - after)
        return df

    # =========================
    # LOGGER
    # =========================
    def _log(self, key, value):
        self.report[key] = value

    def get_report(self):
        return self.report