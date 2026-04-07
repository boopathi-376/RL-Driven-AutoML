"""
scaling.py

Universal Scaling Module
Supports:
- Standard / MinMax / Robust scaling
- Sparse-safe scaling (for TF-IDF)
- Mixed numeric + encoded features
- RL-compatible configs
"""

import numpy as np
import pandas as pd
from typing import Any, Optional
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import sparse


# =========================
# CONFIG
# =========================

@dataclass
class ScalingConfig:
    method: str = "standard"   # standard / minmax / robust / none
    with_mean: bool = False    # MUST be False for sparse safety
    with_std: bool = True


# =========================
# MAIN SCALER
# =========================

class Scaler:
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.scaler = None
        self.fitted = False
        self.report = {}

    # =========================
    # UNIVERSAL ENTRY
    # =========================
    def fit(self, data: Any):
        data = self._to_numpy(data)

        if self.config.method == "none":
            self.fitted = True
            return self

        if sparse.issparse(data):
            # ⚠️ Sparse-safe scaling
            self.scaler = StandardScaler(
                with_mean=False,
                with_std=self.config.with_std
            )
        else:
            self.scaler = self._get_scaler()

        self.scaler.fit(data)
        self.fitted = True

        self._log("method_used", self.config.method)
        self._log("input_shape", data.shape)

        return self

    def transform(self, data: Any):
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted first")

        data_np = self._to_numpy(data)

        if self.config.method == "none":
            return data

        scaled = self.scaler.transform(data_np)

        self._log("output_shape", scaled.shape)

        return scaled

    def fit_transform(self, data: Any):
        return self.fit(data).transform(data)

    # =========================
    # INTERNAL UTILITIES
    # =========================
    def _get_scaler(self):
        if self.config.method == "standard":
            return StandardScaler(
                with_mean=self.config.with_mean,
                with_std=self.config.with_std
            )

        elif self.config.method == "minmax":
            return MinMaxScaler()

        elif self.config.method == "robust":
            return RobustScaler()

        else:
            raise ValueError(f"Unknown scaling method: {self.config.method}")

    def _to_numpy(self, data: Any):
        if isinstance(data, pd.DataFrame):
            return data.values

        if isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)

        if sparse.issparse(data):
            return data

        if isinstance(data, list):
            return np.array(data)

        if isinstance(data, np.ndarray):
            return data

        raise TypeError(f"Unsupported data type: {type(data)}")

    # =========================
    # LOGGER
    # =========================
    def _log(self, key, value):
        self.report[key] = value

    def get_report(self):
        return self.report