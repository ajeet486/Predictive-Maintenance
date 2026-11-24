"""
Preprocessing Pipeline Module for Predictive Maintenance & Failure Detection
Provides robust, scikit-learn compatible transformers for both datasets:
1. AI4IPreprocessor: Handles milling telemetry encoding, domain expansion, and scaling.
2. CMAPSSPreprocessor: Handles turbofan invariant sensor elimination, rolling window creation, and RUL target formatting.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_engineering import add_domain_features_ai4i, add_rolling_temporal_features_cmapss


class AI4IPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for AI4I 2020 milling sensor data.
    - Encodes machine quality Type (L, M, H) -> Ordinal (0, 1, 2)
    - Generates thermodynamic and mechanical interaction features
    - Drops ID columns and component failure target leakage columns
    - Scales numeric features using StandardScaler
    """

    # Columns that represent identifiers or ground-truth leakage
    EXCLUDE_COLUMNS = [
        "udi", "product_id", "machine_failure",
        "twf", "hdf", "pwf", "osf", "rnf"
    ]

    TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.scaler = StandardScaler()
        self.feature_names_out_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        processed_df = self._extract_features(X_df)
        self.feature_names_out_ = list(processed_df.columns)

        if self.scale_features:
            self.scaler.fit(processed_df)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("AI4IPreprocessor is not fitted yet. Call fit() first.")

        X_df = X.copy()
        processed_df = self._extract_features(X_df)

        # Ensure column alignment with training set
        for col in self.feature_names_out_:
            if col not in processed_df.columns:
                processed_df[col] = 0.0
        processed_df = processed_df[self.feature_names_out_]

        if self.scale_features:
            scaled_vals = self.scaler.transform(processed_df)
            return pd.DataFrame(scaled_vals, columns=self.feature_names_out_, index=X_df.index)

        return processed_df

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Handle column names if raw
        df.columns = [c.strip().lower() for c in df.columns]

        # Ordinal encode Type
        type_col = None
        for col in ["type", "product type"]:
            if col in df.columns:
                type_col = col
                break

        if type_col is not None:
            df["type_encoded"] = df[type_col].map(self.TYPE_MAPPING).fillna(0).astype(int)
        else:
            df["type_encoded"] = 0

        # Add domain physics features
        df = add_domain_features_ai4i(df)

        # Drop excluded/identifier columns
        drop_cols = [c for c in df.columns if any(c.startswith(exc) for exc in self.EXCLUDE_COLUMNS)]
        if type_col in df.columns:
            drop_cols.append(type_col)

        features_df = df.drop(columns=drop_cols, errors="ignore")
        return features_df


class CMAPSSPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for NASA C-MAPSS Turbofan Engine Degradation Dataset.
    - Filters out invariant / zero-variance sensor channels
    - Computes piecewise linear capped RUL targets for run-to-failure series
    - Generates rolling statistical aggregations (mean, std, min, max)
    - Normalizes telemetry using MinMaxScaler (-1, 1) to match turbofan modeling standards
    """

    # Invariant sensors identified during EDA in FD001
    DEFAULT_INVARIANT_SENSORS = [
        "s_1", "s_5", "s_10", "s_16", "s_18", "s_19", "setting_3"
    ]

    def __init__(
        self,
        max_rul: int = 125,
        window_sizes: Optional[List[int]] = None,
        scale_features: bool = True
    ):
        self.max_rul = max_rul
        self.window_sizes = window_sizes or [5, 10, 20]
        self.scale_features = scale_features
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.active_sensors: List[str] = []
        self.feature_names_out_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, y=None):
        df = train_df.copy()

        # Identify all sensor columns
        all_sensor_cols = [c for c in df.columns if c.startswith("s_") or c.startswith("setting_")]
        self.active_sensors = [c for c in all_sensor_cols if c not in self.DEFAULT_INVARIANT_SENSORS]

        # Build rolling features on train
        feat_df = add_rolling_temporal_features_cmapss(
            df, sensor_cols=self.active_sensors, window_sizes=self.window_sizes
        )

        # Drop unit and cycle columns from model input features
        drop_cols = ["unit_nr", "time_cycles", "RUL", "wear_pct"] + self.DEFAULT_INVARIANT_SENSORS
        clean_feat = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns], errors="ignore")

        self.feature_names_out_ = list(clean_feat.columns)

        if self.scale_features:
            self.scaler.fit(clean_feat)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("CMAPSSPreprocessor is not fitted. Call fit() first.")

        df = df.copy()
        feat_df = add_rolling_temporal_features_cmapss(
            df, sensor_cols=self.active_sensors, window_sizes=self.window_sizes
        )

        drop_cols = ["unit_nr", "time_cycles", "RUL", "wear_pct"] + self.DEFAULT_INVARIANT_SENSORS
        clean_feat = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns], errors="ignore")

        # Column alignment
        for col in self.feature_names_out_:
            if col not in clean_feat.columns:
                clean_feat[col] = 0.0
        clean_feat = clean_feat[self.feature_names_out_]

        if self.scale_features:
            scaled_vals = self.scaler.transform(clean_feat)
            return pd.DataFrame(scaled_vals, columns=self.feature_names_out_, index=df.index)

        return clean_feat

    def compute_capped_rul(self, train_df: pd.DataFrame) -> pd.Series:
        """
        Compute piecewise linear capped RUL target for run-to-failure training records.
        """
        df = train_df.copy()
        max_cycles_per_unit = df.groupby("unit_nr")["time_cycles"].transform("max")
        raw_rul = max_cycles_per_unit - df["time_cycles"]
        capped_rul = raw_rul.clip(upper=self.max_rul)
        return capped_rul
