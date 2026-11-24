"""
Feature Engineering Module for Predictive Maintenance & Failure Detection System
Constructs physical domain indicators and temporal aggregations:
1. AI4I Milling: Thermodynamics, mechanical power, and wear-stress interaction terms.
2. NASA C-MAPSS: Rolling statistics, degradation deltas, and operational cycle normalizations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def add_domain_features_ai4i(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific physical indicators for AI4I 2020 milling dataset.

    Physics Formulations:
    - Temperature Difference: Delta_T = Process_Temp - Air_Temp (indicates thermal dissipation)
    - Mechanical Power (kW): P = Torque * Rotational_Speed * (2*pi / 60) / 1000
    - Overstrain Interaction: Tool_Wear * Torque (captures torque shock under tool degradation)
    - Temperature Ratio: Process_Temp / Air_Temp
    - Speed to Torque Ratio: Rotational_Speed / (Torque + 1e-5)

    Args:
        df: Input DataFrame with raw sensor columns.

    Returns:
        pd.DataFrame: Augmented DataFrame with domain features.
    """
    df = df.copy()

    # Temperature difference [K]
    if "process_temperature_k" in df.columns and "air_temperature_k" in df.columns:
        df["temp_diff_k"] = df["process_temperature_k"] - df["air_temperature_k"]
        df["temp_ratio"] = df["process_temperature_k"] / df["air_temperature_k"]

    # Mechanical power in kW (2*pi*N*T / 60000)
    if "rotational_speed_rpm" in df.columns and "torque_nm" in df.columns:
        df["power_kw"] = (
            df["rotational_speed_rpm"] * df["torque_nm"] * (2 * np.pi / 60.0) / 1000.0
        )
        df["speed_torque_ratio"] = df["rotational_speed_rpm"] / (df["torque_nm"] + 1e-5)

    # Mechanical overstrain stress term (Tool wear [min] * Torque [Nm])
    if "tool_wear_min" in df.columns and "torque_nm" in df.columns:
        df["overstrain_calc"] = df["tool_wear_min"] * df["torque_nm"]

    return df


def add_rolling_temporal_features_cmapss(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_sizes: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Generate rolling statistics (mean, std, min, max, delta) along engine operating cycles.
    Prevents data leakage by strictly calculating aggregations per engine unit.

    Args:
        df: Input C-MAPSS DataFrame containing 'unit_nr' and 'time_cycles'.
        sensor_cols: List of active sensor column names to aggregate.
        window_sizes: Window cycle lengths (default: [5, 10, 20]).

    Returns:
        pd.DataFrame: DataFrame augmented with rolling temporal features.
    """
    df = df.copy()
    if window_sizes is None:
        window_sizes = [5, 10, 20]

    # Ensure chronological ordering per engine unit
    df = df.sort_values(by=["unit_nr", "time_cycles"]).reset_index(drop=True)

    grouped = df.groupby("unit_nr")

    for w in window_sizes:
        # Rolling Mean
        rolling_mean = (
            grouped[sensor_cols]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolling_mean.columns = [f"{col}_roll_mean_{w}" for col in sensor_cols]

        # Rolling Std (fill min_periods=1 NaN std with 0)
        rolling_std = (
            grouped[sensor_cols]
            .rolling(window=w, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        rolling_std.columns = [f"{col}_roll_std_{w}" for col in sensor_cols]

        # Rolling Min
        rolling_min = (
            grouped[sensor_cols]
            .rolling(window=w, min_periods=1)
            .min()
            .reset_index(level=0, drop=True)
        )
        rolling_min.columns = [f"{col}_roll_min_{w}" for col in sensor_cols]

        # Rolling Max
        rolling_max = (
            grouped[sensor_cols]
            .rolling(window=w, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        rolling_max.columns = [f"{col}_roll_max_{w}" for col in sensor_cols]

        df = pd.concat([df, rolling_mean, rolling_std, rolling_min, rolling_max], axis=1)

    return df
