"""
Data Loader Module for Predictive Maintenance & Failure Detection System
Handles loading, basic validation, and initial metadata management for:
1. AI4I 2020 Predictive Maintenance Dataset (UCI)
2. NASA C-MAPSS Turbofan Degradation Simulation (FD001 - FD004)
"""

import os
import re
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

DEFAULT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI4I_RAW_DIR = os.path.join(DEFAULT_BASE_DIR, "data", "raw", "ai4i")
CMAPSS_RAW_DIR = os.path.join(DEFAULT_BASE_DIR, "data", "raw", "cmapss")

# Standardized column names for AI4I 2020
AI4I_COLUMN_MAPPING = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",  # Tool Wear Failure
    "HDF": "hdf",  # Heat Dissipation Failure
    "PWF": "pwf",  # Power Failure
    "OSF": "osf",  # Overstrain Failure
    "RNF": "rnf"   # Random Failure
}

# Standardized column names for NASA C-MAPSS
CMAPSS_COLUMNS = [
    "unit_nr",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3"
] + [f"s_{i}" for i in range(1, 22)]


def load_ai4i(data_dir: Optional[str] = None, clean_names: bool = True) -> pd.DataFrame:
    """
    Load the AI4I 2020 Predictive Maintenance Dataset from disk.

    Args:
        data_dir: Path to raw ai4i directory. Defaults to data/raw/ai4i.
        clean_names: Whether to rename columns to lowercase snake_case standard.

    Returns:
        pd.DataFrame: Loaded dataset with 10,000 observations.
    """
    target_dir = data_dir or AI4I_RAW_DIR
    csv_path = os.path.join(target_dir, "ai4i2020.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"AI4I dataset not found at '{csv_path}'. "
            f"Please run 'python scripts/download_data.py' to acquire the real dataset."
        )

    df = pd.read_csv(csv_path)

    if clean_names:
        df = df.rename(columns=AI4I_COLUMN_MAPPING)

    # Basic integrity validation
    expected_rows = 10000
    if len(df) != expected_rows:
        raise ValueError(f"Corrupt or incomplete AI4I dataset: expected {expected_rows} rows, found {len(df)}")

    return df


def load_cmapss(
    subset: str = "FD001",
    data_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load NASA C-MAPSS Turbofan Engine Degradation dataset.

    Args:
        subset: One of 'FD001', 'FD002', 'FD003', 'FD004'.
        data_dir: Path to raw cmapss directory. Defaults to data/raw/cmapss.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, test_df, rul_df)
    """
    subset = subset.upper()
    valid_subsets = {"FD001", "FD002", "FD003", "FD004"}
    if subset not in valid_subsets:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {valid_subsets}")

    target_dir = data_dir or CMAPSS_RAW_DIR
    train_file = os.path.join(target_dir, f"train_{subset}.txt")
    test_file = os.path.join(target_dir, f"test_{subset}.txt")
    rul_file = os.path.join(target_dir, f"RUL_{subset}.txt")

    for fpath in (train_file, test_file, rul_file):
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"C-MAPSS file not found at '{fpath}'. "
                f"Please run 'python scripts/download_data.py' to acquire the real dataset."
            )

    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["rul_ground_truth"])

    # Ensure engine ID and cycles are integers
    train_df["unit_nr"] = train_df["unit_nr"].astype(int)
    train_df["time_cycles"] = train_df["time_cycles"].astype(int)
    test_df["unit_nr"] = test_df["unit_nr"].astype(int)
    test_df["time_cycles"] = test_df["time_cycles"].astype(int)
    rul_df["unit_nr"] = np.arange(1, len(rul_df) + 1, dtype=int)

    return train_df, test_df, rul_df


def get_dataset_metadata() -> Dict[str, Any]:
    """
    Return license, provenance, and citation metadata for both datasets.
    """
    return {
        "ai4i_2020": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "source": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "citation": "Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. Third International Conference on Artificial Intelligence for Industries (AI4I 2020).",
            "samples": 10000,
            "target": "Machine failure (binary) & 5 failure modes (TWF, HDF, PWF, OSF, RNF)"
        },
        "nasa_cmapss": {
            "name": "NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset",
            "source": "NASA Prognostics Center of Excellence (PCoE) Data Repository",
            "url": "https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository",
            "license": "Public Domain / NASA Open Data Policy",
            "citation": "A. Saxena, K. Goebel, D. Simon, and N. Eklund, 'Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation', Prognostics and Health Management (PHM08), 2008.",
            "subsets": ["FD001", "FD002", "FD003", "FD004"],
            "target": "Remaining Useful Life (RUL) in operational flight cycles"
        }
    }
