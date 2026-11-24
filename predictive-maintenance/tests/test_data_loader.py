"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_ai4i, load_cmapss, get_dataset_metadata, AI4I_COLUMN_MAPPING, CMAPSS_COLUMNS


def test_load_ai4i_shape_and_columns():
    df = load_ai4i(clean_names=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000
    expected_cols = list(AI4I_COLUMN_MAPPING.values())
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert df["machine_failure"].isin([0, 1]).all()


def test_load_cmapss_fd001():
    train_df, test_df, rul_df = load_cmapss("FD001")
    assert len(train_df) == 20631
    assert len(test_df) == 13096
    assert len(rul_df) == 100
    for col in CMAPSS_COLUMNS:
        assert col in train_df.columns
        assert col in test_df.columns
    assert "rul_ground_truth" in rul_df.columns


def test_load_cmapss_invalid_subset():
    with pytest.raises(ValueError):
        load_cmapss("FD999")


def test_get_dataset_metadata():
    meta = get_dataset_metadata()
    assert "ai4i_2020" in meta
    assert "nasa_cmapss" in meta
    assert "license" in meta["ai4i_2020"]
    assert "citation" in meta["nasa_cmapss"]
