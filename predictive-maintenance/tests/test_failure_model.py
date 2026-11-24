"""
Unit tests for preprocessing and failure_model modules
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_ai4i
from src.preprocessing import AI4IPreprocessor
from src.failure_model import FailurePredictionPipeline, map_failure_modes


def test_ai4i_preprocessor():
    df = load_ai4i()
    preprocessor = AI4IPreprocessor(scale_features=True)
    X = df.drop(columns=["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"])
    
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)

    assert isinstance(X_trans, pd.DataFrame)
    assert len(X_trans) == 10000
    assert not X_trans.isnull().values.any()
    # Check that domain features were generated
    for feat in ["temp_diff_k", "power_kw", "overstrain_calc"]:
        assert feat in X_trans.columns


def test_failure_pipeline_artifacts():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    pipeline_path = os.path.join(base_dir, "models", "failure_pipeline.joblib")
    assert os.path.exists(pipeline_path), "failure_pipeline.joblib must exist"

    artifact = joblib.load(pipeline_path)
    assert "preprocessor" in artifact
    assert "best_model" in artifact
    assert "best_model_name" in artifact
    assert artifact["best_model_name"] in ["Random_Forest", "XGBoost", "Logistic_Regression"]

    # Test inference on a single row
    sample_raw = pd.DataFrame([{
        "type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0
    }])
    X_proc = artifact["preprocessor"].transform(sample_raw)
    prob = artifact["best_model"].predict_proba(X_proc)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_map_failure_modes():
    sample_df = pd.DataFrame({
        "rnf": [0, 0, 0, 0, 1],
        "twf": [1, 0, 0, 0, 0],
        "pwf": [0, 1, 0, 0, 0],
        "hdf": [0, 0, 1, 0, 0],
        "osf": [0, 0, 0, 1, 0]
    })
    mapped = map_failure_modes(sample_df)
    assert list(mapped.values) == [1, 3, 2, 4, 5]
