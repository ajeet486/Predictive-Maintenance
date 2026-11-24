"""
Unit tests for anomaly_model module
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.anomaly_model import AnomalyDetectionPipeline


def test_anomaly_pipeline_fit_and_score():
    sample_df = pd.DataFrame({
        "type": ["L", "M", "H", "L", "M"] * 20,
        "air_temperature_k": np.random.normal(300, 2, 100),
        "process_temperature_k": np.random.normal(310, 2, 100),
        "rotational_speed_rpm": np.random.normal(1500, 100, 100),
        "torque_nm": np.random.normal(40, 5, 100),
        "tool_wear_min": np.random.uniform(0, 200, 100)
    })

    pipe = AnomalyDetectionPipeline(contamination=0.05, random_state=42)
    pipe.fit(sample_df)

    is_anomaly, scores = pipe.score(sample_df)

    assert len(is_anomaly) == 100
    assert len(scores) == 100
    assert set(np.unique(is_anomaly)).issubset({0, 1})
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_saved_anomaly_artifact():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    artifact_path = os.path.join(base_dir, "models", "anomaly_pipeline.joblib")
    assert os.path.exists(artifact_path), "anomaly_pipeline.joblib must exist"

    artifact = joblib.load(artifact_path)
    assert "preprocessor" in artifact
    assert "model" in artifact
    assert "threshold" in artifact

    # Test scoring with nominal sample
    sample_raw = pd.DataFrame([{
        "type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0
    }])
    X_proc = artifact["preprocessor"].transform(sample_raw)
    raw_pred = artifact["model"].predict(X_proc)[0]
    assert raw_pred in [1, -1]
