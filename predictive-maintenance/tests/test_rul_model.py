"""
Unit tests for RUL modeling and CMAPSS preprocessing
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import CMAPSSPreprocessor
from src.evaluation import nasa_asymmetric_scoring, evaluate_rul_regressor


def test_nasa_asymmetric_scoring_properties():
    # Early prediction: predicted 90 when true is 100 (d = -10)
    # Late prediction: predicted 110 when true is 100 (d = +10)
    score_early, _ = nasa_asymmetric_scoring(np.array([100]), np.array([90]))
    score_late, _ = nasa_asymmetric_scoring(np.array([100]), np.array([110]))

    # Late prediction must incur a substantially harsher penalty
    assert score_late > score_early
    assert score_early > 0.0
    assert score_late > 0.0


def test_cmapss_preprocessor_capped_rul():
    prep = CMAPSSPreprocessor(max_rul=125)
    sample_df = pd.DataFrame({
        "unit_nr": [1, 1, 1, 1],
        "time_cycles": [1, 50, 150, 200]
    })
    # Max cycle is 200 -> raw RULs: 199, 150, 50, 0
    # Capped at 125 -> 125, 125, 50, 0
    capped = prep.compute_capped_rul(sample_df)
    assert list(capped.values) == [125, 125, 50, 0]


def test_evaluate_rul_regressor():
    y_true = np.array([100.0, 50.0, 25.0])
    y_pred = np.array([95.0, 52.0, 20.0])
    res = evaluate_rul_regressor(y_true, y_pred, model_name="TestModel")

    assert "mae" in res
    assert "rmse" in res
    assert "r2_score" in res
    assert "nasa_total_score" in res
    assert res["mae"] > 0


def test_saved_rul_artifact():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    artifact_path = os.path.join(base_dir, "models", "rul_pipeline.joblib")
    assert os.path.exists(artifact_path), "rul_pipeline.joblib must exist"

    artifact = joblib.load(artifact_path)
    assert "preprocessor" in artifact
    assert "best_model" in artifact
    assert artifact["best_model_name"] in ["Random_Forest", "XGBoost"]
    assert artifact["max_rul"] == 125
