"""
Unit tests for explainability module (SHAP)
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.explainability import FailureExplainer


def test_failure_explainer_initialization_and_inference():
    explainer = FailureExplainer()
    assert explainer.model is not None
    assert explainer.feature_names is not None
    assert isinstance(explainer.expected_value, float)

    # Test single instance explanation
    sample_raw = pd.DataFrame([{
        "type": "L",
        "air_temperature_k": 301.0,
        "process_temperature_k": 311.0,
        "rotational_speed_rpm": 1400.0,
        "torque_nm": 65.0,
        "tool_wear_min": 215.0
    }])

    res = explainer.explain_instance(sample_raw)
    assert "failure_probability" in res
    assert "risk_tier" in res
    assert "top_risk_drivers" in res
    assert "all_contributions" in res
    assert len(res["top_risk_drivers"]) > 0
    assert 0.0 <= res["failure_probability"] <= 1.0


def test_shap_figures_exist():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    summary_plot = os.path.join(base_dir, "reports", "figures", "shap_summary_plot.png")
    bar_plot = os.path.join(base_dir, "reports", "figures", "shap_global_bar.png")
    assert os.path.exists(summary_plot), "shap_summary_plot.png must exist"
    assert os.path.exists(bar_plot), "shap_global_bar.png must exist"
