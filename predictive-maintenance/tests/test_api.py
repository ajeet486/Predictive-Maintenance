"""
FastAPI Backend Unit & Integration Tests
Directly tests API schemas, validation, and endpoint handlers for lightning-fast execution.
"""

import os
import sys
import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.main import (
    app, get_health, get_models_info, predict_machine_failure,
    detect_sensor_anomaly, predict_turbofan_rul
)
from api.schemas import (
    FailurePredictionRequest, AnomalyDetectionRequest, RULPredictionRequest
)


def test_health_endpoint():
    res = get_health()
    assert res.status in ["healthy", "degraded"]
    assert res.models_loaded["failure_prediction"] is True
    assert res.models_loaded["anomaly_detection"] is True
    assert res.models_loaded["rul_prediction"] is True


def test_models_endpoint():
    res = get_models_info()
    assert len(res.available_endpoints) >= 5
    assert "failure_classifier" in res.models
    assert "rul_regressor" in res.models


def test_predict_failure_nominal():
    req = FailurePredictionRequest(
        type="M",
        air_temperature_k=298.0,
        process_temperature_k=308.0,
        rotational_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=20.0
    )
    res = predict_machine_failure(req)
    assert res.failure_predicted is False
    assert res.failure_probability < 0.50
    assert res.risk_tier == "NOMINAL / LOW RISK"
    assert len(res.top_risk_drivers) > 0


def test_predict_failure_high_risk():
    req = FailurePredictionRequest(
        type="L",
        air_temperature_k=302.0,
        process_temperature_k=312.0,
        rotational_speed_rpm=2800.0,
        torque_nm=75.0,
        tool_wear_min=240.0
    )
    res = predict_machine_failure(req)
    assert res.failure_predicted is True
    assert res.failure_probability > 0.50
    assert res.risk_tier in ["ELEVATED / WARNING", "CRITICAL / HIGH RISK"]
    assert len(res.top_risk_drivers) > 0


def test_predict_failure_validation_error():
    with pytest.raises(ValidationError):
        FailurePredictionRequest(
            type="INVALID_TYPE",
            air_temperature_k=298.0,
            process_temperature_k=308.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=20.0
        )


def test_predict_anomaly_endpoint():
    req = AnomalyDetectionRequest(
        type="H",
        air_temperature_k=298.0,
        process_temperature_k=308.0,
        rotational_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=15.0
    )
    res = detect_sensor_anomaly(req)
    assert isinstance(res.is_anomaly, bool)
    assert 0.0 <= res.anomaly_score <= 1.0


def test_predict_rul_endpoint():
    req = RULPredictionRequest(
        unit_nr=1,
        time_cycles=100,
        setting_1=-0.0007,
        setting_2=-0.0004,
        setting_3=100.0,
        sensors={
            "s_2": 642.5, "s_3": 1588.0, "s_4": 1400.0,
            "s_7": 554.0, "s_8": 2388.0, "s_9": 9050.0,
            "s_11": 47.3, "s_12": 522.0, "s_13": 2388.0,
            "s_14": 8130.0, "s_15": 8.40, "s_17": 392.0,
            "s_20": 39.0, "s_21": 23.4
        }
    )
    res = predict_turbofan_rul(req)
    assert 0.0 <= res.predicted_rul_cycles <= 125.0
    assert "prognostic_status" in res.model_dump()
