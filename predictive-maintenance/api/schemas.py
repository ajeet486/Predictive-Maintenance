"""
Pydantic Schemas for FastAPI Predictive Maintenance API
Validates inputs and formats responses for failure, anomaly, and RUL endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"


class ModelMetadata(BaseModel):
    model_name: str
    model_type: str
    target: str
    input_features: List[str]
    training_dataset: str


class ModelsInfoResponse(BaseModel):
    available_endpoints: List[str]
    models: Dict[str, ModelMetadata]


class FailurePredictionRequest(BaseModel):
    type: str = Field(
        ...,
        description="Machine quality variant: L (Low - 50%), M (Medium - 30%), H (High - 20%)",
        example="M"
    )
    air_temperature_k: float = Field(
        ...,
        description="Ambient air temperature in Kelvin (standard range: 295.0 - 305.0 K)",
        example=298.5
    )
    process_temperature_k: float = Field(
        ...,
        description="Process operating temperature in Kelvin (standard range: 305.0 - 315.0 K)",
        example=308.8
    )
    rotational_speed_rpm: float = Field(
        ...,
        description="Spindle rotational speed in RPM (standard range: 1100 - 2900 RPM)",
        example=1450.0
    )
    torque_nm: float = Field(
        ...,
        description="Spindle torque in Newton-meters (standard range: 10.0 - 80.0 Nm)",
        example=45.2
    )
    tool_wear_min: float = Field(
        ...,
        description="Cumulative cutting tool wear in minutes (standard range: 0 - 250 min)",
        example=120.0
    )

    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in {"L", "M", "H"}:
            raise ValueError("Machine Type must be 'L', 'M', or 'H'")
        return v_upper

    @field_validator("air_temperature_k", "process_temperature_k")
    def validate_temps(cls, v: float) -> float:
        if v < 200.0 or v > 400.0:
            raise ValueError(f"Temperature {v} K is outside realistic physical limits (200K - 400K)")
        return v

    @field_validator("rotational_speed_rpm")
    def validate_rpm(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Rotational speed must be positive")
        return v

    @field_validator("torque_nm", "tool_wear_min")
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Torque and tool wear must be non-negative")
        return v


class RiskDriver(BaseModel):
    feature: str
    shap_impact: float
    scaled_value: float


class FailurePredictionResponse(BaseModel):
    failure_predicted: bool
    failure_probability: float
    risk_tier: str
    failure_mode_prediction: str
    top_risk_drivers: List[RiskDriver]
    top_safety_drivers: List[RiskDriver]
    model_name: str


class AnomalyDetectionRequest(FailurePredictionRequest):
    """Shares identical sensor inputs with failure prediction for unified milling telemetry."""
    pass


class AnomalyDetectionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(..., description="Calibrated score [0.0 = Nominal, 1.0 = Extreme Outlier]")
    status: str
    nominal_threshold: float
    interpretation: str


class RULPredictionRequest(BaseModel):
    unit_nr: Optional[int] = Field(1, description="Engine unit identification number", example=1)
    time_cycles: Optional[int] = Field(120, description="Current operational cycle number", example=120)
    setting_1: Optional[float] = Field(0.001, description="Operational setting 1 (altitude / mach proxy)", example=0.001)
    setting_2: Optional[float] = Field(0.0002, description="Operational setting 2 (throttle proxy)", example=0.0002)
    setting_3: Optional[float] = Field(100.0, description="Operational setting 3 (TRA)", example=100.0)
    sensors: Dict[str, float] = Field(
        ...,
        description="Key-value mapping of sensor channels (e.g. {'s_2': 642.5, 's_3': 1588.0, ...})",
        example={
            "s_2": 642.8, "s_3": 1589.5, "s_4": 1405.0,
            "s_7": 553.2, "s_8": 2388.1, "s_9": 9050.2,
            "s_11": 47.4, "s_12": 521.6, "s_13": 2388.0,
            "s_14": 8130.0, "s_15": 8.42, "s_17": 393.0,
            "s_20": 38.8, "s_21": 23.3
        }
    )


class RULPredictionResponse(BaseModel):
    unit_nr: int
    current_cycle: int
    predicted_rul_cycles: float
    prognostic_status: str
    capped_rul_limit: int
    model_name: str
