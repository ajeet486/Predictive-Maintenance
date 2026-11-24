"""
FastAPI Microservice Backend for Smart Predictive Maintenance System
Serves endpoints:
- GET  /health
- GET  /models
- POST /predict/failure
- POST /predict/anomaly
- POST /predict/rul
"""

import os
import sys
import datetime
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.schemas import (
    HealthResponse, ModelsInfoResponse, ModelMetadata,
    FailurePredictionRequest, FailurePredictionResponse, RiskDriver,
    AnomalyDetectionRequest, AnomalyDetectionResponse,
    RULPredictionRequest, RULPredictionResponse
)
from src.explainability import FailureExplainer

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Global model store
model_store: Dict[str, Any] = {}
explainer_engine: Optional[FailureExplainer] = None


def load_all_models():
    """
    Synchronously load all serialized model pipelines into global store.
    """
    global model_store, explainer_engine
    if model_store:
        return

    print("[INFO] Loading predictive maintenance model pipelines...")
    failure_path = os.path.join(MODELS_DIR, "failure_pipeline.joblib")
    if os.path.exists(failure_path):
        model_store["failure"] = joblib.load(failure_path)
        print("  -> Loaded failure prediction pipeline.")

    multi_path = os.path.join(MODELS_DIR, "multiclass_failure_pipeline.joblib")
    if os.path.exists(multi_path):
        model_store["multiclass"] = joblib.load(multi_path)
        print("  -> Loaded multiclass failure mode pipeline.")

    anomaly_path = os.path.join(MODELS_DIR, "anomaly_pipeline.joblib")
    if os.path.exists(anomaly_path):
        model_store["anomaly"] = joblib.load(anomaly_path)
        print("  -> Loaded anomaly detection pipeline.")

    rul_path = os.path.join(MODELS_DIR, "rul_pipeline.joblib")
    if os.path.exists(rul_path):
        model_store["rul"] = joblib.load(rul_path)
        print("  -> Loaded RUL prediction pipeline.")

    try:
        explainer_engine = FailureExplainer()
        print("  -> Initialized SHAP FailureExplainer engine.")
    except Exception as e:
        print(f"  [WARN] Could not initialize SHAP explainer: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


app = FastAPI(
    title="Smart Predictive Maintenance & Failure Detection API",
    description=(
        "Production REST API for real-time machine failure prediction, "
        "unsupervised sensor anomaly detection, Remaining Useful Life (RUL) estimation, "
        "and SHAP game-theoretic explainability."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Ensure models are loaded immediately
load_all_models()

# Enable CORS for Streamlit / external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System Health"])
def get_health():
    """
    System health check returning loaded models status and service vitality.
    """
    models_status = {
        "failure_prediction": "failure" in model_store,
        "multiclass_classification": "multiclass" in model_store,
        "anomaly_detection": "anomaly" in model_store,
        "rul_prediction": "rul" in model_store,
        "shap_explainability": explainer_engine is not None
    }
    all_healthy = all(models_status.values())
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        models_loaded=models_status,
        version="1.0.0"
    )


@app.get("/models", response_model=ModelsInfoResponse, tags=["Model Registry"])
def get_models_info():
    """
    Return metadata, architecture details, and input requirements for all registered models.
    """
    models_meta = {}

    if "failure" in model_store:
        f_pipe = model_store["failure"]
        models_meta["failure_classifier"] = ModelMetadata(
            model_name=f_pipe["best_model_name"],
            model_type="Binary Classifier (Balanced)",
            target="Machine failure (0 = Healthy, 1 = Failed)",
            input_features=f_pipe["feature_names"],
            training_dataset="AI4I 2020 Predictive Maintenance (UCI)"
        )

    if "multiclass" in model_store:
        m_pipe = model_store["multiclass"]
        models_meta["failure_mode_classifier"] = ModelMetadata(
            model_name="RandomForestClassifier",
            model_type="Multi-class Classifier",
            target="Fault Mode (Healthy, TWF, HDF, PWF, OSF, RNF)",
            input_features=m_pipe["feature_names"],
            training_dataset="AI4I 2020 Predictive Maintenance (UCI)"
        )

    if "anomaly" in model_store:
        a_pipe = model_store["anomaly"]
        models_meta["anomaly_detector"] = ModelMetadata(
            model_name="IsolationForest",
            model_type="Unsupervised Outlier Detector (Calibrated)",
            target="Sensor Anomaly (-1 = Outlier, 1 = Nominal)",
            input_features=a_pipe["feature_names"],
            training_dataset="AI4I 2020 (Trained strictly on nominal cycles)"
        )

    if "rul" in model_store:
        r_pipe = model_store["rul"]
        models_meta["rul_regressor"] = ModelMetadata(
            model_name=r_pipe["best_model_name"],
            model_type="Prognostic Regressor (Capped at 125 cycles)",
            target="Remaining Useful Life (Cycles)",
            input_features=r_pipe["feature_names"],
            training_dataset=f"NASA C-MAPSS Turbofan ({r_pipe['subset']})"
        )

    return ModelsInfoResponse(
        available_endpoints=[
            "GET /health",
            "GET /models",
            "POST /predict/failure",
            "POST /predict/anomaly",
            "POST /predict/rul"
        ],
        models=models_meta
    )


@app.post("/predict/failure", response_model=FailurePredictionResponse, tags=["Prediction"])
def predict_machine_failure(req: FailurePredictionRequest):
    """
    Predict machine failure probability, risk tier, specific failure mode, and SHAP drivers.
    """
    if "failure" not in model_store or explainer_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failure prediction model or SHAP explainer not loaded."
        )

    # Convert request to single-row DataFrame
    input_dict = {
        "type": req.type,
        "air_temperature_k": req.air_temperature_k,
        "process_temperature_k": req.process_temperature_k,
        "rotational_speed_rpm": req.rotational_speed_rpm,
        "torque_nm": req.torque_nm,
        "tool_wear_min": req.tool_wear_min
    }
    input_df = pd.DataFrame([input_dict])

    try:
        # Explain instance and get prediction
        explanation = explainer_engine.explain_instance(input_df)

        # Multi-class failure mode prediction
        failure_mode = "Normal / Healthy"
        if "multiclass" in model_store and explanation["predicted_class"] == 1:
            m_pipe = model_store["multiclass"]
            proc_X = m_pipe["preprocessor"].transform(input_df)
            m_pred = int(m_pipe["model"].predict(proc_X)[0])
            failure_mode = m_pipe["class_names"].get(m_pred, "Unknown Fault")

        # Format drivers
        top_risk = [
            RiskDriver(feature=d["feature"], shap_impact=d["shap_value"], scaled_value=d["scaled_value"])
            for d in explanation["top_risk_drivers"]
        ]
        top_safety = [
            RiskDriver(feature=d["feature"], shap_impact=d["shap_value"], scaled_value=d["scaled_value"])
            for d in explanation["top_safety_drivers"]
        ]

        return FailurePredictionResponse(
            failure_predicted=bool(explanation["predicted_class"] == 1),
            failure_probability=explanation["failure_probability"],
            risk_tier=explanation["risk_tier"],
            failure_mode_prediction=failure_mode,
            top_risk_drivers=top_risk,
            top_safety_drivers=top_safety,
            model_name=explanation["model_name"]
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/anomaly", response_model=AnomalyDetectionResponse, tags=["Prediction"])
def detect_sensor_anomaly(req: AnomalyDetectionRequest):
    """
    Evaluate machine telemetry for operational anomalies using Isolation Forest.
    """
    if "anomaly" not in model_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anomaly detection model not loaded."
        )

    pipe = model_store["anomaly"]
    input_df = pd.DataFrame([{
        "type": req.type,
        "air_temperature_k": req.air_temperature_k,
        "process_temperature_k": req.process_temperature_k,
        "rotational_speed_rpm": req.rotational_speed_rpm,
        "torque_nm": req.torque_nm,
        "tool_wear_min": req.tool_wear_min
    }])

    try:
        X_proc = pipe["preprocessor"].transform(input_df)
        raw_pred = pipe["model"].predict(X_proc)[0]  # -1 is anomaly, 1 is normal
        raw_score = pipe["model"].score_samples(X_proc)[0]

        is_anom = bool(raw_pred == -1)

        # Calibrated normalized score
        min_s, max_s = -0.75, -0.35
        norm_score = float(np.clip((max_s - raw_score) / (max_s - min_s), 0.0, 1.0))

        if is_anom:
            interpretation = (
                "Unusual multi-sensor correlation detected. Telemetry deviates from nominal "
                "operational envelope. May indicate impending failure, extreme load, or sensor drift."
            )
            status_str = "ANOMALOUS SENSOR SIGNATURE"
        else:
            interpretation = "Telemetry is consistent with nominal baseline machine operations."
            status_str = "NOMINAL OPERATING STATE"

        return AnomalyDetectionResponse(
            is_anomaly=is_anom,
            anomaly_score=round(norm_score, 4),
            status=status_str,
            nominal_threshold=round(pipe["threshold"], 4),
            interpretation=interpretation
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/rul", response_model=RULPredictionResponse, tags=["Prediction"])
def predict_turbofan_rul(req: RULPredictionRequest):
    """
    Predict Remaining Useful Life (RUL) in operational cycles for turbofan engines.
    """
    if "rul" not in model_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RUL prediction model not loaded."
        )

    pipe = model_store["rul"]
    preprocessor = pipe["preprocessor"]
    model = pipe["best_model"]
    max_rul = pipe["max_rul"]

    try:
        # Build single observation row
        row = {
            "unit_nr": req.unit_nr or 1,
            "time_cycles": req.time_cycles or 100,
            "setting_1": req.setting_1,
            "setting_2": req.setting_2,
            "setting_3": req.setting_3
        }
        # Ingest sensors
        for i in range(1, 22):
            s_key = f"s_{i}"
            row[s_key] = req.sensors.get(s_key, 0.0)

        input_df = pd.DataFrame([row])
        X_proc = preprocessor.transform(input_df)

        raw_pred = float(model.predict(X_proc)[0])
        pred_rul = float(np.clip(raw_pred, 0.0, float(max_rul)))

        if pred_rul < 25.0:
            status_str = "CRITICAL: Engine Nearing End-of-Life (<25 cycles)"
        elif pred_rul < 60.0:
            status_str = "WARNING: Accelerated Degradation Active"
        else:
            status_str = "HEALTHY: Normal Remaining Useful Life"

        return RULPredictionResponse(
            unit_nr=req.unit_nr or 1,
            current_cycle=req.time_cycles or 100,
            predicted_rul_cycles=round(pred_rul, 1),
            prognostic_status=status_str,
            capped_rul_limit=max_rul,
            model_name=pipe["best_model_name"]
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
