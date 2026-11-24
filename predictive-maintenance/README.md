# Smart Predictive Maintenance & Failure Detection System

An industrial-grade predictive maintenance application that analyzes machine sensor telemetry to predict equipment failures, detect operational anomalies, estimate Remaining Useful Life (RUL), explain predictions using SHAP, and surface insights via a FastAPI backend and interactive Streamlit analytics dashboard.

---

## Architecture Overview

```
                          +-------------------------------------------------+
                          |   Industrial Sensor Telemetry Ingestion         |
                          |   (AI4I 2020 Milling & NASA C-MAPSS Turbofans)  |
                          +-------------------------------------------------+
                                                  |
                         +------------------------+-------------------------+
                         |                                                  |
                         v                                                  v
          +-----------------------------+                    +-----------------------------+
          |      Component A:           |                    |      Component B:           |
          |  AI4I Milling Failure &     |                    |  NASA C-MAPSS Turbofan      |
          |  Anomaly Detection Pipeline |                    |  RUL Regression Pipeline    |
          +-----------------------------+                    +-----------------------------+
          | - Cleaning & Scaling        |                    | - Run-to-Failure Analysis   |
          | - Domain Feature Engine     |                    | - Capped RUL (125 cycles)   |
          | - Imbalance Mitigation      |                    | - Rolling / Lag Dynamics    |
          | - Classifier Ensemble       |                    | - Regressor Ensemble        |
          |   (LogReg, RF, XGBoost)     |                    |   (Random Forest, XGBoost)  |
          | - Isolation Forest Anomaly  |                    | - NASA PHM Asymmetric Loss  |
          | - SHAP TreeExplainer        |                    | - Engine-wise Validation    |
          +-----------------------------+                    +-----------------------------+
                         |                                                  |
                         +------------------------+-------------------------+
                                                  |
                                                  v
                               +------------------------------------+
                               |     FastAPI Microservice Backend   |
                               |  - /health   - /models             |
                               |  - /predict/failure                |
                               |  - /predict/anomaly                |
                               |  - /predict/rul                    |
                               +------------------------------------+
                                                  |
                                                  v
                               +------------------------------------+
                               |    Streamlit Industrial Dashboard  |
                               |  - Overview & Fleet Health         |
                               |  - Failure Risk & SHAP Analysis    |
                               |  - Real-Time Anomaly Scoring       |
                               |  - Turbofan Engine RUL Monitor     |
                               |  - Model Benchmark & Metrics       |
                               +------------------------------------+
```

---

## Datasets & Licensing

| Dataset | Provider | Records | License | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **AI4I 2020 Predictive Maintenance** | UCI Machine Learning Repository | 10,000 milling cycles | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Binary machine failure, 5 failure modes (TWF, HDF, PWF, OSF, RNF), anomaly scoring |
| **NASA C-MAPSS Turbofan Engine** | NASA Prognostics Center of Excellence (PCoE) | FD001 (20,631 cycles train / 13,096 test), extensible to FD002-FD004 | Public Domain / NASA Open Data | Remaining Useful Life (RUL) prognostic regression |

---

## Directory Structure

```
predictive-maintenance/
    data/
        raw/
            ai4i/               # AI4I 2020 milling dataset (ai4i2020.csv)
            cmapss/             # NASA C-MAPSS FD001 - FD004 txt files
        processed/             # Feature-engineered training/validation arrays
    notebooks/
        01_eda_ai4i.ipynb       # Exploratory analysis on milling sensor data
        02_failure_prediction.ipynb # Binary & multiclass failure modeling
        03_anomaly_detection.ipynb  # Isolation Forest unsupervised outlier detection
        04_nasa_rul_analysis.ipynb  # Turbofan degradation trends & RUL curves
        05_rul_modeling.ipynb       # RUL regression benchmarks & evaluation
    src/
        data_loader.py          # Data ingestion and verification
        preprocessing.py        # Pipelines, scalers, and encoders
        feature_engineering.py  # Domain indicators & rolling temporal features
        failure_model.py        # Binary & multi-type failure classifiers
        anomaly_model.py        # Isolation Forest anomaly scoring
        rul_model.py            # RUL regressors with capped targets
        evaluation.py           # Precision, Recall, F1, ROC-AUC, PR-AUC, NASA score
        explainability.py       # SHAP TreeExplainer & feature attribution
    models/                     # Serialized joblib pipelines & model artifacts
    api/
        main.py                 # FastAPI application
        schemas.py              # Pydantic request/response validation schemas
    dashboard/
        app.py                  # Main Streamlit dashboard application
        pages/                  # Multi-page analytics interface
    tests/                      # Pytest automated test suite
    reports/
        figures/                # Saved visualizations and metrics charts
    scripts/
        download_data.py        # Automated dataset downloader
    requirements.txt
    Dockerfile
    docker-compose.yml
    README.md
```

---

## Quickstart

```bash
# 1. Clone repository & install dependencies
pip install -r requirements.txt

# 2. Acquire real datasets
python scripts/download_data.py

# 3. Execute test suite
pytest tests/ -v
```
