# Predictive Maintenance

This project uses machine sensor data to explore three practical maintenance questions:

- Is a machine likely to fail, and which failure mode is most likely?
- Is its current sensor pattern unusual?
- How many operating cycles might a turbofan engine have left?

It includes data preparation and modeling code, saved model files, evaluation reports, and a FastAPI service for making predictions. The models are experiments built with public datasets, not a substitute for validated industrial safety systems.

## What’s in the project

- **Failure prediction:** Binary failure and failure-mode classification using the AI4I 2020 dataset.
- **Anomaly detection:** An Isolation Forest trained on nominal AI4I operating records. An anomaly is an unusual sensor pattern; it does not necessarily mean a failure.
- **Remaining useful life (RUL):** Regression on NASA C-MAPSS turbofan data. The current saved RUL model is for subset FD001 and caps predictions at 125 cycles.
- **Explainability:** SHAP-based drivers for failure predictions.
- **API:** FastAPI endpoints for health, model information, and the three prediction types.
- **Analysis:** Notebooks, reports, and plots in `notebooks/` and `reports/`.

## Get started

Use Python 3.10 or later. From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository includes datasets under `data/raw/` and saved model pipelines under `models/`. If you need to download the datasets again, run:

```powershell
python scripts/download_data.py
```

## Run the API

Start the service from the project directory:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/docs> for interactive API documentation. The endpoints are:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service and model availability |
| `GET` | `/models` | See available models and their input features |
| `POST` | `/predict/failure` | Estimate failure risk and return SHAP drivers |
| `POST` | `/predict/anomaly` | Score a sensor record for anomalous behavior |
| `POST` | `/predict/rul` | Estimate turbofan remaining useful life |

For example, send this JSON to `POST /predict/failure`:

```json
{
  "type": "M",
  "air_temperature_k": 298.5,
  "process_temperature_k": 308.8,
  "rotational_speed_rpm": 1450,
  "torque_nm": 45.2,
  "tool_wear_min": 120
}
```

The anomaly endpoint accepts the same sensor fields. For RUL predictions, provide the engine operating settings and sensor values as a `sensors` object with keys such as `s_2`, `s_3`, and `s_4`. See the schema in the API docs for the full request format.

## Train models and run tests

Training scripts read the datasets from `data/raw/`, save model artifacts in `models/`, and write evaluation results and plots to `reports/`.

```powershell
python -m src.failure_model
python -m src.anomaly_model
python -m src.rul_model
pytest tests/ -v
```

The RUL training command uses FD001 by default. To train on another supported subset, call `train_rul_models("FD002")` (or FD003/FD004) from Python.

## Data sources

- **AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository, 10,000 records, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used for failure classification and anomaly detection.
- **NASA C-MAPSS Turbofan Engine Degradation dataset** — NASA Prognostics Center of Excellence. The project supports subsets FD001–FD004 for RUL work. See NASA’s [Prognostics Center of Excellence data repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) for dataset details and usage terms.

## Project layout

```text
api/          FastAPI application and request/response schemas
data/         Raw and processed datasets
models/       Saved model pipelines
notebooks/    Exploratory analysis and modeling notebooks
reports/      Metrics, summaries, and generated figures
scripts/      Dataset download and analysis utilities
src/          Data loading, preprocessing, models, and evaluation
tests/        Automated tests
```
