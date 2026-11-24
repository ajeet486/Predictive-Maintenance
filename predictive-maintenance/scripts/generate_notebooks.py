"""
Notebook Generator Utility
Generates clean, professional Jupyter notebooks for the project:
- 01_eda_ai4i.ipynb
- 02_failure_prediction.ipynb
- 03_anomaly_detection.ipynb
- 04_nasa_rul_analysis.ipynb
- 05_rul_modeling.ipynb
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

def md_cell(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(code):
    lines = [l + "\n" for l in code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# 1. 01_eda_ai4i.ipynb
cells_01 = [
    md_cell("# 01 - Exploratory Data Analysis: AI4I 2020 Milling Telemetry\nThis notebook inspects the real AI4I 2020 Predictive Maintenance dataset from UCI."),
    code_cell("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))
from src.data_loader import load_ai4i, get_dataset_metadata

ai4i_df = load_ai4i()
print("Dataset shape:", ai4i_df.shape)
ai4i_df.head()"""),
    md_cell("## Data Integrity, Missing Values, and Types"),
    code_cell("""print("Missing values per column:\\n", ai4i_df.isnull().sum())
print("\\nData types:\\n", ai4i_df.dtypes)
print("\\nFailure class balance:\\n", ai4i_df['machine_failure'].value_counts(normalize=True))"""),
    md_cell("## Failure Mechanism Breakdown\nThe dataset records 5 distinct failure modes: TWF, HDF, PWF, OSF, and RNF."),
    code_cell("""failure_modes = ['twf', 'hdf', 'pwf', 'osf', 'rnf']
print(ai4i_df[failure_modes].sum())"""),
    md_cell("## Sensor Distributions and Physics Features"),
    code_cell("""ai4i_df['temp_diff_k'] = ai4i_df['process_temperature_k'] - ai4i_df['air_temperature_k']
ai4i_df['power_kw'] = ai4i_df['rotational_speed_rpm'] * ai4i_df['torque_nm'] * (2 * np.pi / 60) / 1000.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=ai4i_df, x='rotational_speed_rpm', y='torque_nm', hue='machine_failure', alpha=0.6, ax=axes[0])
axes[0].set_title("Torque vs. Rotational Speed")

sns.scatterplot(data=ai4i_df, x='temp_diff_k', y='rotational_speed_rpm', hue='hdf', palette={0: 'blue', 1: 'red'}, alpha=0.7, ax=axes[1])
axes[1].set_title("Temp Difference vs Speed (HDF Detection)")
plt.tight_layout()
plt.show()""")
]

# 2. 04_nasa_rul_analysis.ipynb
cells_04 = [
    md_cell("# 04 - NASA C-MAPSS Turbofan Engine Degradation Analysis\nExploratory analysis of flight cycle degradation trajectories in FD001."),
    code_cell("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))
from src.data_loader import load_cmapss

train_df, test_df, rul_df = load_cmapss('FD001')
print(f"Train FD001: {train_df.shape}, Test: {test_df.shape}, RUL ground truth: {rul_df.shape}")
train_df.head()"""),
    md_cell("## Engine Lifespans and Run-to-Failure Distributions"),
    code_cell("""engine_lifespans = train_df.groupby('unit_nr')['time_cycles'].max()
print(f"Engine lifetime statistics:\\nMin: {engine_lifespans.min()} | Max: {engine_lifespans.max()} | Mean: {engine_lifespans.mean():.1f}")

plt.figure(figsize=(9, 4))
sns.histplot(engine_lifespans, bins=20, kde=True, color='#2b5c8f')
plt.title("Distribution of Run-to-Failure Engine Lifespans (FD001)")
plt.xlabel("Total Cycles to Failure")
plt.show()"""),
    md_cell("## Sensor Variance & Degradation Tracking"),
    code_cell("""sensor_cols = [f's_{i}' for i in range(1, 22)]
stds = train_df[sensor_cols].std()
constant_sensors = stds[stds < 1e-4].index.tolist()
print("Constant sensors with near-zero variance (should be pruned):", constant_sensors)

# Plot key degrading sensors across 5 engines
sample_engines = [1, 2, 3, 4, 5]
plt.figure(figsize=(12, 6))
for eng in sample_engines:
    sub = train_df[train_df['unit_nr'] == eng]
    plt.plot(sub['time_cycles'], sub['s_2'], label=f'Engine #{eng}')
plt.title("Sensor 2 (LPC Outlet Temp) Degradation Curve Across Flight Cycles")
plt.xlabel("Operational Flight Cycles")
plt.ylabel("Sensor 2 Reading")
plt.legend()
plt.show()""")
]

# 2. 02_failure_prediction.ipynb
cells_02 = [
    md_cell("# 02 - Failure Prediction Modeling (AI4I 2020)\nBinary failure prediction (Healthy vs Failure) and multi-class failure mode classification."),
    code_cell("""import sys
import os
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from src.data_loader import load_ai4i
from src.failure_model import FailurePredictionPipeline

pipeline = FailurePredictionPipeline(random_state=42)
results = pipeline.train_and_evaluate()"""),
    md_cell("## Model Comparison Results"),
    code_cell("""comparison_rows = []
for m in ['Logistic_Regression', 'Random_Forest', 'XGBoost']:
    r = results[m]
    comparison_rows.append({
        'Model': m,
        'Precision': f"{r['precision']:.3f}",
        'Recall': f"{r['recall']:.3f}",
        'F1-Score': f"{r['f1_score']:.3f}",
        'ROC-AUC': f"{r['roc_auc']:.3f}",
        'PR-AUC': f"{r['pr_auc']:.3f}",
        'False Negatives': f"{r['false_negatives']} / 68"
    })
pd.DataFrame(comparison_rows)"""),
    md_cell("## Inspecting Saved Pipeline"),
    code_cell("""artifact = joblib.load('../models/failure_pipeline.joblib')
print("Best Model Selected:", artifact['best_model_name'])
print("Fitted Feature Names:", artifact['feature_names'])""")
]

# 3. 03_anomaly_detection.ipynb
cells_03 = [
    md_cell("# 03 - Unsupervised Anomaly Detection (Isolation Forest)\nDetects unusual sensor combinations without relying on failure labels."),
    code_cell("""import sys
import os
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from src.anomaly_model import AnomalyDetectionPipeline, train_anomaly_model

metrics = train_anomaly_model()
print("Anomaly Detection Evaluation Summary:")
for k, v in metrics.items():
    print(f"  {k}: {v}")"""),
    md_cell("## Inspecting Calibrated Anomaly Scores"),
    code_cell("""pipe = joblib.load('../models/anomaly_pipeline.joblib')
print("Model contamination:", pipe['contamination'])
print("Decision threshold:", pipe['threshold'])
print("Fitted features:", pipe['feature_names'])""")
]

# 5. 05_rul_modeling.ipynb
cells_05 = [
    md_cell("# 05 - Remaining Useful Life (RUL) Modeling (NASA C-MAPSS FD001)\nEngine-wise regression modeling comparing Random Forest and XGBoost with NASA PHM scoring."),
    code_cell("""import sys
import os
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from src.rul_model import RULPredictionPipeline

pipe = RULPredictionPipeline(subset='FD001', max_rul=125, random_state=42)
results = pipe.train_and_evaluate()"""),
    md_cell("## Model Comparison on Blind Test Engines (100 Turbofans)"),
    code_cell("""rows = []
for m in ['Random_Forest', 'XGBoost']:
    t = results[m]['test']
    v = results[m]['validation']
    rows.append({
        'Model': m,
        'Val RMSE': f"{v['rmse']:.2f}",
        'Test RMSE': f"{t['rmse']:.2f}",
        'Test MAE': f"{t['mae']:.2f}",
        'Test R2': f"{t['r2_score']:.3f}",
        'NASA Score': f"{t['nasa_total_score']:.1f}"
    })
pd.DataFrame(rows)"""),
    md_cell("## Inspecting Serialized RUL Pipeline"),
    code_cell("""artifact = joblib.load('../models/rul_pipeline.joblib')
print("Best RUL Model:", artifact['best_model_name'])
print("Total Features Engineered:", len(artifact['feature_names']))""")
]

create_notebook(cells_01, os.path.join(NB_DIR, "01_eda_ai4i.ipynb"))
create_notebook(cells_02, os.path.join(NB_DIR, "02_failure_prediction.ipynb"))
create_notebook(cells_03, os.path.join(NB_DIR, "03_anomaly_detection.ipynb"))
create_notebook(cells_04, os.path.join(NB_DIR, "04_nasa_rul_analysis.ipynb"))
create_notebook(cells_05, os.path.join(NB_DIR, "05_rul_modeling.ipynb"))
