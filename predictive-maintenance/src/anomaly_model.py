"""
Anomaly Detection Module (Module 3)
Implements unsupervised anomaly detection using Isolation Forest:
- Trains on nominal (normal operating) sensor patterns
- Computes calibrated continuous anomaly scores [0, 1]
- Distinguishes sensor-level anomalies from confirmed failures
- Compares detected anomalies against ground-truth failure labels
- Serializes pipeline into models/anomaly_pipeline.joblib
"""

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_ai4i
from src.preprocessing import AI4IPreprocessor

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


class AnomalyDetectionPipeline:
    """
    Industrial Anomaly Detection using Isolation Forest.
    Trained strictly on nominal (healthy) machine states to capture baseline dynamics.
    """

    def __init__(self, contamination: float = 0.04, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.preprocessor = AI4IPreprocessor(scale_features=True)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_names: list = []
        self.threshold: float = 0.0

    def fit(self, X_train: pd.DataFrame, normal_mask: Optional[pd.Series] = None):
        """
        Fit preprocessor and Isolation Forest on nominal operating data.
        """
        if normal_mask is not None:
            nominal_X = X_train[normal_mask].copy()
        else:
            nominal_X = X_train.copy()

        self.preprocessor.fit(nominal_X)
        X_proc = self.preprocessor.transform(nominal_X)
        self.feature_names = self.preprocessor.feature_names_out_

        print(f"Training Isolation Forest on {len(X_proc)} nominal operating records...")
        self.model.fit(X_proc)

        # Compute decision threshold for calibrated anomaly score
        raw_scores = self.model.score_samples(X_proc)
        self.threshold = float(np.percentile(raw_scores, self.contamination * 100))
        print(f"Calibrated nominal anomaly threshold: {self.threshold:.4f}")
        return self

    def score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly predictions and normalized anomaly scores [0, 1].

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - is_anomaly: 1 if anomalous, 0 if normal
            - anomaly_score: float in [0, 1] where 1.0 indicates extreme anomaly
        """
        X_proc = self.preprocessor.transform(X)
        raw_scores = self.model.score_samples(X_proc)  # Lower is more abnormal

        # Scikit-learn: -1 = anomaly, 1 = normal
        raw_preds = self.model.predict(X_proc)
        is_anomaly = (raw_preds == -1).astype(int)

        # Calibrate score to [0, 1] range where higher = more anomalous
        # Use sigmoid-like linear normalization around threshold
        min_s, max_s = -0.75, -0.35
        norm_score = np.clip((max_s - raw_scores) / (max_s - min_s), 0.0, 1.0)

        return is_anomaly, norm_score

    def evaluate_against_known_failures(
        self,
        X_eval: pd.DataFrame,
        y_failures: pd.Series
    ) -> Dict[str, Any]:
        """
        Compare unsupervised anomalies with known ground-truth failure records.
        Crucial insight: Anomalies != Failures. Anomalies identify operational outliers,
        many of which precede or coincide with failures, while others represent peak loads.
        """
        is_anomaly, scores = self.score(X_eval)
        df_res = pd.DataFrame({
            "is_anomaly": is_anomaly,
            "anomaly_score": scores,
            "actual_failure": y_failures.values
        })

        total_obs = len(df_res)
        total_anomalies = int(is_anomaly.sum())
        total_failures = int(y_failures.sum())

        # Failures that were successfully flagged as anomalies (Recall on failures)
        failures_detected = int(df_res[(df_res["actual_failure"] == 1) & (df_res["is_anomaly"] == 1)].shape[0])
        failure_detection_rate = float(failures_detected / max(total_failures, 1))

        # Anomalies that correspond to confirmed failures (Precision)
        anomaly_precision = float(failures_detected / max(total_anomalies, 1))

        # Mean anomaly score for healthy vs failure
        mean_score_healthy = float(df_res[df_res["actual_failure"] == 0]["anomaly_score"].mean())
        mean_score_failure = float(df_res[df_res["actual_failure"] == 1]["anomaly_score"].mean())

        metrics = {
            "total_observations": total_obs,
            "detected_anomalies": total_anomalies,
            "anomaly_rate": float(total_anomalies / total_obs),
            "actual_failures": total_failures,
            "failures_captured_as_anomalies": failures_detected,
            "failure_capture_rate": failure_detection_rate,
            "anomaly_to_failure_precision": anomaly_precision,
            "mean_anomaly_score_healthy": mean_score_healthy,
            "mean_anomaly_score_failure": mean_score_failure
        }

        # Save Plots
        self._plot_anomaly_distribution(df_res)
        self._plot_scatter_anomalies(X_eval, df_res)

        return metrics

    def _plot_anomaly_distribution(self, df_res: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Anomaly score distribution
        sns.kdeplot(df_res[df_res["actual_failure"] == 0]["anomaly_score"], ax=axes[0], label="Healthy Operations", color="#2b5c8f", fill=True)
        sns.kdeplot(df_res[df_res["actual_failure"] == 1]["anomaly_score"], ax=axes[0], label="Known Failures", color="#d9534f", fill=True)
        axes[0].set_title("Isolation Forest Anomaly Score Distribution")
        axes[0].set_xlabel("Anomaly Score [0 = Nominal, 1 = Severe Outlier]")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # Breakdown of anomalies vs failures
        contingency = pd.crosstab(
            df_res["is_anomaly"].map({0: "Normal", 1: "Anomalous"}),
            df_res["actual_failure"].map({0: "Healthy", 1: "Failed"})
        )
        sns.heatmap(contingency, annot=True, fmt="d", cmap="Purples", cbar=False, ax=axes[1])
        axes[1].set_title("Contingency Matrix: Anomalies vs Known Failures")
        axes[1].set_xlabel("Actual Machine Status")
        axes[1].set_ylabel("Isolation Forest Status")

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "anomaly_distribution.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    def _plot_scatter_anomalies(self, X_eval: pd.DataFrame, df_res: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        X_df = X_eval.copy()
        X_df["is_anomaly"] = df_res["is_anomaly"].values
        X_df["actual_failure"] = df_res["actual_failure"].values

        normal_pts = X_df[(X_df["is_anomaly"] == 0) & (X_df["actual_failure"] == 0)]
        anomaly_only = X_df[(X_df["is_anomaly"] == 1) & (X_df["actual_failure"] == 0)]
        failure_pts = X_df[X_df["actual_failure"] == 1]

        ax.scatter(normal_pts["rotational_speed_rpm"], normal_pts["torque_nm"], c="#2b5c8f", alpha=0.2, s=15, label="Nominal (Normal)")
        ax.scatter(anomaly_only["rotational_speed_rpm"], anomaly_only["torque_nm"], c="#f0ad4e", alpha=0.8, s=35, label="Anomalous (Non-Failure)")
        ax.scatter(failure_pts["rotational_speed_rpm"], failure_pts["torque_nm"], c="#d9534f", alpha=0.9, s=50, marker="x", label="Known Failures")

        ax.set_title("Operational Domain: Normal vs Outlier Anomalies vs Failures")
        ax.set_xlabel("Rotational Speed [rpm]")
        ax.set_ylabel("Torque [Nm]")
        ax.legend()

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "anomaly_vs_failures.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    def save_artifact(self):
        artifact = {
            "preprocessor": self.preprocessor,
            "model": self.model,
            "contamination": self.contamination,
            "threshold": self.threshold,
            "feature_names": self.feature_names
        }
        pipe_path = os.path.join(MODELS_DIR, "anomaly_pipeline.joblib")
        joblib.dump(artifact, pipe_path)
        print(f"Saved anomaly detection pipeline to {pipe_path}")


def train_anomaly_model():
    print("\n--- Training Unsupervised Anomaly Detection Model ---")
    df = load_ai4i()
    X = df.drop(columns=["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"])
    y_fail = df["machine_failure"]

    # Filter purely normal cycles for training
    normal_mask = (y_fail == 0)

    pipeline = AnomalyDetectionPipeline(contamination=0.04, random_state=42)
    pipeline.fit(X, normal_mask=normal_mask)

    # Evaluate across entire dataset
    metrics = pipeline.evaluate_against_known_failures(X, y_fail)
    print("\nAnomaly Detection Evaluation:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    pipeline.save_artifact()

    # Save metrics JSON
    json_path = os.path.join(REPORTS_DIR, "anomaly_detection_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved anomaly detection report to {json_path}")
    return metrics


if __name__ == "__main__":
    train_anomaly_model()
