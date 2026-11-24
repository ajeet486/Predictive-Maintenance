"""
Model Explainability Module using SHAP (Module 5)
Provides:
1. Exact SHAP TreeExplainer for tree ensembles (XGBoost / Random Forest)
2. Global feature importance and beeswarm summary plots
3. Local instance-level attribution with top risk drivers
4. Serialized background data and explainer for FastAPI / Streamlit integration
"""

import os
import sys
import json
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_ai4i

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


class FailureExplainer:
    """
    SHAP-based explainability engine for machine failure predictions.
    """

    def __init__(self, pipeline_path: Optional[str] = None):
        target_path = pipeline_path or os.path.join(MODELS_DIR, "failure_pipeline.joblib")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Failure pipeline artifact not found at '{target_path}'. Run failure_model.py first.")

        artifact = joblib.load(target_path)
        self.preprocessor = artifact["preprocessor"]
        self.model = artifact["best_model"]
        self.model_name = artifact["best_model_name"]
        self.feature_names = artifact["feature_names"]

        # Initialize TreeExplainer
        # For XGBClassifier or RandomForest, TreeExplainer is optimal
        self.explainer = shap.TreeExplainer(self.model)
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr = np.ravel(ev)
            self.expected_value = float(ev_arr[-1])
        else:
            self.expected_value = float(ev)

    def generate_global_explanations(self, sample_size: int = 500) -> Dict[str, float]:
        """
        Compute global SHAP values across representative sample and generate summary plots.
        """
        print(f"Generating global SHAP explanations using {sample_size} observations...")
        df = load_ai4i()
        X_raw = df.drop(columns=["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"])
        
        # Sample stratifying failures
        if sample_size < len(X_raw):
            sample_df = X_raw.sample(n=sample_size, random_state=42)
        else:
            sample_df = X_raw

        X_proc = self.preprocessor.transform(sample_df)
        shap_values = self.explainer.shap_values(X_proc)

        # Handle binary classification output shape
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_matrix = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        # 1. Summary Beeswarm Plot
        plt.figure(figsize=(11, 7))
        shap.summary_plot(shap_matrix, X_proc, feature_names=self.feature_names, show=False)
        plt.title(f"SHAP Beeswarm Summary Plot ({self.model_name})", fontsize=13)
        plt.tight_layout()
        summary_path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {summary_path}")

        # 2. Global Mean Absolute SHAP Bar Plot
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        importance_dict = dict(sorted(zip(self.feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=(10, 6))
        feats = list(importance_dict.keys())[::-1]
        scores = list(importance_dict.values())[::-1]
        ax.barh(feats, scores, color="#2b5c8f", edgecolor="black")
        ax.set_title("Global Mean |SHAP| Feature Importance (Failure Prediction)")
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Log-Odds)")
        plt.tight_layout()
        bar_path = os.path.join(FIGURES_DIR, "shap_global_bar.png")
        fig.savefig(bar_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {bar_path}")

        # Save importance scores to JSON
        json_path = os.path.join(REPORTS_DIR, "shap_feature_importance.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in importance_dict.items()}, f, indent=2)
        print(f"Saved: {json_path}")

        return importance_dict

    def explain_instance(self, raw_input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single machine prediction with decomposed risk drivers.

        Args:
            raw_input_df: Single-row or multi-row DataFrame with raw sensor telemetry.

        Returns:
            Dict[str, Any]: Dictionary detailing base value, predicted probability,
                            risk category, and top positive/negative contributing factors.
        """
        X_proc = self.preprocessor.transform(raw_input_df)
        pred_prob = float(self.model.predict_proba(X_proc)[0, 1])
        pred_class = int(self.model.predict(X_proc)[0])

        shap_vals = self.explainer.shap_values(X_proc)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            instance_shap = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            instance_shap = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
            instance_shap = shap_vals[0]
        else:
            instance_shap = shap_vals

        # Map contributions
        contributions = []
        for feat_name, shap_val, val in zip(self.feature_names, instance_shap, X_proc.iloc[0]):
            contributions.append({
                "feature": feat_name,
                "shap_value": float(shap_val),
                "scaled_value": float(val)
            })

        # Sort drivers
        top_risk_drivers = sorted([c for c in contributions if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
        top_safety_drivers = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda x: x["shap_value"])

        # Determine industrial risk level
        if pred_prob >= 0.70:
            risk_tier = "CRITICAL / HIGH RISK"
        elif pred_prob >= 0.35:
            risk_tier = "ELEVATED / WARNING"
        else:
            risk_tier = "NOMINAL / LOW RISK"

        return {
            "model_name": self.model_name,
            "failure_probability": pred_prob,
            "predicted_class": pred_class,
            "risk_tier": risk_tier,
            "base_value": self.expected_value,
            "top_risk_drivers": top_risk_drivers[:3],
            "top_safety_drivers": top_safety_drivers[:3],
            "all_contributions": contributions
        }

    def save_explainer(self):
        """
        Serialize explainer artifact to enable rapid retrieval in FastAPI/Streamlit.
        """
        artifact = {
            "expected_value": self.expected_value,
            "feature_names": self.feature_names,
            "model_name": self.model_name
        }
        path = os.path.join(MODELS_DIR, "shap_explainer_meta.joblib")
        joblib.dump(artifact, path)
        print(f"Saved SHAP explainer metadata to {path}")


def run_explainability():
    explainer = FailureExplainer()
    explainer.generate_global_explanations(sample_size=600)
    explainer.save_explainer()

    # Test sample explanation on a high-risk machine
    df = load_ai4i()
    failed_sample = df[df["machine_failure"] == 1].iloc[:1]
    explanation = explainer.explain_instance(failed_sample)
    print("\nSample Failure Case Explanation:")
    print(f"  Risk Tier: {explanation['risk_tier']}")
    print(f"  Failure Probability: {explanation['failure_probability']:.4f}")
    print("  Top Failure Risk Drivers:")
    for d in explanation["top_risk_drivers"]:
        print(f"    - {d['feature']}: SHAP impact = +{d['shap_value']:.3f}")


if __name__ == "__main__":
    run_explainability()
