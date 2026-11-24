"""
Failure Prediction Modeling Module (AI4I 2020 Dataset)
Implements:
1. Baseline Logistic Regression (with balanced class weights)
2. Random Forest Classifier (cost-sensitive ensemble)
3. XGBoost Classifier (gradient boosting with scale_pos_weight)
4. Multi-class Failure Mode Classifier (TWF, HDF, PWF, OSF, RNF)

Saves trained artifacts into models/ and evaluation reports into reports/.
"""


import os
import sys
import json
from typing import Dict, Any, Tuple
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_ai4i
from src.preprocessing import AI4IPreprocessor
from src.evaluation import evaluate_binary_classifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def map_failure_modes(df: pd.DataFrame) -> pd.Series:
    """
    Construct mutually exclusive multi-class target:
    0: Healthy / No Failure
    1: TWF (Tool Wear Failure)
    2: HDF (Heat Dissipation Failure)
    3: PWF (Power Failure)
    4: OSF (Overstrain Failure)
    5: RNF (Random Failure)
    """
    y_multiclass = pd.Series(0, index=df.index, dtype=int)
    
    # Priority assignment based on fault severity
    y_multiclass[df['rnf'] == 1] = 5
    y_multiclass[df['twf'] == 1] = 1
    y_multiclass[df['pwf'] == 1] = 3
    y_multiclass[df['hdf'] == 1] = 2
    y_multiclass[df['osf'] == 1] = 4
    return y_multiclass


FAILURE_MODE_NAMES = {
    0: "Normal / Healthy",
    1: "Tool Wear Failure (TWF)",
    2: "Heat Dissipation Failure (HDF)",
    3: "Power Failure (PWF)",
    4: "Overstrain Failure (OSF)",
    5: "Random Failure (RNF)"
}


class FailurePredictionPipeline:
    """
    Complete training and inference harness for AI4I Machine Failure Detection.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = AI4IPreprocessor(scale_features=True)
        self.models: Dict[str, Any] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.multiclass_model: Any = None
        self.metrics_summary: Dict[str, Any] = {}

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = load_ai4i()
        y_binary = df["machine_failure"].copy()
        y_multi = map_failure_modes(df)
        X = df.drop(columns=["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"])
        return X, y_binary, y_multi

    def train_and_evaluate(self) -> Dict[str, Any]:
        print("\n--- Training Failure Prediction Models on AI4I 2020 ---")
        X, y, y_multi = self.prepare_data()

        # Stratified train/test split to preserve failure ratio
        X_train_raw, X_test_raw, y_train, y_test, y_m_train, y_m_test = train_test_split(
            X, y, y_multi, test_size=0.20, random_state=self.random_state, stratify=y
        )

        print(f"Train samples: {len(X_train_raw)} (Failures: {y_train.sum()})")
        print(f"Test samples:  {len(X_test_raw)} (Failures: {y_test.sum()})")

        # Fit preprocessor on train data only (no data leakage)
        self.preprocessor.fit(X_train_raw)
        X_train = self.preprocessor.transform(X_train_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        # Calculate imbalance ratio for XGBoost scale_pos_weight
        num_neg = int((y_train == 0).sum())
        num_pos = int((y_train == 1).sum())
        pos_weight = float(num_neg / max(num_pos, 1))

        # Model Definitions
        candidate_models = {
            "Logistic_Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            )
        }

        results = {}
        predictions = {}

        for name, model in candidate_models.items():
            print(f"Fitting {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_binary_classifier(y_test.values, y_pred, y_prob, model_name=name)
            results[name] = metrics
            predictions[name] = {"pred": y_pred, "prob": y_prob}

            print(f"  [{name}] Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
                  f"F1: {metrics['f1_score']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f} | PR-AUC: {metrics['pr_auc']:.3f}")
            print(f"  False Negatives: {metrics['false_negatives']} / {metrics['true_positives'] + metrics['false_negatives']}")

        # Select Best Model based on F1-score & Recall balance
        best_name = max(results.keys(), key=lambda k: results[k]["f1_score"])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        print(f"\n-> Selected Best Binary Model: {best_name} (F1: {results[best_name]['f1_score']:.3f})")

        # Train Multi-Class Failure Type Model
        print("\n--- Training Multi-Class Failure Mode Classifier ---")
        self.multiclass_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1
        )
        self.multiclass_model.fit(X_train, y_m_train)
        y_m_pred = self.multiclass_model.predict(X_test)
        multi_report = classification_report(
            y_m_test, y_m_pred,
            labels=list(FAILURE_MODE_NAMES.keys()),
            target_names=list(FAILURE_MODE_NAMES.values()),
            output_dict=True,
            zero_division=0
        )
        results["MultiClass_Random_Forest"] = multi_report
        print("Multi-class classification macro F1:", multi_report["macro avg"]["f1-score"])

        self.metrics_summary = results

        # Generate Evaluation Plots
        self._plot_confusion_matrices(y_test.values, predictions)
        self._plot_metrics_comparison(results)

        # Save artifacts
        self.save_artifacts()

        return results

    def _plot_confusion_matrices(self, y_true: np.ndarray, predictions: dict):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for idx, (name, preds) in enumerate(predictions.items()):
            cm = confusion_matrix(y_true, preds["pred"])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Failure"],
                yticklabels=["Healthy", "Failure"],
                ax=axes[idx], cbar=False
            )
            axes[idx].set_title(f"Confusion Matrix: {name}")
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("True Label")

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    def _plot_metrics_comparison(self, results: dict):
        models = ["Logistic_Regression", "Random_Forest", "XGBoost"]
        metrics = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]

        data = []
        for m in models:
            for met in metrics:
                data.append({"Model": m, "Metric": met.upper().replace("_", "-"), "Score": results[m][met]})
        plot_df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax)
        ax.set_title("Failure Classifier Performance Comparison (Test Set)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        plt.legend(loc="lower right")

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "failure_model_comparison.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    def save_artifacts(self):
        pipeline_artifact = {
            "preprocessor": self.preprocessor,
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "all_models": self.models,
            "feature_names": self.preprocessor.feature_names_out_
        }
        pipe_path = os.path.join(MODELS_DIR, "failure_pipeline.joblib")
        joblib.dump(pipeline_artifact, pipe_path)
        print(f"Saved primary failure pipeline to {pipe_path}")

        # Multi-class artifact
        multi_artifact = {
            "preprocessor": self.preprocessor,
            "model": self.multiclass_model,
            "class_names": FAILURE_MODE_NAMES,
            "feature_names": self.preprocessor.feature_names_out_
        }
        multi_path = os.path.join(MODELS_DIR, "multiclass_failure_pipeline.joblib")
        joblib.dump(multi_artifact, multi_path)
        print(f"Saved multiclass pipeline to {multi_path}")

        # Metrics JSON
        json_path = os.path.join(REPORTS_DIR, "failure_model_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2)
        print(f"Saved evaluation metrics to {json_path}")


def train_failure_models():
    pipe = FailurePredictionPipeline(random_state=42)
    return pipe.train_and_evaluate()


if __name__ == "__main__":
    train_failure_models()
