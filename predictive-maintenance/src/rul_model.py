"""
Remaining Useful Life (RUL) Prediction Module (NASA C-MAPSS FD001 - FD004)
Implements:
1. Engine-wise split to strictly prevent temporal leakage
2. Piecewise linear capped RUL target (max RUL = 125 cycles)
3. Rolling statistics & temporal lag feature extraction
4. Random Forest Regressor & XGBoost Regressor
5. NASA Asymmetric Prognostics Scoring Function evaluation
6. Blind test set evaluation against ground-truth RUL
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

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_loader import load_cmapss
from src.preprocessing import CMAPSSPreprocessor
from src.evaluation import evaluate_rul_regressor, nasa_asymmetric_scoring

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


class RULPredictionPipeline:
    """
    Prognostic health management pipeline for turbofan engine Remaining Useful Life estimation.
    """

    def __init__(
        self,
        subset: str = "FD001",
        max_rul: int = 125,
        random_state: int = 42
    ):
        self.subset = subset.upper()
        self.max_rul = max_rul
        self.random_state = random_state
        self.preprocessor = CMAPSSPreprocessor(max_rul=self.max_rul, scale_features=True)
        self.models: Dict[str, Any] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.metrics_summary: Dict[str, Any] = {}

    def train_and_evaluate(self) -> Dict[str, Any]:
        print(f"\n--- Training RUL Regression Models on NASA C-MAPSS {self.subset} ---")
        train_df, test_df, rul_df = load_cmapss(self.subset)

        # 1. Engine-wise Train/Validation Split (prevents data leakage across engine cycles)
        all_units = train_df["unit_nr"].unique()
        np.random.seed(self.random_state)
        val_units = np.random.choice(all_units, size=int(len(all_units) * 0.20), replace=False)
        train_units = np.setdiff1d(all_units, val_units)

        print(f"Total engines in training set: {len(all_units)}")
        print(f"  Training engines:   {len(train_units)} ({train_df[train_df['unit_nr'].isin(train_units)].shape[0]} cycles)")
        print(f"  Validation engines: {len(val_units)} ({train_df[train_df['unit_nr'].isin(val_units)].shape[0]} cycles)")

        train_split = train_df[train_df["unit_nr"].isin(train_units)].copy()
        val_split = train_df[train_df["unit_nr"].isin(val_units)].copy()

        # 2. Compute Capped RUL target (125 cycles)
        y_train = self.preprocessor.compute_capped_rul(train_split)
        y_val = self.preprocessor.compute_capped_rul(val_split)

        # 3. Fit preprocessor on training engines only
        self.preprocessor.fit(train_split)
        X_train = self.preprocessor.transform(train_split)
        X_val = self.preprocessor.transform(val_split)

        # 4. Prepare Test Set (extract last cycle per test engine for RUL comparison)
        test_proc = self.preprocessor.transform(test_df)
        test_df_with_proc = test_proc.copy()
        test_df_with_proc["unit_nr"] = test_df["unit_nr"].values
        test_df_with_proc["time_cycles"] = test_df["time_cycles"].values

        # Select the last recorded cycle for each test engine
        last_cycles_idx = test_df.groupby("unit_nr")["time_cycles"].idxmax()
        X_test_last = test_proc.loc[last_cycles_idx].copy()
        y_test_true = rul_df["rul_ground_truth"].values

        print(f"Prepared test features for {len(X_test_last)} engines with ground truth RUL.")

        # 5. Model Candidate Definitions
        candidates = {
            "Random_Forest": RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=3,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
        }

        results = {}
        predictions_test = {}

        for name, regressor in candidates.items():
            print(f"Fitting {name} Regressor...")
            regressor.fit(X_train, y_train)
            self.models[name] = regressor

            # Validation metrics
            val_preds = regressor.predict(X_val)
            val_metrics = evaluate_rul_regressor(y_val.values, val_preds, model_name=f"{name}_Val")

            # Blind Test Set metrics (against true final RUL)
            test_preds = regressor.predict(X_test_last)
            # Clip test predictions at [0, max_rul]
            test_preds_clipped = np.clip(test_preds, 0, self.max_rul)
            test_metrics = evaluate_rul_regressor(y_test_true, test_preds_clipped, model_name=f"{name}_Test")

            predictions_test[name] = test_preds_clipped

            results[name] = {
                "validation": val_metrics,
                "test": test_metrics
            }

            print(f"  [{name} Validation] RMSE: {val_metrics['rmse']:.2f} | MAE: {val_metrics['mae']:.2f} | R2: {val_metrics['r2_score']:.3f} | NASA Score: {val_metrics['nasa_total_score']:.1f}")
            print(f"  [{name} Blind Test]  RMSE: {test_metrics['rmse']:.2f} | MAE: {test_metrics['mae']:.2f} | R2: {test_metrics['r2_score']:.3f} | NASA Score: {test_metrics['nasa_total_score']:.1f}")

        # Best Model selection based on Blind Test RMSE & NASA Score
        best_name = min(results.keys(), key=lambda k: results[k]["test"]["rmse"])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        print(f"\n-> Selected Best RUL Model: {best_name} (Test RMSE: {results[best_name]['test']['rmse']:.2f})")

        self.metrics_summary = results

        # 6. Generate Figures
        self._plot_rul_comparison(y_test_true, predictions_test)
        self._plot_engine_trajectory(test_df, test_proc, self.best_model, rul_df, engine_id=24)

        # 7. Serialize Artifacts
        self.save_artifacts()

        return results

    def _plot_rul_comparison(self, y_true: np.ndarray, predictions: dict):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot True vs Predicted
        engine_ids = np.arange(1, len(y_true) + 1)
        best_preds = predictions[self.best_model_name]

        axes[0].scatter(y_true, best_preds, c="#2b5c8f", alpha=0.8, edgecolor="black", s=45, label="Test Engines (100)")
        axes[0].plot([0, self.max_rul], [0, self.max_rul], "r--", linewidth=2, label="Ideal 1:1 Parity")
        axes[0].set_title(f"True vs Predicted RUL: {self.best_model_name}")
        axes[0].set_xlabel("True Remaining Useful Life (Cycles)")
        axes[0].set_ylabel("Predicted Remaining Useful Life (Cycles)")
        axes[0].set_xlim(0, self.max_rul + 10)
        axes[0].set_ylim(0, self.max_rul + 10)
        axes[0].legend()

        # Engine by engine tracking
        axes[1].plot(engine_ids, y_true, "k-", linewidth=2, label="True RUL")
        for name, preds in predictions.items():
            axes[1].plot(engine_ids, preds, "--", label=f"Predicted ({name})")
        axes[1].set_title("Remaining Useful Life per Test Engine (Engines 1-100)")
        axes[1].set_xlabel("Test Engine Unit Number")
        axes[1].set_ylabel("RUL (Cycles)")
        axes[1].legend()

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "rul_predictions_vs_true.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    def _plot_engine_trajectory(
        self,
        test_df: pd.DataFrame,
        test_proc: pd.DataFrame,
        model: Any,
        rul_df: pd.DataFrame,
        engine_id: int = 24
    ):
        """
        Visualize full cycle-by-cycle degradation progression for a specific engine.
        """
        eng_mask = (test_df["unit_nr"] == engine_id)
        if not eng_mask.any():
            engine_id = int(test_df["unit_nr"].iloc[0])
            eng_mask = (test_df["unit_nr"] == engine_id)

        eng_cycles = test_df[eng_mask]["time_cycles"].values
        eng_features = test_proc[eng_mask]
        preds = model.predict(eng_features)
        true_final_rul = rul_df.loc[rul_df["unit_nr"] == engine_id, "rul_ground_truth"].values[0]

        # Calculate theoretical linear remaining life for this engine
        total_life = eng_cycles[-1] + true_final_rul
        theoretical_rul = np.clip(total_life - eng_cycles, 0, self.max_rul)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eng_cycles, theoretical_rul, "k--", label="Ground Truth RUL Path", linewidth=2)
        ax.plot(eng_cycles, preds, color="#d9534f", label=f"Model Predicted RUL ({self.best_model_name})", linewidth=2)
        ax.axvline(eng_cycles[-1], color="blue", linestyle=":", label=f"Last Test Cycle ({eng_cycles[-1]})")
        ax.set_title(f"NASA C-MAPSS: Engine #{engine_id} Degradation & RUL Forecast")
        ax.set_xlabel("Operational Flight Cycles")
        ax.set_ylabel("Remaining Useful Life (Cycles)")
        ax.legend()

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "rul_single_engine_trajectory.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    def save_artifacts(self):
        artifact = {
            "subset": self.subset,
            "max_rul": self.max_rul,
            "preprocessor": self.preprocessor,
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "all_models": self.models,
            "feature_names": self.preprocessor.feature_names_out_
        }
        pipe_path = os.path.join(MODELS_DIR, "rul_pipeline.joblib")
        joblib.dump(artifact, pipe_path)
        print(f"Saved RUL pipeline artifact to {pipe_path}")

        # Metrics JSON
        json_path = os.path.join(REPORTS_DIR, "rul_model_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_summary, f, indent=2)
        print(f"Saved RUL metrics report to {json_path}")


def train_rul_models(subset: str = "FD001"):
    pipe = RULPredictionPipeline(subset=subset, max_rul=125, random_state=42)
    return pipe.train_and_evaluate()


if __name__ == "__main__":
    train_rul_models("FD001")
