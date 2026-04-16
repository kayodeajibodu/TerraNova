"""
Model Training & Evaluation — with MLflow Integration
Trains Linear Regression, Random Forest, and XGBoost.
Every run is tracked in MLflow (metrics, params, artifacts, model registry).
The best model is promoted to the 'champion' alias in the registry
and also saved locally as models/best_model.pkl for API fallback.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

MODELS_DIR       = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE        = 0.20
RANDOM_SEED      = 42
EXPERIMENT_NAME  = "terra_nova_disaster_cost_forecasting"
REGISTERED_NAME  = "terra_nova_cost_model"      # name in MLflow Model Registry
CHAMPION_ALIAS   = "champion"                   # alias promoted on best model


class ModelTrainer:
    """
    Trains all three model families, logs every run to MLflow,
    registers the best model, and saves a local pkl fallback.

    Usage
    -----
    trainer = ModelTrainer(tracking_uri="http://localhost:5000")
    results = trainer.train(X, y)
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = EXPERIMENT_NAME,
    ):
        self.models_dir      = models_dir
        self.tracking_uri    = tracking_uri
        self.experiment_name = experiment_name
        self.results_: dict[str, dict]  = {}
        self.best_model_name_: str      = ""
        self.best_pipeline_: Any        = None
        self.best_run_id_: Optional[str] = None

        # Configure MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, dict]:
        """
        Train all models, log to MLflow, promote champion, save pkl.

        Returns
        -------
        dict mapping model name → evaluation metrics
        """
        logger.info("Splitting data: %.0f%% train / %.0f%% test",
                    (1 - TEST_SIZE) * 100, TEST_SIZE * 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        candidates = self._build_candidates()
        trained_pipelines: dict = {}

        for name, pipeline in candidates.items():
            logger.info("Training: %s ...", name)
            run_id = self._train_and_log(
                name, pipeline, X, X_train, X_test, y_train, y_test
            )
            self.results_[name]["run_id"] = run_id
            trained_pipelines[name] = pipeline
            logger.info(
                "  R²=%.4f | RMSE=%.4f | MAE=%.4f | RMSE($)=$%.0f | run_id=%s",
                self.results_[name]["r2"],
                self.results_[name]["rmse"],
                self.results_[name]["mae"],
                self.results_[name]["rmse_dollars"],
                run_id,
            )

        self._promote_champion()
        self._save_local_pkl(trained_pipelines)
        self._save_results_csv()

        return self.results_

    def load_best_model(self) -> Any:
        """
        Load model — tries MLflow registry first, falls back to local pkl.
        """
        # Try MLflow registry champion
        try:
            model_uri = f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}"
            logger.info("Loading champion model from MLflow registry: %s", model_uri)
            return mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            logger.warning("MLflow registry load failed (%s), using local pkl.", exc)

        # Fallback: local pickle
        path = self.models_dir / "best_model.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path} and MLflow registry unavailable. Run train() first."
            )
          #  f = open(path, "rb") # just to check if file is readable
           # f.close() # just to check if file is readable
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        if self.best_pipeline_ is None:
            raise RuntimeError("No trained model. Call train() first.")
        model = self.best_pipeline_.named_steps.get("model")
        if not hasattr(model, "feature_importances_"):
            logger.warning("Best model has no feature_importances_ attribute.")
            return pd.DataFrame()
        return (
            pd.DataFrame({
                "feature":    feature_names,
                "importance": model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # MLflow training loop
    # ------------------------------------------------------------------

    def _train_and_log(
        self,
        name: str,
        pipeline: Pipeline,
        X_full: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> str:
        """Train one model inside an MLflow run. Returns run_id."""

        with mlflow.start_run(run_name=name) as run:
            # ── Tags ────────────────────────────────────────────────
            mlflow.set_tags({
                "model_family": name,
                "dataset":      "fema_open_data",
                "target":       "log1p_total_obligated_amount",
                "framework":    "scikit-learn" if name != "xgboost" else "xgboost",
            })

            # ── Params ──────────────────────────────────────────────
            model_step = pipeline.named_steps["model"]
            params     = model_step.get_params()
            params["test_size"]    = TEST_SIZE
            params["random_seed"]  = RANDOM_SEED
            params["n_features"]   = X_full.shape[1]
            params["n_train_rows"] = len(X_train)
            mlflow.log_params(params)

            # ── Fit ─────────────────────────────────────────────────
            pipeline.fit(X_train, y_train)

            # ── Evaluate ────────────────────────────────────────────
            metrics = self._evaluate(pipeline, X_test, y_test, name)

            cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
            metrics["cv_r2_mean"] = float(cv_r2.mean())
            metrics["cv_r2_std"]  = float(cv_r2.std())

            self.results_[name] = metrics

            # ── Log metrics ─────────────────────────────────────────
            mlflow.log_metrics({
                "r2":           metrics["r2"],
                "rmse":         metrics["rmse"],
                "mae":          metrics["mae"],
                "rmse_dollars": metrics["rmse_dollars"],
                "cv_r2_mean":   metrics["cv_r2_mean"],
                "cv_r2_std":    metrics["cv_r2_std"],
            })

            # ── Log model to registry ────────────────────────────────
            if name == "xgboost":
                mlflow.xgboost.log_model(
                    xgb_model=model_step,
                    artifact_path="model",
                    registered_model_name=REGISTERED_NAME,
                    input_example=X_test.head(3),
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name=REGISTERED_NAME,
                    input_example=X_test.head(3),
                )

            # ── Log feature importance artifact (tree-based models only) ──
            model_obj = pipeline.named_steps["model"]
            if hasattr(model_obj, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature":    list(X_full.columns),
                    "importance": model_obj.feature_importances_,
                }).sort_values("importance", ascending=False)
                fi_path = self.models_dir / f"feature_importance_{name}.csv"
                fi.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path), artifact_path="feature_importance")

            return run.info.run_id

    # ------------------------------------------------------------------
    # Registry promotion
    # ------------------------------------------------------------------

    def _promote_champion(self) -> None:
        """
        Find the run with the highest R² across all trained models,
        set it as the 'champion' alias in the MLflow Model Registry.
        """
        best_name = max(self.results_, key=lambda k: self.results_[k]["r2"])
        self.best_model_name_ = best_name
        best_run_id           = self.results_[best_name]["run_id"]
        self.best_run_id_     = best_run_id

        # Find the model version that was logged in this run
        versions = self.client.search_model_versions(
            f"name='{REGISTERED_NAME}'"
        )
        target_version = None
        for v in versions:
            if v.run_id == best_run_id:
                target_version = v.version
                break

        if target_version is None:
            logger.warning("Could not find registry version for run %s", best_run_id)
            return

        # Set the champion alias
        self.client.set_registered_model_alias(
            name=REGISTERED_NAME,
            alias=CHAMPION_ALIAS,
            version=target_version,
        )

        # Add descriptive tags to the champion version
        self.client.set_model_version_tag(
            name=REGISTERED_NAME,
            version=target_version,
            key="r2",
            value=str(round(self.results_[best_name]["r2"], 4)),
        )
        self.client.set_model_version_tag(
            name=REGISTERED_NAME,
            version=target_version,
            key="rmse_dollars",
            value=str(round(self.results_[best_name]["rmse_dollars"], 0)),
        )

        logger.info(
            "Champion promoted: %s v%s (R²=%.4f) → alias='%s'",
            REGISTERED_NAME, target_version,
            self.results_[best_name]["r2"],
            CHAMPION_ALIAS,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_candidates(self) -> dict[str, Pipeline]:
        return {
            "linear_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  LinearRegression()),
            ]),
            "random_forest": Pipeline([
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=4,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                )),
            ]),
            "xgboost": Pipeline([
                ("model", XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=RANDOM_SEED,
                    verbosity=0,
                )),
            ]),
        }

    def _evaluate(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        name: str,
    ) -> dict:
        y_pred         = pipeline.predict(X_test)
        y_test_dollars = np.expm1(y_test)
        y_pred_dollars = np.expm1(y_pred)
        return {
            "model":        name,
            "r2":           float(r2_score(y_test, y_pred)),
            "rmse":         float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae":          float(mean_absolute_error(y_test, y_pred)),
            "rmse_dollars": float(np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))),
        }

    def _save_local_pkl(self, trained_pipelines: dict) -> None:
        """Persist the best trained pipeline as a local pkl fallback for the API."""
        best_name = self.best_model_name_
        pipeline  = trained_pipelines.get(best_name)
        if pipeline is None:
            logger.warning("No pipeline found for '%s' — skipping pkl save.", best_name)
            return
        self.best_pipeline_ = pipeline
        path = self.models_dir / "best_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("best_model.pkl saved: %s -> %s", best_name, path)

    def _save_results_csv(self) -> None:
        df = pd.DataFrame(self.results_.values())
        df.to_csv(self.models_dir / "evaluation_results.csv", index=False)
        logger.info("Evaluation results → models/evaluation_results.csv")