"""
Model Training & Evaluation
Trains Linear Regression (baseline), Random Forest, and XGBoost.
Evaluates using R², RMSE, and MAE on a held-out test set.
Persists the best model for API serving.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE   = 0.20
RANDOM_SEED = 42


class ModelTrainer:
    """
    Trains and evaluates all three model families.
    Saves the best-performing model to disk.

    Usage
    -----
    trainer = ModelTrainer()
    results = trainer.train(X, y)
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.results_: dict[str, dict] = {}
        self.best_model_name_: str     = ""
        self.best_pipeline_: Any       = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, dict]:
        """
        Train all models, evaluate, select the best, and save it.

        Returns
        -------
        dict mapping model name → evaluation metrics dict
        """
        logger.info("Splitting data: %.0f%% train / %.0f%% test", (1 - TEST_SIZE) * 100, TEST_SIZE * 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        candidates = self._build_candidates()

        for name, pipeline in candidates.items():
            logger.info("Training: %s …", name)
            pipeline.fit(X_train, y_train)

            metrics       = self._evaluate(pipeline, X_test, y_test, name)
            cv_r2         = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
            metrics["cv_r2_mean"] = float(cv_r2.mean())
            metrics["cv_r2_std"]  = float(cv_r2.std())

            self.results_[name] = metrics
            logger.info(
                "  R²=%.4f | RMSE=%.2f | MAE=%.2f | CV-R²=%.4f ± %.4f",
                metrics["r2"], metrics["rmse"], metrics["mae"],
                metrics["cv_r2_mean"], metrics["cv_r2_std"],
            )

        self._select_and_save_best(candidates)
        self._save_results()

        return self.results_

    def load_best_model(self) -> Any:
        """Load the persisted best model pipeline from disk."""
        path = self.models_dir / "best_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Run train() first.")
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Private helpers
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
                    early_stopping_rounds=None,
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
        y_pred = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))

        # Dollar-space RMSE (inverse log1p)
        y_test_dollars = np.expm1(y_test)
        y_pred_dollars = np.expm1(y_pred)
        rmse_dollars   = float(np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars)))

        return {
            "model":        name,
            "r2":           r2,
            "rmse":         rmse,
            "mae":          mae,
            "rmse_dollars": rmse_dollars,
        }

    def _select_and_save_best(self, candidates: dict[str, Pipeline]) -> None:
        best_name = max(self.results_, key=lambda k: self.results_[k]["r2"])
        self.best_model_name_ = best_name
        self.best_pipeline_   = candidates[best_name]

        path = self.models_dir / "best_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.best_pipeline_, f)

        logger.info("Best model: %s (R²=%.4f) → saved to %s", best_name, self.results_[best_name]["r2"], path)

    def _save_results(self) -> None:
        df = pd.DataFrame(self.results_.values())
        df.to_csv(self.models_dir / "evaluation_results.csv", index=False)
        logger.info("Evaluation results → models/evaluation_results.csv")

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Extract feature importances from the best model (if tree-based).
        Returns a sorted DataFrame for dashboard display.
        """
        if self.best_pipeline_ is None:
            raise RuntimeError("No trained model. Call train() first.")

        model = self.best_pipeline_.named_steps.get("model")
        if not hasattr(model, "feature_importances_"):
            logger.warning("Best model has no feature_importances_ attribute.")
            return pd.DataFrame()

        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df