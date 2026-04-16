"""
run_pipeline.py
Orchestrates the full Terra Nova pipeline:
  1. Ingest from FEMA API
  2. Engineer features
  3. Train models — tracked in MLflow
  4. Promote champion to MLflow Model Registry
  5. Print evaluation summary

Usage:
    python run_pipeline.py                        # full run from 2000
    python run_pipeline.py --start-year 2015      # partial run
    python run_pipeline.py --tracking-uri http://localhost:5000
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_ingestion.pipeline import IngestionPipeline
from feature_engineering.engineer import FeatureEngineer
from models.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("terra_nova")


def run(
    start_year: int = 2000,
    end_year: int = None,
    tracking_uri: str = "http://localhost:5000",
):
    # 1. Ingest
    logger.info("STEP 1 — Data Ingestion")
    pipeline = IngestionPipeline()
    datasets = pipeline.run(start_year=start_year)

    # 2. Feature engineering
    logger.info("STEP 2 — Feature Engineering")
    fe = FeatureEngineer(save_output=True)
    X, y = fe.build(
        declarations=datasets["declarations"],
        public_assistance=datasets["public_assistance"],
        disaster_summaries=datasets["disaster_summaries"],
    )

    # 3. Train + MLflow tracking
    logger.info("STEP 3 — Model Training (MLflow tracking_uri=%s)", tracking_uri)
    trainer = ModelTrainer(tracking_uri=tracking_uri)
    results = trainer.train(X, y)

    # 4. Summary
    logger.info("=== Evaluation Summary ===")
    for name, m in results.items():
        logger.info(
            "  %-25s R²=%.4f  RMSE=%.4f  MAE=%.4f  RMSE($)=$%.0f  run_id=%s",
            name, m["r2"], m["rmse"], m["mae"], m["rmse_dollars"],
            m.get("run_id", "n/a"),
        )

    fi = trainer.get_feature_importance(list(X.columns))
    if not fi.empty:
        logger.info("Top 5 features:")
        for _, row in fi.head(5).iterrows():
            logger.info("  %-30s %.4f", row["feature"], row["importance"])

    logger.info(
        "MLflow UI → %s  (experiment: terra_nova_disaster_cost_forecasting)",
        tracking_uri,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terra Nova pipeline runner")
    parser.add_argument("--start-year",    type=int, default=2000)
    parser.add_argument("--end-year",      type=int, default=None)
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    args = parser.parse_args()
    run(
        start_year=args.start_year,
        end_year=args.end_year,
        tracking_uri=args.tracking_uri,
    )
