"""
run_pipeline.py
Orchestrates the full Terra Nova pipeline end-to-end:
  1. Ingest from FEMA API
  2. Engineer features
  3. Train models
  4. Print evaluation summary
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))
from data_ingestion.pipeline import IngestionPipeline
from feature_engineering.engineer import FeatureEngineer
from models.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("terra_nova")


def run(start_year: int = 2000, end_year: int = None):
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

    # 3. Train models
    logger.info("STEP 3 — Model Training")
    trainer = ModelTrainer()
    results = trainer.train(X, y)

    # 4. Summary
    logger.info("=== Evaluation Summary ===")
    for name, m in results.items():
        logger.info(
            "  %-25s R²=%.4f  RMSE=%.2f  MAE=%.2f  RMSE($)=%s",
            name,
            m["r2"],
            m["rmse"],
            m["mae"],
            f"${m['rmse_dollars']:,.0f}",
        )

    fi = trainer.get_feature_importance(list(X.columns))
    if not fi.empty:
        logger.info("Top 5 features:")
        for _, row in fi.head(5).iterrows():
            logger.info("  %-30s %.4f", row["feature"], row["importance"])


if __name__ == "__main__":
    run(start_year=2000)