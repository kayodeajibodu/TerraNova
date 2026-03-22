"""
Ingestion Pipeline
Orchestrates fetching, cleaning, and persisting raw FEMA datasets.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from .fema_client import FEMAClient

logger = logging.getLogger(_name_)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBLIGATED_AMOUNT = 0.01


class IngestionPipeline:
    def _init_(self, client=None, output_dir=RAW_DATA_DIR):
        self.client     = client or FEMAClient()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, start_year=None, end_year=None, save=True):
        logger.info("=== Terra Nova Ingestion Pipeline starting ===")
        raw = self.client.fetch_all(start_year=start_year, end_year=end_year)

        cleaned = {
            "declarations":       self._clean_declarations(raw["declarations"]),
            "public_assistance":  self._clean_public_assistance(raw["public_assistance"]),
            "disaster_summaries": self._clean_disaster_summaries(raw["disaster_summaries"]),
        }

        self._log_summary(cleaned)

        if save:
            self._save(cleaned)

        logger.info("=== Ingestion Pipeline complete ===")
        return cleaned

    def _clean_declarations(self, df):
        if df.empty:
            return df
        df = df.copy()

        for col in ["declarationDate", "incidentBeginDate", "incidentEndDate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        for col in ["state", "incidentType", "declarationType"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.upper()

        df["disasterNumber"] = pd.to_numeric(df["disasterNumber"], errors="coerce")
        df = df.dropna(subset=["disasterNumber"])
        df["disasterNumber"] = df["disasterNumber"].astype(int)
        df = df.drop_duplicates(subset=["disasterNumber"])
        return df.reset_index(drop=True)

    def _clean_public_assistance(self, df):
        if df.empty:
            return df
        df = df.copy()

        # Rename v2 fields to internal spec-aligned names
        df = df.rename(columns={
            "stateAbbreviation": "state",
            "federalShareObligated": "obligatedAmount",
            "damageCategoryCode": "projectCategory",
            "pwNumber": "project_id",
        })

        df["obligatedAmount"] = pd.to_numeric(df["obligatedAmount"], errors="coerce").fillna(0.0)

        before = len(df)
        df = df[df["obligatedAmount"] >= MIN_OBLIGATED_AMOUNT]
        logger.debug("public_assistance: removed %d zero/cancelled rows", before - len(df))

        for col in ["state", "incidentType", "projectCategory"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        df["disasterNumber"] = pd.to_numeric(df["disasterNumber"], errors="coerce")
        df = df.dropna(subset=["disasterNumber"])
        df["disasterNumber"] = df["disasterNumber"].astype(int)
        return df.reset_index(drop=True)

    def _clean_disaster_summaries(self, df):
        if df.empty:
            return df
        df = df.copy()

        if "totalObligatedAmountPa" in df.columns:
            df["totalObligatedAmountPa"] = pd.to_numeric(
                df["totalObligatedAmountPa"], errors="coerce"
            ).fillna(0.0)

        df["disasterNumber"] = pd.to_numeric(df["disasterNumber"], errors="coerce")
        df = df.dropna(subset=["disasterNumber"])
        df["disasterNumber"] = df["disasterNumber"].astype(int)
        return df.reset_index(drop=True)

    def _save(self, datasets):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        for name, df in datasets.items():
            path = self.output_dir / f"{name}_{timestamp}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Saved %s -> %s (%d rows)", name, path, len(df))

    def _log_summary(self, datasets):
        for name, df in datasets.items():
            logger.info("%-25s  rows=%-8d  cols=%d", name, len(df), len(df.columns))