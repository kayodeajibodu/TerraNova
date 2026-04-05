"""
Feature Engineering Pipeline
Transforms cleaned raw DataFrames into a model-ready feature matrix.

Target variable: total_obligated_amount (per disaster)

Key engineered features:
  - incident_duration_days
  - declaration_lag_days
  - project_count / category diversity
  - regional risk encoding
  - disaster_frequency (state rolling)
  - log-transformed target (to handle right-skew)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Project categories that carry the most cost signal (from domain knowledge)
HIGH_COST_CATEGORIES = {"E", "F", "G"}   # Roads, buildings, utilities
CATEGORY_COLUMNS = ["A", "B", "C", "D", "E", "F", "G", "Z"]

# High-risk states based on historical FEMA disaster frequency
HIGH_RISK_STATES = {
    "TX", "FL", "CA", "LA", "AL", "MS", "OK", "KS", "MO", "TN"
}
MEDIUM_RISK_STATES = {
    "NC", "SC", "GA", "AR", "NE", "SD", "ND", "WY", "CO", "NM",
    "WA", "OR", "ID", "MT", "AZ", "NV", "UT"
}


class FeatureEngineer:
    """
    Builds the master feature matrix from the three cleaned datasets.

    Usage
    -----
    fe = FeatureEngineer()
    X, y = fe.build(declarations, public_assistance, disaster_summaries)
    """

    def __init__(self, save_output: bool = True):
        self.save_output   = save_output
        self.label_encoders: dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(
        self,
        declarations:       pd.DataFrame,
        public_assistance:  pd.DataFrame,
        disaster_summaries: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]: # return type hint for (X, y)
        """
        Build (X, y) ready for model training.

        Returns
        -------
        X : pd.DataFrame   — feature matrix
        y : pd.Series      — log1p(total_obligated_amount)
        """
        logger.info("Building feature matrix …")

        # Step 1: Aggregate project funding per disaster
        pa_agg = self._aggregate_public_assistance(public_assistance)

        # Step 2: Merge all three sources on disasterNumber
        master = self._merge_datasets(declarations, pa_agg, disaster_summaries)

        # Step 3: Derive temporal features
        master = self._add_temporal_features(master)

        # Step 4: Encode categorical features
        master = self._encode_categoricals(master)

        # Step 5: Add regional risk encoding
        master = self._add_regional_risk(master)

        # Step 6: Add disaster frequency (rolling count per state)
        master = self._add_disaster_frequency(master)

        # Step 7: Compute the target variable
        master, y = self._build_target(master)

        # Step 8: Select and finalise the feature columns
        X = self._select_features(master)

        logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)

        if self.save_output:
            self._save(X, y)

        return X, y

    # ------------------------------------------------------------------
    # Step 1: Aggregate public assistance per disaster
    # ------------------------------------------------------------------

    def _aggregate_public_assistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Roll project rows up to disaster level."""
        if df.empty:
            return pd.DataFrame(columns=["disasterNumber"])

        agg: dict = {
            "obligatedAmount": ["sum", "mean", "max", "count"],
        }

        # Pivot project-category totals as separate features
        category_dummies = pd.get_dummies(
            df["projectCategory"].fillna("UNKNOWN"),
            prefix="cat",
        )
        df_cats = pd.concat([df[["disasterNumber", "obligatedAmount"]], category_dummies], axis=1)

        # Base aggregations
        base = (
            df.groupby("disasterNumber")
            .agg(
                total_project_amount=("obligatedAmount", "sum"),
                mean_project_amount =("obligatedAmount", "mean"),
                max_project_amount  =("obligatedAmount", "max"),
                project_count       =("obligatedAmount", "count"),
            )
            .reset_index()
        )

        # Category spread (number of distinct project categories)
        cat_diversity = (
            df.groupby("disasterNumber")["projectCategory"]
            .nunique()
            .rename("category_diversity")
            .reset_index()
        )

        # High-cost category flag
        df["is_high_cost_cat"] = df["projectCategory"].isin(HIGH_COST_CATEGORIES).astype(int)
        high_cost = (
            df.groupby("disasterNumber")["is_high_cost_cat"]
            .max()
            .rename("has_high_cost_category")
            .reset_index()
        )

        # County spread (geographic scope proxy)
        county_scope = (
            df.groupby("disasterNumber")["countyCode"]
            .nunique()
            .rename("county_scope")
            .reset_index()
        )

        result = (
            base
            .merge(cat_diversity, on="disasterNumber", how="left")
            .merge(high_cost,     on="disasterNumber", how="left")
            .merge(county_scope,  on="disasterNumber", how="left")
        )

        return result

    # ------------------------------------------------------------------
    # Step 2: Merge datasets
    # ------------------------------------------------------------------

    def _merge_datasets(
        self,
        declarations:       pd.DataFrame,
        pa_agg:             pd.DataFrame,
        disaster_summaries: pd.DataFrame,
    ) -> pd.DataFrame:
        master = declarations.merge(pa_agg, on="disasterNumber", how="left")
        master = master.merge(disaster_summaries, on="disasterNumber", how="left")
        logger.debug("Master shape after merge: %s", master.shape)
        return master

    # ------------------------------------------------------------------
    # Step 3: Temporal features
    # ------------------------------------------------------------------

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Incident duration in days
        if "incidentBeginDate" in df.columns and "incidentEndDate" in df.columns:
            df["incident_duration_days"] = (
                df["incidentEndDate"] - df["incidentBeginDate"]
            ).dt.days.clip(lower=0).fillna(0)
        else:
            df["incident_duration_days"] = 0

        # Declaration lag — time from incident start to federal declaration
        if "declarationDate" in df.columns and "incidentBeginDate" in df.columns:
            df["declaration_lag_days"] = (
                df["declarationDate"] - df["incidentBeginDate"]
            ).dt.days.clip(lower=0).fillna(0)
        else:
            df["declaration_lag_days"] = 0

        # Declaration year and month (seasonal patterns)
        if "declarationDate" in df.columns:
            df["declaration_year"]  = df["declarationDate"].dt.year
            df["declaration_month"] = df["declarationDate"].dt.month
        else:
            df["declaration_year"]  = 0
            df["declaration_month"] = 0

        return df

    # ------------------------------------------------------------------
    # Step 4: Categorical encoding
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Incident type — label encode (manageable cardinality ~15 types)
        if "incidentType" in df.columns:
            le = LabelEncoder()
            df["incident_type_enc"] = le.fit_transform(
                df["incidentType"].fillna("UNKNOWN")
            )
            self.label_encoders["incidentType"] = le

        # Declaration type (DR=major disaster, EM=emergency, FM=fire management)
        decl_type_map = {"DR": 3, "EM": 2, "FM": 1}
        if "declarationType" in df.columns:
            df["declaration_type_severity"] = (
                df["declarationType"]
                .str.strip()
                .str.upper()
                .map(decl_type_map)
                .fillna(0)
                .astype(int)
            )

        return df

    # ------------------------------------------------------------------
    # Step 5: Regional risk score
    # ------------------------------------------------------------------

    def _add_regional_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def risk_score(state: str) -> int:
            s = str(state).strip().upper()
            if s in HIGH_RISK_STATES:
                return 3
            if s in MEDIUM_RISK_STATES:
                return 2
            return 1

        if "state" in df.columns:
            df["regional_risk_score"] = df["state"].apply(risk_score)

        return df

    # ------------------------------------------------------------------
    # Step 6: Disaster frequency (rolling count per state)
    # ------------------------------------------------------------------

    def _add_disaster_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "state" not in df.columns or "declarationDate" not in df.columns:
            df["disaster_frequency_5yr"] = 0
            return df

        df_sorted = df.sort_values("declarationDate")

        # Count disasters per state in the preceding 5 years
        def freq_5yr(row):
            cutoff = row["declarationDate"] - pd.DateOffset(years=5)
            same_state = df_sorted[
                (df_sorted["state"] == row["state"])
                & (df_sorted["declarationDate"] >= cutoff)
                & (df_sorted["declarationDate"] < row["declarationDate"])
            ]
            return len(same_state)

        df["disaster_frequency_5yr"] = df_sorted.apply(freq_5yr, axis=1)
        return df

    # ------------------------------------------------------------------
    # Step 7: Target variable
    # ------------------------------------------------------------------

    def _build_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Compute total_obligated_amount and apply log1p transform.
        Removes rows where we have no financial data at all.
        """
        df = df.copy()

        # Prefer the FEMA summary total; fall back to PA aggregation
        if "totalFederalShareObligations" in df.columns:
            df["total_obligated_amount"] = df["totalFederalShareObligations"].fillna(0)
        elif "total_project_amount" in df.columns:
            df["total_obligated_amount"] = df["total_project_amount"].fillna(0)
        else:
            df["total_obligated_amount"] = 0.0

        # Drop rows with no funding data
        df = df[df["total_obligated_amount"] > 0].copy()

        # Log1p transform to reduce right-skew
        y = np.log1p(df["total_obligated_amount"]).rename("log_total_obligated_amount")

        return df, y

    # ------------------------------------------------------------------
    # Step 8: Feature selection
    # ------------------------------------------------------------------

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            # Temporal
            "incident_duration_days",
            "declaration_lag_days",
            "declaration_year",
            "declaration_month",
            # Categorical
            "incident_type_enc",
            "declaration_type_severity",
            # Regional
            "regional_risk_score",
            "disaster_frequency_5yr",
            # Project-level aggregates
            "project_count",
            "mean_project_amount",
            "max_project_amount",
            "category_diversity",
            "has_high_cost_category",
            "county_scope",
        ]

        available = [c for c in feature_cols if c in df.columns]
        missing   = set(feature_cols) - set(available)
        if missing:
            logger.warning("Missing features (will be filled with 0): %s", missing)

        X = df[available].copy()
        for col in missing:
            X[col] = 0

        # Fill any remaining NaNs
        X = X.fillna(0)
        return X

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, X: pd.DataFrame, y: pd.Series) -> None:
        X.to_parquet(PROCESSED_DATA_DIR / "features.parquet", index=False)
        y.to_frame().to_parquet(PROCESSED_DATA_DIR / "target.parquet", index=False)
        logger.info("Saved features → data/processed/features.parquet")
        logger.info("Saved target   → data/processed/target.parquet")