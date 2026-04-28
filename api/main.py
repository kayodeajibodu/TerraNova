"""
Terra Nova FastAPI Backend
Exposes the trained predictive model via RESTful endpoints.

Endpoints:
  GET  /health           — liveness probe
  POST /predict-cost     — single disaster cost forecast
  POST /predict-batch    — batch predictions
  GET  /feature-importance — ranked feature weights for dashboard
  GET  /model-info       — active model metadata
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Terra Nova — Disaster Recovery Cost Forecasting API",
    version="1.0.0",
    description=(
        "Provides real-time predictions of disaster recovery costs "
        "based on incident characteristics and socioeconomic exposure."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (loaded once at startup)
# ---------------------------------------------------------------------------

MODEL_PATH = Path("models/best_model.pkl")
_model     = None


def _load_model():
    global _model
    if not MODEL_PATH.exists():
        logger.warning("Model file not found at %s — predictions will fail.", MODEL_PATH)
        return
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)


@app.on_event("startup")
async def startup_event():
    _load_model()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

INCIDENT_TYPES = [
    "HURRICANE", "FLOOD", "TORNADO", "FIRE", "EARTHQUAKE",
    "SEVERE STORM", "WINTER STORM", "DROUGHT", "BIOLOGICAL",
    "CHEMICAL", "DAM/LEVEE BREAK", "TSUNAMI", "VOLCANO", "OTHER",
]

DECLARATION_TYPES = ["DR", "EM", "FM"]


class DisasterInput(BaseModel):
    """Input features available at the point of disaster declaration."""

    disaster_number:          int            = Field(..., description="FEMA disaster number")
    state:                    str            = Field(..., min_length=2, max_length=2)
    incident_type:            str            = Field(..., description=f"One of: {INCIDENT_TYPES}")
    declaration_type:         str            = Field("DR", description="DR | EM | FM")
    incident_begin_date:      str            = Field(..., description="ISO date, e.g. 2023-08-15")
    incident_end_date:        Optional[str]  = Field(None, description="ISO date or null if ongoing")
    declaration_date:         str            = Field(..., description="ISO date of federal declaration")
    project_count_estimate:   Optional[int]  = Field(None, ge=0, description="Estimated number of PA projects")
    county_scope_estimate:    Optional[int]  = Field(None, ge=1, description="Number of designated counties")

    @field_validator("state")
    @classmethod
    def state_upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("incident_type")
    @classmethod
    def incident_type_upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("declaration_type")
    @classmethod
    def decl_type_upper(cls, v: str) -> str:
        return v.strip().upper()


class CostPrediction(BaseModel):
    disaster_number:           int
    predicted_cost_log:        float
    predicted_cost_dollars:    float
    lower_bound_dollars:       float
    upper_bound_dollars:       float
    confidence_level:          float = 0.80
    model_version:             str
    predicted_at:              str


class BatchInput(BaseModel):
    disasters: list[DisasterInput] = Field(..., min_length=1, max_length=100)


class BatchPrediction(BaseModel):
    predictions: list[CostPrediction]
    total_predicted_cost_dollars: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    rank: int


# ---------------------------------------------------------------------------
# Feature construction helpers (mirrors feature_engineering/engineer.py)
# ---------------------------------------------------------------------------

HIGH_RISK_STATES   = {"TX","FL","CA","LA","AL","MS","OK","KS","MO","TN"}
MEDIUM_RISK_STATES = {"NC","SC","GA","AR","NE","SD","ND","WY","CO","NM",
                      "WA","OR","ID","MT","AZ","NV","UT"}

INCIDENT_TYPE_MAP = {t: i for i, t in enumerate(sorted(INCIDENT_TYPES))}
DECL_TYPE_SEV_MAP = {"DR": 3, "EM": 2, "FM": 1}

FEATURE_NAMES = [
    "incident_duration_days", "declaration_lag_days",
    "declaration_year", "declaration_month",
    "incident_type_enc", "declaration_type_severity",
    "regional_risk_score", "disaster_frequency_5yr",
    "project_count", "mean_project_amount", "max_project_amount",
    "category_diversity", "has_high_cost_category", "county_scope",
]


def _build_feature_vector(inp: DisasterInput) -> np.ndarray:
    begin = datetime.fromisoformat(inp.incident_begin_date)
    decl  = datetime.fromisoformat(inp.declaration_date)
    end   = datetime.fromisoformat(inp.incident_end_date) if inp.incident_end_date else decl

    duration    = max((end - begin).days, 0)
    decl_lag    = max((decl - begin).days, 0)
    state       = inp.state.upper()
    risk_score  = 3 if state in HIGH_RISK_STATES else (2 if state in MEDIUM_RISK_STATES else 1)
    inc_enc     = INCIDENT_TYPE_MAP.get(inp.incident_type, len(INCIDENT_TYPES))
    decl_sev    = DECL_TYPE_SEV_MAP.get(inp.declaration_type, 0)

    features = [
        duration,                               # incident_duration_days
        decl_lag,                               # declaration_lag_days
        decl.year,                              # declaration_year
        decl.month,                             # declaration_month
        inc_enc,                                # incident_type_enc
        decl_sev,                               # declaration_type_severity
        risk_score,                             # regional_risk_score
        0,                                      # disaster_frequency_5yr (unknown at prediction time)
        inp.project_count_estimate or 0,        # project_count
        0,                                      # mean_project_amount
        0,                                      # max_project_amount
        0,                                      # category_diversity
        1 if inp.incident_type in {"HURRICANE","FLOOD","EARTHQUAKE"} else 0,
        inp.county_scope_estimate or 1,         # county_scope
    ]
    return np.array(features, dtype=float).reshape(1, -1)


def _prediction_interval(log_pred: float, risk_score: int) -> tuple[float, float]:
    """
    Simple ±σ interval in log-space, wider for high-severity events.
    In production replace with quantile regression or conformal prediction.
    """
    sigma = {1: 0.40, 2: 0.55, 3: 0.70}.get(risk_score, 0.50)
    lower = np.expm1(log_pred - 1.28 * sigma)   # 80 % CI lower
    upper = np.expm1(log_pred + 1.28 * sigma)   # 80 % CI upper
    return max(lower, 0.0), max(upper, 0.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict-cost", response_model=CostPrediction, tags=["Prediction"])
async def predict_cost(inp: DisasterInput):
    """
    Forecast total disaster recovery cost at the point of declaration.

    Returns a point estimate plus an 80 % prediction interval in dollars.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Retrain and deploy.")

    try:
        X          = _build_feature_vector(inp)
        log_pred   = float(_model.predict(X)[0])
        dollars    = float(np.expm1(log_pred))
        risk_score = 3 if inp.state in HIGH_RISK_STATES else (2 if inp.state in MEDIUM_RISK_STATES else 1)
        lower, upper = _prediction_interval(log_pred, risk_score)

        return CostPrediction(
            disaster_number=inp.disaster_number,
            predicted_cost_log=log_pred,
            predicted_cost_dollars=dollars,
            lower_bound_dollars=lower,
            upper_bound_dollars=upper,
            model_version="1.0.0",
            predicted_at=datetime.utcnow().isoformat(),
        )
    except Exception as exc:
        logger.exception("Prediction failed for disaster %d", inp.disaster_number)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-batch", response_model=BatchPrediction, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """Forecast costs for multiple disasters in a single request."""
    predictions = []
    for inp in batch.disasters:
        single = await predict_cost(inp)
        predictions.append(single)

    total = sum(p.predicted_cost_dollars for p in predictions)
    return BatchPrediction(predictions=predictions, total_predicted_cost_dollars=total)


@app.get("/feature-importance", response_model=list[FeatureImportanceItem], tags=["Model"])
async def feature_importance():
    """Return feature importances for the active model (tree-based only)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model_step = _model.named_steps.get("model")
    if not hasattr(model_step, "feature_importances_"):
        raise HTTPException(
            status_code=422,
            detail="Active model does not expose feature importances (linear model).",
        )

    importances = model_step.feature_importances_
    ranked = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        FeatureImportanceItem(feature=f, importance=float(imp), rank=i + 1)
        for i, (f, imp) in enumerate(ranked)
    ]


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if _model is None:
        return {"status": "no model loaded"}

    model_step = _model.named_steps.get("model")
    return {
        "model_class": type(model_step).__name__,
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "model_path": str(MODEL_PATH),
    }
