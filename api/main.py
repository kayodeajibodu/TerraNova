"""
Terra Nova FastAPI Backend — with MLflow Model Registry integration
Loads the 'champion' model from the MLflow registry at startup,
with a local pkl fallback if the registry is unavailable.

Endpoints:
  GET  /health              — liveness probe + model source info
  POST /predict-cost        — single disaster cost forecast
  POST /predict-batch       — batch predictions
  GET  /feature-importance  — ranked feature weights
  GET  /model-info          — active model metadata + MLflow run details
"""

import logging
import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import mlflow
import mlflow.sklearn
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
        "Provides real-time predictions of disaster recovery costs. "
        "Model served from MLflow Model Registry (champion alias)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# MLflow + model configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL     = os.getenv("MLFLOW_MODEL_NAME",   "terra_nova_cost_model")
CHAMPION_ALIAS       = "champion"
LOCAL_MODEL_PATH     = Path(os.getenv("MODEL_PATH", "models/best_model.pkl"))

_model        = None
_model_source = "none"      # "registry" | "local_pkl" | "none"
_model_meta   = {}


def _load_model():
    global _model, _model_source, _model_meta

    # ── Try MLflow Model Registry ──────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"
        logger.info("Loading champion model from MLflow registry: %s", model_uri)
        _model        = mlflow.sklearn.load_model(model_uri)
        _model_source = "registry"

        # Fetch metadata for /model-info
        client  = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
        run     = client.get_run(version.run_id)
        _model_meta = {
            "registry_name":    REGISTERED_MODEL,
            "version":          version.version,
            "alias":            CHAMPION_ALIAS,
            "run_id":           version.run_id,
            "run_name":         run.data.tags.get("mlflow.runName", ""),
            "model_family":     run.data.tags.get("model_family", ""),
            "r2":               run.data.metrics.get("r2"),
            "rmse_dollars":     run.data.metrics.get("rmse_dollars"),
            "registered_at":    version.creation_timestamp,
        }
        logger.info(
            "Loaded: %s v%s | family=%s | R²=%.4f",
            REGISTERED_MODEL, version.version,
            _model_meta.get("model_family"),
            _model_meta.get("r2") or 0,
        )
        return

    except Exception as exc:
        logger.warning(
            "MLflow registry unavailable (%s). Falling back to local pkl.", exc
        )

    # ── Fallback: local pickle ──────────────────────────────────
    if LOCAL_MODEL_PATH.exists():
        with open(LOCAL_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _model_source = "local_pkl"
        _model_meta   = {"source_file": str(LOCAL_MODEL_PATH)}
        logger.info("Model loaded from local pkl: %s", LOCAL_MODEL_PATH)
    else:
        logger.warning(
            "No model found at %s. Predictions will return 503.", LOCAL_MODEL_PATH
        )


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
DECL_TYPE_SEV_MAP = {"DR": 3, "EM": 2, "FM": 1}
INCIDENT_TYPE_MAP = {t: i for i, t in enumerate(sorted(INCIDENT_TYPES))}

HIGH_RISK_STATES   = {"TX","FL","CA","LA","AL","MS","OK","KS","MO","TN"}
MEDIUM_RISK_STATES = {"NC","SC","GA","AR","NE","SD","ND","WY","CO","NM",
                      "WA","OR","ID","MT","AZ","NV","UT"}

FEATURE_NAMES = [
    "incident_duration_days", "declaration_lag_days",
    "declaration_year", "declaration_month",
    "incident_type_enc", "declaration_type_severity",
    "regional_risk_score", "disaster_frequency_5yr",
    "project_count", "mean_project_amount", "max_project_amount",
    "category_diversity", "has_high_cost_category", "county_scope",
]


class DisasterInput(BaseModel):
    disaster_number:         int
    state:                   str = Field(..., min_length=2, max_length=2)
    incident_type:           str
    declaration_type:        str = "DR"
    incident_begin_date:     str
    incident_end_date:       Optional[str] = None
    declaration_date:        str
    project_count_estimate:  Optional[int] = Field(None, ge=0)
    county_scope_estimate:   Optional[int] = Field(None, ge=1)

    @field_validator("state", "incident_type", "declaration_type")
    @classmethod
    def upper(cls, v): return v.strip().upper()


class CostPrediction(BaseModel):
    disaster_number:        int
    predicted_cost_log:     float
    predicted_cost_dollars: float
    lower_bound_dollars:    float
    upper_bound_dollars:    float
    confidence_level:       float = 0.80
    model_source:           str
    model_version:          str = "1.0.0"
    predicted_at:           str


class BatchInput(BaseModel):
    disasters: list[DisasterInput] = Field(..., min_length=1, max_length=100)


class BatchPrediction(BaseModel):
    predictions:                  list[CostPrediction]
    total_predicted_cost_dollars: float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_source: str
    timestamp:    str


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def _build_feature_vector(inp: DisasterInput) -> "np.ndarray":
    from datetime import datetime
    begin = datetime.fromisoformat(inp.incident_begin_date)
    decl  = datetime.fromisoformat(inp.declaration_date)
    end   = datetime.fromisoformat(inp.incident_end_date) if inp.incident_end_date else decl

    state      = inp.state.upper()
    risk_score = 3 if state in HIGH_RISK_STATES else (2 if state in MEDIUM_RISK_STATES else 1)

    return np.array([
        max((end - begin).days, 0),
        max((decl - begin).days, 0),
        decl.year,
        decl.month,
        INCIDENT_TYPE_MAP.get(inp.incident_type, len(INCIDENT_TYPES)),
        DECL_TYPE_SEV_MAP.get(inp.declaration_type, 0),
        risk_score,
        0,
        inp.project_count_estimate or 0,
        0, 0, 0,
        1 if inp.incident_type in {"HURRICANE", "FLOOD", "EARTHQUAKE"} else 0,
        inp.county_scope_estimate or 1,
    ], dtype=float).reshape(1, -1)


def _prediction_interval(log_pred: float, risk_score: int):
    sigma = {1: 0.40, 2: 0.55, 3: 0.70}.get(risk_score, 0.50)
    lower = max(np.expm1(log_pred - 1.28 * sigma), 0.0)
    upper = max(np.expm1(log_pred + 1.28 * sigma), 0.0)
    return lower, upper


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_source=_model_source,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict-cost", response_model=CostPrediction, tags=["Prediction"])
async def predict_cost(inp: DisasterInput):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline and ensure MLflow is reachable.",
        )
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
            model_source=_model_source,
            predicted_at=datetime.utcnow().isoformat(),
        )
    except Exception as exc:
        logger.exception("Prediction failed for disaster %d", inp.disaster_number)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-batch", response_model=BatchPrediction, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    predictions = [await predict_cost(inp) for inp in batch.disasters]
    return BatchPrediction(
        predictions=predictions,
        total_predicted_cost_dollars=sum(p.predicted_cost_dollars for p in predictions),
    )


@app.get("/feature-importance", tags=["Model"])
async def feature_importance():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    model_step = getattr(_model, "named_steps", {}).get("model", _model)
    if not hasattr(model_step, "feature_importances_"):
        raise HTTPException(
            status_code=422,
            detail="Active model does not expose feature importances (linear model).",
        )
    importances = model_step.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    return [
        {"feature": f, "importance": float(imp), "rank": i + 1}
        for i, (f, imp) in enumerate(ranked)
    ]


@app.get("/model-info", tags=["Model"])
async def model_info():
    if _model is None:
        return {"status": "no model loaded"}
    model_step = getattr(_model, "named_steps", {}).get("model", _model)
    return {
        "model_class":  type(model_step).__name__,
        "model_source": _model_source,
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        **_model_meta,
    }


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Hot-reload the model from MLflow registry without restarting the container."""
    _load_model()
    return {
        "status": "reloaded",
        "model_source": _model_source,
        "timestamp": datetime.utcnow().isoformat(),
        **_model_meta,
    }
