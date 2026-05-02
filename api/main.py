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
import pandas as pd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

# Resolve MLflow tracking URI:
#   MLFLOW_TRACKING_URI env var → explicit remote server (Docker / cloud)
#   Not set                     → local SQLite file (development)
_env_uri = os.getenv("MLFLOW_TRACKING_URI")
if _env_uri:
    MLFLOW_TRACKING_URI = _env_uri
else:
    _db_path = Path(__file__).resolve().parent.parent / "mlflow.db"
    MLFLOW_TRACKING_URI = f"sqlite:///{_db_path}"
    logger.info("No MLFLOW_TRACKING_URI env var set, using local SQLite: %s", MLFLOW_TRACKING_URI)

logger.info("MLflow tracking URI initialized: %s", MLFLOW_TRACKING_URI)


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
        logger.info("Setting MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"
        logger.info("Loading champion model from MLflow registry: %s", model_uri)
        
        # Try sklearn first, then fall back to generic load (handles xgboost, etc.)
        try:
            _model = mlflow.sklearn.load_model(model_uri)
            logger.info("Successfully loaded model with sklearn flavor")
        except Exception as sklearn_err:
            logger.info("sklearn flavor not available (%s), trying generic load...", sklearn_err)
            _model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Successfully loaded model with generic pyfunc flavor")
        
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
    logger.info("Attempting to load model from local path: %s", LOCAL_MODEL_PATH)

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
    global _STATE_RISK_MAP
    _STATE_RISK_MAP = _load_state_risk_map()
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


# Loaded from data/processed/state_risk_tiers.csv (generated by FeatureEngineer)
# Falls back to score=2 (medium) for any unknown state
_STATE_RISK_MAP: dict[str, int] = {}

def _load_state_risk_map() -> dict[str, int]:
    """Load data-driven state risk tiers produced by the feature engineering step."""
    #import pandas as pd
    path = Path(os.getenv("STATE_RISK_PATH", "data/processed/state_risk_tiers.csv"))
    if path.exists():
        df = pd.read_csv(path)[["state", "risk_tier"]]
        mapping = dict(zip(df["state"].str.strip().str.upper(), df["risk_tier"].astype(int)))
        logger.info("Loaded state risk map: %d states from %s", len(mapping), path)
        return mapping
    logger.warning(
        "state_risk_tiers.csv not found at %s — defaulting all states to risk=2. "
        "Run run_pipeline.py to generate it.", path
    )
    return {}


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

def _build_feature_vector(inp: DisasterInput) -> "pd.DataFrame":
    """
    Returns a single-row DataFrame with named columns matching the MLflow
    model schema exactly. MLflow enforces column names AND data types.
    - 'integer' in schema = int32 in numpy/pandas
    - 'long' in schema = int64 in numpy/pandas
    - 'double' in schema = float64 in numpy/pandas
    """
    from datetime import datetime
    begin = datetime.fromisoformat(inp.incident_begin_date)
    decl  = datetime.fromisoformat(inp.declaration_date)
    end   = datetime.fromisoformat(inp.incident_end_date) if inp.incident_end_date else decl

    state      = inp.state.upper()
    risk_score = _STATE_RISK_MAP.get(state, 2)   # default=2 if state not in training data

    row = {
        "incident_duration_days":    np.float64(max((end - begin).days, 0)),
        "declaration_lag_days":      np.int64(max((decl - begin).days, 0)),
        "declaration_year":          np.int32(decl.year),
        "declaration_month":         np.int32(decl.month),
        "incident_type_enc":         np.int64(INCIDENT_TYPE_MAP.get(inp.incident_type, len(INCIDENT_TYPES))),
        "declaration_type_severity": np.int64(DECL_TYPE_SEV_MAP.get(inp.declaration_type, 0)),
        "regional_risk_score":       np.int64(risk_score),
        "disaster_frequency_5yr":    np.int64(0),
        "project_count":             np.float64(inp.project_count_estimate or 0),
        "mean_project_amount":       np.float64(0.0),
        "max_project_amount":        np.float64(0.0),
        "category_diversity":        np.float64(0.0),
        "has_high_cost_category":    np.float64(1 if inp.incident_type in {"HURRICANE", "FLOOD", "EARTHQUAKE"} else 0),
        "county_scope":              np.float64(inp.county_scope_estimate or 1),
    }
    # Ensure column order matches FEATURE_NAMES (MLflow schema order)
    return pd.DataFrame([row])[FEATURE_NAMES]



def _prediction_interval(log_pred: float, risk_score: int):
    # Wider interval for higher-risk states (more cost variability historically)
    sigma = {1: 0.40, 2: 0.55, 3: 0.70}.get(int(risk_score), 0.55)
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
        risk_score = _STATE_RISK_MAP.get(inp.state.upper(), 2)
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
    
    # Try to get feature importances from the underlying model
    model_step = getattr(_model, "named_steps", {}).get("model", _model)
    
    # For pyfunc models, check the underlying model
    if hasattr(_model, "metadata") and hasattr(_model.metadata, "model_type"):
        logger.info("Model is pyfunc type: %s", _model.metadata.model_type)
        # Can't extract feature importance from pyfunc wrapper
        raise HTTPException(
            status_code=422,
            detail="Feature importance not available for this model type.",
        )
    
    if not hasattr(model_step, "feature_importances_"):
        raise HTTPException(
            status_code=422,
            detail="Active model does not expose feature importances.",
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
