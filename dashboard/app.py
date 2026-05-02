"""
Terra Nova — Disaster Recovery Cost Forecasting Dashboard
Streamlit frontend connected to the FastAPI backend.

Run with:
    streamlit run dashboard/app.py
"""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("TERRA_NOVA_API_URL", "http://localhost:8000")

# MLflow — resolve same way as trainer/api: env var → local sqlite
_mlflow_env = os.getenv("MLFLOW_TRACKING_URI")
if _mlflow_env:
    MLFLOW_TRACKING_URI = _mlflow_env
else:
    _db = Path(__file__).resolve().parent.parent / "mlflow.db"
    MLFLOW_TRACKING_URI = f"sqlite:///{_db}"

EXPERIMENT_NAME = "terra_nova_disaster_cost_forecasting"
REGISTERED_NAME = "terra_nova_cost_model"
CHAMPION_ALIAS  = "champion"

INCIDENT_TYPES = [
    "HURRICANE", "FLOOD", "TORNADO", "FIRE", "EARTHQUAKE",
    "SEVERE STORM", "WINTER STORM", "DROUGHT", "BIOLOGICAL",
    "CHEMICAL", "DAM/LEVEE BREAK", "TSUNAMI", "VOLCANO", "OTHER",
]

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]

BLUE   = "#4A90D9"
ORANGE = "#F5A623"
RED    = "#E74C3C"
GREEN  = "#2ECC71"
DARK   = "#1F3864"

st.set_page_config(
    page_title="Terra Nova | Disaster Cost Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers — API
# ---------------------------------------------------------------------------

def fmt_dollars(v: float) -> str:
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.2f}M"
    if v >= 1e3:  return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


def api_predict(payload: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_URL}/predict-cost", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️  Cannot connect to the FastAPI backend. Is it running?")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error: {exc.response.json().get('detail', str(exc))}")
        return None


def api_feature_importance() -> Optional[list]:
    try:
        r = requests.get(f"{API_URL}/feature-importance", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False, "model_source": "none"}


def api_model_info() -> dict:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        return r.json()
    except Exception:
        return {}


def api_reload_model() -> dict:
    try:
        r = requests.post(f"{API_URL}/model/reload", timeout=15)
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers — MLflow (direct client, bypasses API)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_mlflow_client():
    try:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    except Exception:
        return None


def load_runs_df(client, experiment_name: str) -> pd.DataFrame:
    """Return all runs for the experiment as a tidy DataFrame."""
    try:
        import mlflow
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.r2 DESC"],
        )
        rows = []
        for r in runs:
            rows.append({
                "run_id":       r.info.run_id[:8],
                "run_id_full":  r.info.run_id,
                "model":        r.data.tags.get("model_family", r.info.run_name),
                "status":       r.info.status,
                "r2":           r.data.metrics.get("r2"),
                "rmse":         r.data.metrics.get("rmse"),
                "mae":          r.data.metrics.get("mae"),
                "rmse_dollars": r.data.metrics.get("rmse_dollars"),
                "cv_r2_mean":   r.data.metrics.get("cv_r2_mean"),
                "cv_r2_std":    r.data.metrics.get("cv_r2_std"),
                "n_train_rows": r.data.params.get("n_train_rows"),
                "n_features":   r.data.params.get("n_features"),
                "started":      pd.to_datetime(r.info.start_time, unit="ms"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def load_champion_info(client) -> Optional[dict]:
    try:
        v = client.get_model_version_by_alias(REGISTERED_NAME, CHAMPION_ALIAS)
        run = client.get_run(v.run_id)
        return {
            "version":      v.version,
            "run_id":       v.run_id,
            "model_family": run.data.tags.get("model_family", "unknown"),
            "r2":           run.data.metrics.get("r2"),
            "rmse_dollars": run.data.metrics.get("rmse_dollars"),
            "cv_r2_mean":   run.data.metrics.get("cv_r2_mean"),
            "cv_r2_std":    run.data.metrics.get("cv_r2_std"),
            "n_train_rows": run.data.params.get("n_train_rows"),
            "registered_at":pd.to_datetime(v.creation_timestamp, unit="ms"),
        }
    except Exception:
        return None


def load_state_risk() -> pd.DataFrame:
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "state_risk_tiers.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://em-content.zobj.net/source/google/387/globe-showing-americas_1f30e.png",
        width=60,
    )
    st.title("Terra Nova")
    st.caption("Disaster Recovery Cost Forecasting")
    st.divider()

    health = api_health()
    loaded = health.get("model_loaded", False)
    src    = health.get("model_source", "none")
    color  = "green" if loaded else "red"
    st.markdown(f"**API:** :{color}[{'● Online' if loaded else '● Offline'}]")
    if loaded:
        st.caption(f"Model source: `{src}`")
    st.caption(f"Backend: `{API_URL}`")
    st.divider()

    st.markdown("**Navigation**")
    page = st.radio(
        label="page",
        options=[
            "Cost Forecast",
            "Scenario Simulation",
            "Feature Importance",
            "Budget Gap Analysis",
            "MLflow Tracking",
        ],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Page 1 — Cost Forecast
# ---------------------------------------------------------------------------

if page == "Cost Forecast":
    st.header("🔍 Disaster Cost Forecast")
    st.caption("Enter incident details to generate a real-time recovery cost estimate.")

    col1, col2 = st.columns(2)
    with col1:
        disaster_number  = st.number_input("FEMA Disaster Number", min_value=1000, value=4000, step=1)
        state            = st.selectbox("State", US_STATES, index=US_STATES.index("TX"))
        incident_type    = st.selectbox("Incident Type", INCIDENT_TYPES)
        declaration_type = st.selectbox("Declaration Type", ["DR", "EM", "FM"])

    with col2:
        begin_date   = st.date_input("Incident Begin Date", value=date.today() - timedelta(days=14))
        end_date_raw = st.date_input("Incident End Date",   value=date.today())
        decl_date    = st.date_input("Declaration Date",    value=date.today())
        proj_count   = st.number_input("Estimated PA Projects", min_value=0,  value=50,  step=10)
        county_scope = st.number_input("Designated Counties",   min_value=1,  value=5,   step=1)

    if st.button("Generate Forecast", type="primary", use_container_width=True):
        payload = {
            "disaster_number":        int(disaster_number),
            "state":                  state,
            "incident_type":          incident_type,
            "declaration_type":       declaration_type,
            "incident_begin_date":    begin_date.isoformat(),
            "incident_end_date":      end_date_raw.isoformat(),
            "declaration_date":       decl_date.isoformat(),
            "project_count_estimate": int(proj_count),
            "county_scope_estimate":  int(county_scope),
        }
        with st.spinner("Forecasting …"):
            result = api_predict(payload)

        if result:
            st.success("Forecast generated")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Cost",    fmt_dollars(result["predicted_cost_dollars"]))
            m2.metric("Lower Bound (80%)", fmt_dollars(result["lower_bound_dollars"]))
            m3.metric("Upper Bound (80%)", fmt_dollars(result["upper_bound_dollars"]))

            mu    = result["predicted_cost_log"]
            sigma = 0.55
            x     = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 300)
            pdf   = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            x_dollars = np.expm1(x)
            mask  = (x >= np.log1p(result["lower_bound_dollars"])) & \
                    (x <= np.log1p(result["upper_bound_dollars"]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_dollars, y=pdf, fill="tozeroy",
                fillcolor="rgba(74,144,217,0.15)",
                line=dict(color=BLUE, width=2), name="Cost distribution",
            ))
            fig.add_trace(go.Scatter(
                x=x_dollars[mask], y=pdf[mask], fill="tozeroy",
                fillcolor="rgba(74,144,217,0.45)",
                line=dict(color=BLUE, width=0), name="80% CI",
            ))
            fig.add_vline(
                x=result["predicted_cost_dollars"],
                line=dict(color=RED, width=2, dash="dash"),
                annotation_text=f"Forecast: {fmt_dollars(result['predicted_cost_dollars'])}",
            )
            fig.update_layout(
                title="Recovery Cost Probability Distribution",
                xaxis_title="Total Obligated Amount (USD)",
                yaxis_title="Probability Density",
                height=340, margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw API response"):
                st.json(result)

# ---------------------------------------------------------------------------
# Page 2 — Scenario Simulation
# ---------------------------------------------------------------------------

elif page == "Scenario Simulation":
    st.header("🌡️ Scenario Simulation")
    st.caption("Adjust climate and scale parameters to model cost under different future conditions.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Base scenario")
        base_state = st.selectbox("State", US_STATES, index=US_STATES.index("FL"))
        base_type  = st.selectbox("Incident type", INCIDENT_TYPES,
                                   index=INCIDENT_TYPES.index("HURRICANE"))
        base_dur   = st.slider("Incident duration (days)", 1, 60, 10)
        base_scope = st.slider("Counties affected", 1, 67, 12)
        base_proj  = st.slider("Estimated PA projects", 10, 2000, 200)
        st.subheader("Climate multipliers")
        intensity_mult = st.slider("Intensity scaling (1× = baseline)", 1.0, 3.0, 1.0, 0.1)
        scope_mult     = st.slider("Geographic spread multiplier",       1.0, 3.0, 1.0, 0.1)

    with col2:
        scenarios = {
            "Baseline":               (1.0, 1.0),
            "+50% intensity":         (1.5, 1.0),
            "+100% intensity":        (2.0, 1.0),
            "Wider geographic scope": (1.0, 1.5),
            "Compound (2× / 2×)":    (2.0, 2.0),
            "Custom":                 (intensity_mult, scope_mult),
        }
        rows = []
        for label, (im, sm) in scenarios.items():
            pl = {
                "disaster_number":        9999,
                "state":                  base_state,
                "incident_type":          base_type,
                "declaration_type":       "DR",
                "incident_begin_date":    date.today().isoformat(),
                "incident_end_date":      (date.today() + timedelta(days=int(base_dur * im))).isoformat(),
                "declaration_date":       date.today().isoformat(),
                "project_count_estimate": int(base_proj * im),
                "county_scope_estimate":  int(base_scope * sm),
            }
            res = api_predict(pl)
            if res:
                rows.append({"Scenario": label,
                             "Cost ($)":  res["predicted_cost_dollars"],
                             "Lower ($)": res["lower_bound_dollars"],
                             "Upper ($)": res["upper_bound_dollars"]})

        if rows:
            df_s = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_s["Scenario"], y=df_s["Cost ($)"],
                error_y=dict(
                    type="data", symmetric=False,
                    array=df_s["Upper ($)"] - df_s["Cost ($)"],
                    arrayminus=df_s["Cost ($)"] - df_s["Lower ($)"],
                ),
                marker_color=[BLUE if s != "Custom" else ORANGE for s in df_s["Scenario"]],
                name="Predicted cost",
            ))
            fig.update_layout(title="Scenario Cost Comparison",
                              yaxis_title="Total Obligated Amount (USD)",
                              height=420, margin=dict(t=50, b=80))
            st.plotly_chart(fig, use_container_width=True)
            disp = df_s.copy()
            for c in ["Cost ($)", "Lower ($)", "Upper ($)"]:
                disp[c] = disp[c].apply(fmt_dollars)
            st.dataframe(disp, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Page 3 — Feature Importance
# ---------------------------------------------------------------------------

elif page == "Feature Importance":
    st.header("📊 Feature Importance")
    st.caption("Relative contribution of each input variable to the predicted recovery cost.")

    fi_data = api_feature_importance()
    if fi_data:
        df_fi = pd.DataFrame(fi_data).sort_values("importance", ascending=True)
        fig = px.bar(
            df_fi, x="importance", y="feature", orientation="h",
            labels={"importance": "Importance Score", "feature": "Feature"},
            color="importance",
            color_continuous_scale=["#B5D4F4", BLUE, DARK],
            title="Feature importances (active model)",
        )
        fig.update_layout(height=480, showlegend=False,
                          coloraxis_showscale=False, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

        descriptions = {
            "incident_duration_days":    "Duration from incident start to end.",
            "declaration_lag_days":      "Days from incident start to federal declaration.",
            "project_count":             "Number of approved public assistance projects.",
            "county_scope":              "Number of designated counties — geographic extent.",
            "regional_risk_score":       "Data-driven state risk tier (1=low, 2=medium, 3=high). Computed from historical disaster frequency, mean cost, and incident diversity.",
            "incident_type_enc":         "Encoded incident type (hurricane, flood, etc.).",
            "declaration_type_severity": "Severity class of declaration (DR=3 > EM=2 > FM=1).",
            "declaration_year":          "Year of declaration — captures long-term cost trends.",
            "declaration_month":         "Month — captures seasonal disaster patterns.",
            "disaster_frequency_5yr":    "Count of prior disasters in same state over 5 years.",
            "has_high_cost_category":    "Whether high-cost project categories (E/F/G) are present.",
            "category_diversity":        "Number of distinct project categories in the disaster.",
            "mean_project_amount":       "Average per-project obligated amount.",
            "max_project_amount":        "Largest single project obligation — extreme event signal.",
        }
        for item in sorted(fi_data, key=lambda x: x["rank"]):
            with st.expander(f"#{item['rank']}  {item['feature']}  — {item['importance']:.4f}"):
                st.write(descriptions.get(item["feature"], "No description available."))
    else:
        st.info("Feature importance unavailable. Ensure the API is running with a tree-based model.")

# ---------------------------------------------------------------------------
# Page 4 — Budget Gap Analysis
# ---------------------------------------------------------------------------

elif page == "Budget Gap Analysis":
    st.header("💰 Budget Gap Analysis")
    st.caption("Compare your allocated disaster budget against the forecast cost distribution.")

    col1, col2 = st.columns([1, 2])
    with col1:
        budget = st.number_input("Allocated budget (USD millions)",
                                  min_value=0.0, value=500.0, step=50.0) * 1_000_000
        st.subheader("Forecast scenario")
        gap_state = st.selectbox("State", US_STATES, key="gap_state")
        gap_type  = st.selectbox("Incident type", INCIDENT_TYPES, key="gap_type")
        gap_dur   = st.slider("Duration (days)", 1, 90, 14, key="gap_dur")
        gap_scope = st.slider("Counties affected", 1, 67, 10, key="gap_scope")
        gap_proj  = st.slider("PA projects", 10, 2000, 150, key="gap_proj")
        run_gap   = st.button("Run Analysis", type="primary", use_container_width=True)

    with col2:
        if run_gap:
            pl = {
                "disaster_number":        8888,
                "state":                  gap_state,
                "incident_type":          gap_type,
                "declaration_type":       "DR",
                "incident_begin_date":    date.today().isoformat(),
                "incident_end_date":      (date.today() + timedelta(days=gap_dur)).isoformat(),
                "declaration_date":       date.today().isoformat(),
                "project_count_estimate": int(gap_proj),
                "county_scope_estimate":  int(gap_scope),
            }
            with st.spinner("Running …"):
                result = api_predict(pl)

            if result:
                predicted = result["predicted_cost_dollars"]
                lower     = result["lower_bound_dollars"]
                upper     = result["upper_bound_dollars"]
                gap_amt   = predicted - budget
                worst_gap = upper - budget

                m1, m2 = st.columns(2)
                m1.metric("Predicted cost", fmt_dollars(predicted))
                m2.metric("Budget gap", fmt_dollars(abs(gap_amt)),
                           delta=f"{'Over' if gap_amt > 0 else 'Under'} budget",
                           delta_color="inverse")

                max_val = upper * 1.2
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted,
                    delta={"reference": budget, "valueformat": "$,.0f"},
                    gauge={
                        "axis": {"range": [0, max_val]},
                        "bar":  {"color": BLUE},
                        "steps": [
                            {"range": [0, lower],      "color": "rgba(46,204,113,0.25)"},
                            {"range": [lower, upper],  "color": "rgba(245,166,35,0.25)"},
                            {"range": [upper, max_val],"color": "rgba(231,76,60,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 3},
                            "thickness": 0.85, "value": budget,
                        },
                    },
                    number={"prefix": "$", "valueformat": ",.0f"},
                    title={"text": "Forecast vs Budget"},
                ))
                fig.update_layout(height=320, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                if predicted <= budget:
                    st.success(f"✅ Budget adequate. Forecast is {fmt_dollars(abs(gap_amt))} under allocation.")
                elif predicted <= budget * 1.25:
                    st.warning(f"⚠️ Slightly over budget by {fmt_dollars(gap_amt)} ({gap_amt/budget*100:.0f}%).")
                else:
                    st.error(
                        f"🚨 Significant shortfall: {fmt_dollars(gap_amt)} over "
                        f"({gap_amt/budget*100:.0f}%). Worst-case gap: {fmt_dollars(worst_gap)}."
                    )

# ---------------------------------------------------------------------------
# Page 5 — MLflow Tracking
# ---------------------------------------------------------------------------

elif page == "MLflow Tracking":
    st.header("🧪 MLflow Experiment Tracking")
    st.caption(f"Tracking store: `{MLFLOW_TRACKING_URI}`")

    client = get_mlflow_client()
    if client is None:
        st.error("MLflow client could not be initialised. Is `mlflow` installed?")
        st.stop()

    # ── Champion banner ───────────────────────────────────────────────────────
    st.subheader("🏆 Current Champion Model")
    champ = load_champion_info(client)

    if champ:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model family",  champ["model_family"].replace("_", " ").title())
        c2.metric("R²",            f"{champ['r2']:.4f}" if champ["r2"] else "—")
        c3.metric("RMSE ($)",      fmt_dollars(champ["rmse_dollars"]) if champ["rmse_dollars"] else "—")
        c4.metric("CV R² (mean)",  f"{champ['cv_r2_mean']:.4f} ± {champ['cv_r2_std']:.4f}"
                                    if champ["cv_r2_mean"] else "—")

        with st.expander("Champion details"):
            st.json({
                "registered_model": REGISTERED_NAME,
                "version":          champ["version"],
                "alias":            CHAMPION_ALIAS,
                "run_id":           champ["run_id"],
                "model_family":     champ["model_family"],
                "r2":               champ["r2"],
                "rmse_dollars":     champ["rmse_dollars"],
                "n_train_rows":     champ["n_train_rows"],
                "registered_at":    str(champ["registered_at"]),
            })
    else:
        st.warning("No champion model found. Run `python run_pipeline.py` to train and register a model.")

    st.divider()

    # ── Hot reload ────────────────────────────────────────────────────────────
    st.subheader("🔄 Model Reload")
    st.caption("Reload the champion from the registry into the live API without restarting the container.")
    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        if st.button("Reload Champion into API", type="primary", use_container_width=True):
            with st.spinner("Reloading …"):
                resp = api_reload_model()
            if "error" in resp:
                st.error(f"Reload failed: {resp['error']}")
            else:
                st.success(
                    f"✅ Reloaded  |  source: `{resp.get('model_source')}`  "
                    f"|  version: `{resp.get('version', 'n/a')}`"
                )
    with col_r2:
        info = api_model_info()
        if info:
            src = info.get("model_source", "unknown")
            src_color = "green" if src == "registry" else "orange"
            st.markdown(
                f"API currently serving from: :{src_color}[**{src}**]  "
                f"· model class: `{info.get('model_class', '?')}`  "
                f"· R²: `{info.get('r2', '?')}`"
            )

    st.divider()

    # ── All runs table ────────────────────────────────────────────────────────
    st.subheader("📋 All Experiment Runs")
    runs_df = load_runs_df(client, EXPERIMENT_NAME)

    if runs_df.empty:
        st.info("No runs found. Execute the training pipeline to populate the experiment.")
    else:
        # Highlight champion run
        if champ:
            runs_df["champion"] = runs_df["run_id_full"].str.startswith(
                champ["run_id"][:8]
            ).map({True: "⭐ champion", False: ""})
        else:
            runs_df["champion"] = ""

        display_cols = ["champion", "run_id", "model", "r2", "rmse",
                        "mae", "rmse_dollars", "cv_r2_mean", "cv_r2_std", "started"]
        disp = runs_df[[c for c in display_cols if c in runs_df.columns]].copy()

        # Format floats
        for col in ["r2", "rmse", "mae", "cv_r2_mean", "cv_r2_std"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
        if "rmse_dollars" in disp.columns:
            disp["rmse_dollars"] = disp["rmse_dollars"].apply(
                lambda v: fmt_dollars(v) if pd.notna(v) else "—"
            )

        st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── R² comparison bar chart ───────────────────────────────────────────
        st.subheader("📈 R² by Model — All Runs")
        chart_df = runs_df.dropna(subset=["r2"]).copy()
        chart_df["r2"] = chart_df["r2"].astype(float)
        chart_df["label"] = chart_df["model"] + " (" + chart_df["run_id"] + ")"
        chart_df = chart_df.sort_values("r2", ascending=False)

        champ_run_id = champ["run_id"][:8] if champ else ""
        colors = [
            "#F5A623" if rid == champ_run_id else "#4A90D9"
            for rid in chart_df["run_id"]
        ]

        fig_r2 = go.Figure(go.Bar(
            x=chart_df["label"],
            y=chart_df["r2"],
            marker_color=colors,
            text=chart_df["r2"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        ))
        fig_r2.add_hline(
            y=0.75, line_dash="dash", line_color="red",
            annotation_text="Target R² = 0.75",
            annotation_position="bottom right",
        )
        fig_r2.update_layout(
            title="R² Score by Run  (🟠 = champion)",
            yaxis_title="R²", yaxis_range=[0, 1.05],
            xaxis_title="Run", height=380,
            margin=dict(t=60, b=100),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        # ── Metric radar for latest run per model ─────────────────────────────
        st.subheader("🕸️ Model Comparison — Normalised Metrics")
        latest = runs_df.sort_values("started", ascending=False).drop_duplicates("model")
        radar_metrics = ["r2", "cv_r2_mean"]
        radar_df = latest.dropna(subset=radar_metrics).copy()
        for col in radar_metrics:
            radar_df[col] = pd.to_numeric(radar_df[col], errors="coerce")

        if len(radar_df) >= 2:
            fig_radar = go.Figure()
            categories = ["R²", "CV R²", "R²"]   # close the loop
            for _, row in radar_df.iterrows():
                values = [
                    float(row["r2"] or 0),
                    float(row["cv_r2_mean"] or 0),
                    float(row["r2"] or 0),   # close polygon
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=categories,
                    fill="toself", name=row["model"].replace("_", " ").title(),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=380,
                title="Model comparison — normalised score radar",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── State risk tiers (data-driven) ────────────────────────────────────────
    st.subheader("🗺️ Data-Driven State Risk Tiers")
    risk_df = load_state_risk()

    if risk_df.empty:
        st.info(
            "State risk tiers not yet computed. "
            "Run the pipeline to generate `data/processed/state_risk_tiers.csv`."
        )
    else:
        tier_colors = {3: RED, 2: ORANGE, 1: GREEN}
        tier_labels = {3: "High (3)", 2: "Medium (2)", 1: "Low (1)"}

        r1, r2 = st.columns([2, 1])
        with r1:
            fig_risk = px.bar(
                risk_df.sort_values("composite_risk_score", ascending=False),
                x="state", y="composite_risk_score",
                color="risk_tier",
                color_discrete_map={3: RED, 2: ORANGE, 1: GREEN},
                labels={"composite_risk_score": "Composite Risk Score",
                        "state": "State", "risk_tier": "Tier"},
                title="State Risk Score — computed from disaster frequency, cost, and diversity",
                category_orders={"risk_tier": [3, 2, 1]},
            )
            fig_risk.update_layout(height=400, margin=dict(t=60))
            st.plotly_chart(fig_risk, use_container_width=True)

        with r2:
            st.markdown("**Score components**")
            display_risk = risk_df[[
                "state", "disaster_count", "mean_obligated_cost",
                "incident_diversity", "composite_risk_score", "risk_tier"
            ]].copy()
            display_risk["risk_tier"] = display_risk["risk_tier"].map(tier_labels)
            display_risk["mean_obligated_cost"] = display_risk["mean_obligated_cost"].apply(
                lambda v: fmt_dollars(v) if pd.notna(v) else "—"
            )
            display_risk["composite_risk_score"] = display_risk[
                "composite_risk_score"
            ].apply(lambda v: f"{v:.3f}")
            display_risk.columns = [
                "State", "# Disasters", "Mean Cost", "Incident Types",
                "Composite Score", "Risk Tier"
            ]
            st.dataframe(
                display_risk.sort_values("Composite Score", ascending=False),
                use_container_width=True, hide_index=True,
            )
