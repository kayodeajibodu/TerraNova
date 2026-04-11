"""
Terra Nova — Disaster Recovery Cost Forecasting Dashboard
Streamlit frontend connected to the FastAPI backend.

Run with:
    streamlit run dashboard/app.py
"""

import os
from datetime import date, timedelta
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

PASTEL_BLUE   = "#4A90D9"
PASTEL_ORANGE = "#F5A623"
PASTEL_RED    = "#E74C3C"
PASTEL_GREEN  = "#2ECC71"

st.set_page_config(
    page_title="Terra Nova | Disaster Cost Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_dollars(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}K"
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


def api_feature_importance() -> Optional[list[dict]]:
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
        return {"status": "unreachable", "model_loaded": False}


# ---------------------------------------------------------------------------
# Sidebar — API status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://em-content.zobj.net/source/google/387/globe-showing-americas_1f30e.png", width=60)
    st.title("Terra Nova")
    st.caption("Disaster Recovery Cost Forecasting")
    st.divider()

    health = api_health()
    status_color = "green" if health.get("model_loaded") else "red"
    st.markdown(
        f"**API status:** :{status_color}[{'● Online' if health.get('model_loaded') else '● Offline'}]"
    )
    st.caption(f"Backend: `{API_URL}`")
    st.divider()

    st.markdown("**Navigation**")
    page = st.radio(
        label="page",
        options=["Cost Forecast", "Scenario Simulation", "Feature Importance", "Budget Gap Analysis"],
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
        declaration_type = st.selectbox("Declaration Type", ["DR", "EM", "FM"], index=0)

    with col2:
        begin_date    = st.date_input("Incident Begin Date",  value=date.today() - timedelta(days=14))
        end_date_raw  = st.date_input("Incident End Date",    value=date.today())
        decl_date     = st.date_input("Declaration Date",     value=date.today())
        proj_count    = st.number_input("Estimated PA Projects", min_value=0, value=50, step=10)
        county_scope  = st.number_input("Designated Counties",   min_value=1, value=5,  step=1)

    if st.button("Generate Forecast", type="primary", use_container_width=True):
        payload = {
            "disaster_number":         int(disaster_number),
            "state":                   state,
            "incident_type":           incident_type,
            "declaration_type":        declaration_type,
            "incident_begin_date":     begin_date.isoformat(),
            "incident_end_date":       end_date_raw.isoformat(),
            "declaration_date":        decl_date.isoformat(),
            "project_count_estimate":  int(proj_count),
            "county_scope_estimate":   int(county_scope),
        }

        with st.spinner("Forecasting …"):
            result = api_predict(payload)

        if result:
            st.success("Forecast generated")

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Cost",  fmt_dollars(result["predicted_cost_dollars"]))
            m2.metric("Lower Bound (80%)", fmt_dollars(result["lower_bound_dollars"]))
            m3.metric("Upper Bound (80%)", fmt_dollars(result["upper_bound_dollars"]))

            # Probability distribution chart
            mu    = result["predicted_cost_log"]
            sigma = 0.55
            x     = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 300)
            pdf   = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            x_dollars = np.expm1(x)

            lower_log = np.log1p(result["lower_bound_dollars"])
            upper_log = np.log1p(result["upper_bound_dollars"])
            mask = (x >= lower_log) & (x <= upper_log)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_dollars, y=pdf,
                fill="tozeroy", fillcolor="rgba(74,144,217,0.15)",
                line=dict(color=PASTEL_BLUE, width=2),
                name="Cost distribution",
            ))
            fig.add_trace(go.Scatter(
                x=x_dollars[mask], y=pdf[mask],
                fill="tozeroy", fillcolor="rgba(74,144,217,0.45)",
                line=dict(color=PASTEL_BLUE, width=0),
                name="80% confidence interval",
            ))
            fig.add_vline(
                x=result["predicted_cost_dollars"],
                line=dict(color=PASTEL_RED, width=2, dash="dash"),
                annotation_text=f"Forecast: {fmt_dollars(result['predicted_cost_dollars'])}",
            )
            fig.update_layout(
                title="Recovery Cost Probability Distribution",
                xaxis_title="Total Obligated Amount (USD)",
                yaxis_title="Probability Density",
                showlegend=True,
                height=340,
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 2 — Scenario Simulation
# ---------------------------------------------------------------------------

elif page == "Scenario Simulation":
    st.header("🌡️ Scenario Simulation")
    st.caption("Adjust climate and scale parameters to model cost under different future conditions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Base scenario")
        base_state   = st.selectbox("State", US_STATES, index=US_STATES.index("FL"))
        base_type    = st.selectbox("Incident type", INCIDENT_TYPES, index=INCIDENT_TYPES.index("HURRICANE"))
        base_dur     = st.slider("Incident duration (days)", 1, 60, 10)
        base_scope   = st.slider("Counties affected", 1, 67, 12)
        base_proj    = st.slider("Estimated PA projects", 10, 2000, 200)

        st.subheader("Climate multipliers")
        intensity_mult = st.slider("Intensity scaling (1× = baseline)", 1.0, 3.0, 1.0, 0.1)
        scope_mult     = st.slider("Geographic spread multiplier", 1.0, 3.0, 1.0, 0.1)

    with col2:
        scenarios = {
            "Baseline":             (1.0,  1.0),
            "+50% intensity":       (1.5,  1.0),
            "+100% intensity":      (2.0,  1.0),
            "Wider geographic scope": (1.0, 1.5),
            "Compound (2× / 2×)":  (2.0,  2.0),
            "Custom":               (intensity_mult, scope_mult),
        }

        results_rows = []
        for label, (i_mult, s_mult) in scenarios.items():
            payload = {
                "disaster_number":        9999,
                "state":                  base_state,
                "incident_type":          base_type,
                "declaration_type":       "DR",
                "incident_begin_date":    date.today().isoformat(),
                "incident_end_date":      (date.today() + timedelta(days=int(base_dur * i_mult))).isoformat(),
                "declaration_date":       date.today().isoformat(),
                "project_count_estimate": int(base_proj * i_mult),
                "county_scope_estimate":  int(base_scope * s_mult),
            }
            res = api_predict(payload)
            if res:
                results_rows.append({
                    "Scenario":    label,
                    "Cost ($)":    res["predicted_cost_dollars"],
                    "Lower ($)":   res["lower_bound_dollars"],
                    "Upper ($)":   res["upper_bound_dollars"],
                })

        if results_rows:
            df_s = pd.DataFrame(results_rows)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_s["Scenario"],
                y=df_s["Cost ($)"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=df_s["Upper ($)"] - df_s["Cost ($)"],
                    arrayminus=df_s["Cost ($)"] - df_s["Lower ($)"],
                ),
                marker_color=[PASTEL_BLUE if s != "Custom" else PASTEL_ORANGE for s in df_s["Scenario"]],
                name="Predicted cost",
            ))
            fig.update_layout(
                title="Scenario Cost Comparison",
                yaxis_title="Total Obligated Amount (USD)",
                height=420,
                margin=dict(t=50, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)

            display_df = df_s.copy()
            for col in ["Cost ($)", "Lower ($)", "Upper ($)"]:
                display_df[col] = display_df[col].apply(fmt_dollars)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

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
            df_fi, x="importance", y="feature",
            orientation="h",
            labels={"importance": "Importance Score", "feature": "Feature"},
            color="importance",
            color_continuous_scale=["#B5D4F4", "#4A90D9", "#0C447C"],
            title="Feature importances (active model)",
        )
        fig.update_layout(
            height=480,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature descriptions")
        descriptions = {
            "incident_duration_days":   "Duration from incident start to end.",
            "declaration_lag_days":     "Days from incident start to federal declaration.",
            "project_count":            "Number of approved public assistance projects.",
            "county_scope":             "Number of designated counties — geographic extent.",
            "regional_risk_score":      "State-level historical disaster risk (1=low, 3=high).",
            "incident_type_enc":        "Encoded incident type (hurricane, flood, etc.).",
            "declaration_type_severity":"Severity class of declaration (DR > EM > FM).",
            "declaration_year":         "Year of declaration — captures long-term cost trends.",
            "declaration_month":        "Month — captures seasonal event patterns.",
            "disaster_frequency_5yr":   "Count of prior disasters in same state over 5 years.",
            "has_high_cost_category":   "Whether high-cost project categories (E/F/G) are present.",
            "category_diversity":       "Number of distinct project categories in the disaster.",
            "mean_project_amount":      "Average per-project obligated amount.",
            "max_project_amount":       "Largest single project obligation — extreme event signal.",
        }
        for item in sorted(fi_data, key=lambda x: x["rank"]):
            with st.expander(f"#{item['rank']}  {item['feature']}  — {item['importance']:.4f}"):
                st.write(descriptions.get(item["feature"], "No description available."))
    else:
        st.info("Feature importance unavailable. Ensure the API is running and a tree-based model is active.")

# ---------------------------------------------------------------------------
# Page 4 — Budget Gap Analysis
# ---------------------------------------------------------------------------

elif page == "Budget Gap Analysis":
    st.header("💰 Budget Gap Analysis")
    st.caption("Compare your allocated disaster budget against the forecast cost distribution.")

    col1, col2 = st.columns([1, 2])

    with col1:
        budget_input = st.number_input(
            "Allocated budget (USD millions)",
            min_value=0.0, value=500.0, step=50.0,
        )
        budget = budget_input * 1_000_000

        st.subheader("Forecast scenario")
        gap_state = st.selectbox("State", US_STATES, key="gap_state")
        gap_type  = st.selectbox("Incident type", INCIDENT_TYPES, key="gap_type")
        gap_dur   = st.slider("Duration (days)", 1, 90, 14, key="gap_dur")
        gap_scope = st.slider("Counties affected", 1, 67, 10, key="gap_scope")
        gap_proj  = st.slider("PA projects", 10, 2000, 150, key="gap_proj")

        run = st.button("Run Analysis", type="primary", use_container_width=True)

    with col2:
        if run:
            payload = {
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
                result = api_predict(payload)

            if result:
                predicted  = result["predicted_cost_dollars"]
                lower      = result["lower_bound_dollars"]
                upper      = result["upper_bound_dollars"]
                gap        = predicted - budget
                worst_gap  = upper - budget

                st.subheader("Results")

                m1, m2 = st.columns(2)
                m1.metric("Predicted cost",  fmt_dollars(predicted))
                m2.metric(
                    "Budget gap",
                    fmt_dollars(abs(gap)),
                    delta=f"{'Over' if gap > 0 else 'Under'} budget",
                    delta_color="inverse",
                )

                # Gauge chart
                max_val = upper * 1.2
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted,
                    delta={"reference": budget, "valueformat": "$,.0f"},
                    gauge={
                        "axis": {"range": [0, max_val]},
                        "bar": {"color": PASTEL_BLUE},
                        "steps": [
                            {"range": [0, lower],     "color": "rgba(46,204,113,0.25)"},
                            {"range": [lower, upper], "color": "rgba(245,166,35,0.25)"},
                            {"range": [upper, max_val],"color": "rgba(231,76,60,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 3},
                            "thickness": 0.85,
                            "value": budget,
                        },
                    },
                    number={"prefix": "$", "valueformat": ",.0f"},
                    title={"text": "Forecast vs Budget"},
                ))
                fig.update_layout(height=320, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # Risk summary
                if predicted <= budget:
                    st.success(f"✅ Budget appears adequate. Forecast is {fmt_dollars(abs(gap))} under allocation.")
                elif predicted <= budget * 1.25:
                    st.warning(f"⚠️  Forecast slightly exceeds budget by {fmt_dollars(gap)} ({gap/budget*100:.0f}%).")
                else:
                    st.error(
                        f"🚨 Significant budget shortfall: forecast exceeds allocation by "
                        f"{fmt_dollars(gap)} ({gap/budget*100:.0f}%). "
                        f"Worst-case gap: {fmt_dollars(worst_gap)}."
                    )
