import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import plotly.graph_objects as go

RISK_COLOR = {"HIGH": "#e63946", "MEDIUM": "#f4a261", "LOW": "#2a9d8f"}
MODEL_OPTIONS = ["Gradient Boosting", "Random Forest", "Logistic Regression"]


def render():
    st.title("Predict Disruption")
    st.markdown("Enter current supply-chain metrics to get a real-time disruption probability estimate.")

    if "df" not in st.session_state or "model" not in st.session_state:
        st.warning("Please go to Dashboard first to load the data and train the model.")
        return

    model = st.session_state["model"]

    st.markdown("---")
    st.subheader("Input Current Conditions")

    c1, c2, c3 = st.columns(3)
    with c1:
        shipment_delay = st.slider("Shipment Delay (hours)", 0.0, 120.0, 8.0, 0.5)
        inventory_level = st.slider("Inventory Level (%)", 0.0, 100.0, 65.0, 1.0)
        lead_time = st.slider("Lead Time (days)", 1.0, 30.0, 7.0, 0.5)
        demand_index = st.slider("Demand Index", 50.0, 200.0, 100.0, 1.0)
    with c2:
        transport_stress = st.slider("Transport Stress (0-100)", 0.0, 100.0, 40.0, 1.0)
        weather_risk = st.slider("Weather Risk (0-100)", 0.0, 100.0, 20.0, 1.0)
        supplier_rel = st.slider("Supplier Reliability (0-100)", 0.0, 100.0, 78.0, 1.0)
        month = st.selectbox("Month", list(range(1, 13)), index=0)
    with c3:
        delay_7d = st.number_input("Shipment Delay 7-day avg", 0.0, 120.0, shipment_delay)
        inv_7d = st.number_input("Inventory 7-day avg (%)", 0.0, 100.0, inventory_level)
        lead_7d = st.number_input("Lead Time 7-day avg", 0.0, 30.0, lead_time)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)

    record = {
        "shipment_delay_hours": shipment_delay,
        "inventory_level_pct": inventory_level,
        "lead_time_days": lead_time,
        "demand_index": demand_index,
        "transport_stress": transport_stress,
        "weather_risk": weather_risk,
        "supplier_reliability": supplier_rel,
        "shipment_delay_hours_7d_avg": delay_7d,
        "inventory_level_pct_7d_avg": inv_7d,
        "lead_time_days_7d_avg": lead_7d,
        "shipment_delay_hours_velocity": shipment_delay - delay_7d,
        "inventory_level_pct_velocity": inventory_level - inv_7d,
        "lead_time_days_velocity": lead_time - lead_7d,
        "day_of_week": 0,
        "month": month,
        "quarter": quarter,
        "is_holiday_qtr": int(quarter == 4),
    }

    if st.button("Run Prediction", type="primary"):
        result = model.predict(record)
        prob = result["probability"]
        level = result["risk_level"]
        color = RISK_COLOR[level]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_gauge, col_detail = st.columns([1, 1])
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 40}},
                title={"text": "Disruption Risk: " + level, "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 35], "color": "#d4edda"},
                        {"range": [35, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": prob * 100,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_detail:
            st.markdown("**Disruption Probability:** " + f"{prob:.1%}")
            st.markdown("**Risk Level:** " + level)
            st.markdown("**Model Used:** " + model.model_name)

            if level == "HIGH":
                st.error("Action Required: Probability exceeds 60% threshold. Consider activating backup suppliers.")
            elif level == "MEDIUM":
                st.warning("Monitor Closely: Moderate disruption risk. Review incoming shipments.")
            else:
                st.success("Stable: Low disruption probability. No immediate action required.")

            st.markdown("---")
            st.markdown("**Top risk drivers:**")
            importance = model.feature_importance
            if importance:
                top5 = sorted(importance, key=importance.get, reverse=True)[:5]
                for feat in top5:
                    st.markdown("- " + feat + ": " + f"{importance[feat]:.3f}")
