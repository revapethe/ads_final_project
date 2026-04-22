import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import generate_supply_chain_data, get_feature_columns
from models.predictor_model import SupplyChainPredictor


def render():
    st.markdown('<h1 style="color:#ffffff; font-weight:700;">Supply Chain Collapse Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#cccccc;">AI-powered disruption detection - Phase 1 to 6 Research Pipeline</p>', unsafe_allow_html=True)
    st.markdown("---")

    if "df" not in st.session_state:
        with st.spinner("Generating dataset, please wait..."):
            st.session_state["df"] = generate_supply_chain_data(n_days=730)

    if "model" not in st.session_state:
        with st.spinner("Training model, please wait..."):
            m = SupplyChainPredictor(model_name="Gradient Boosting")
            m.train(st.session_state["df"], get_feature_columns())
            st.session_state["model"] = m

    df = st.session_state["df"]
    model = st.session_state["model"]

    recent = df.tail(90).copy()
    recent["predicted_prob"] = model.predict_batch(recent)
    recent["risk_level"] = recent["predicted_prob"].apply(
        lambda p: "HIGH" if p > 0.6 else ("MEDIUM" if p > 0.35 else "LOW")
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        n_disruptions = int(df["disruption"].sum())
        st.metric("Historical Disruptions", n_disruptions, delta=f"{n_disruptions/len(df)*100:.1f}% rate")
    with col3:
        high_risk = int((recent["risk_level"] == "HIGH").sum())
        st.metric("High-Risk Days (90d)", high_risk)
    with col4:
        auc = model.metrics.get("roc_auc", 0)
        st.metric("Model ROC-AUC", f"{auc:.3f}")

    st.markdown("---")
    st.subheader("Disruption Risk Probability - Last 90 Days")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["predicted_prob"],
        mode="lines+markers", name="Risk Probability",
        line=dict(color="#4361ee", width=2),
        marker=dict(size=4),
    ))
    fig.add_hrect(y0=0.6, y1=1.0, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.35, y1=0.6, fillcolor="orange", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.35, fillcolor="green", opacity=0.05, line_width=0)
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="HIGH threshold")
    fig.add_hline(y=0.35, line_dash="dash", line_color="orange", annotation_text="MEDIUM threshold")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=20),
                      xaxis_title="Date", yaxis_title="Disruption Probability",
                      yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Level Breakdown (90 days)")
        counts = recent["risk_level"].value_counts()
        colors = {"HIGH": "#e63946", "MEDIUM": "#f4a261", "LOW": "#2a9d8f"}
        fig2 = px.pie(values=counts.values, names=counts.index,
                      color=counts.index, color_discrete_map=colors, hole=0.45)
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Composite Risk by Month")
        df["month_label"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
        monthly = df.groupby("month_label")["composite_risk"].mean().reset_index()
        fig3 = px.bar(monthly, x="month_label", y="composite_risk",
                      color="composite_risk", color_continuous_scale="RdYlGn_r",
                      labels={"composite_risk": "Risk Score", "month_label": "Month"})
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Active Alerts")
    alerts = recent[recent["risk_level"] == "HIGH"].tail(5)
    if alerts.empty:
        st.success("No high-risk days in the latest window. Supply chain appears stable.")
    else:
        for _, row in alerts.iterrows():
            st.markdown(
                '<div style="background:#b0e0e6; border-left:5px solid #f4a261; border-radius:8px; padding:12px; margin:8px 0; color:#000000;">'
                + "<b>" + row["date"].strftime("%Y-%m-%d") + "</b> - "
                + "Disruption probability: <b>" + f"{row['predicted_prob']:.1%}" + "</b> | "
                + "Inventory: " + f"{row['inventory_level_pct']:.1f}" + "% | "
                + "Delay: " + f"{row['shipment_delay_hours']:.1f}" + "h | "
                + "Supplier: " + str(row["supplier_id"])
                + "</div>",
                unsafe_allow_html=True,
            )
