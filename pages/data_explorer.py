import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px


def render():
    st.title("Data Explorer")
    st.markdown("Explore the synthetic supply-chain dataset and underlying distributions.")

    if "df" not in st.session_state:
        st.warning("Please go to Dashboard first to load the data.")
        return

    df = st.session_state["df"]

    col1, col2, col3 = st.columns(3)
    with col1:
        suppliers = ["All"] + sorted(df["supplier_id"].unique().tolist())
        sel_sup = st.selectbox("Supplier", suppliers)
    with col2:
        show_disrupted = st.selectbox("Filter by disruption", ["All", "Disrupted only", "Normal only"])
    with col3:
        n_rows = st.slider("Rows to display", 20, 200, 50)

    filtered = df.copy()
    if sel_sup != "All":
        filtered = filtered[filtered["supplier_id"] == sel_sup]
    if show_disrupted == "Disrupted only":
        filtered = filtered[filtered["disruption"] == 1]
    elif show_disrupted == "Normal only":
        filtered = filtered[filtered["disruption"] == 0]

    st.markdown(f"**{len(filtered):,} records** after filtering")
    st.dataframe(filtered.tail(n_rows).reset_index(drop=True), use_container_width=True, height=300)

    st.markdown("---")
    st.subheader("Feature Distributions - Disrupted vs Normal")
    numeric_cols = ["shipment_delay_hours", "inventory_level_pct", "lead_time_days",
                    "demand_index", "transport_stress", "weather_risk",
                    "supplier_reliability", "composite_risk"]
    sel_col = st.selectbox("Select feature", numeric_cols)

    fig = px.histogram(df, x=sel_col,
                       color=df["disruption"].map({0: "Normal", 1: "Disrupted"}),
                       barmode="overlay", opacity=0.7, nbins=40,
                       color_discrete_map={"Normal": "#4361ee", "Disrupted": "#e63946"},
                       labels={sel_col: sel_col, "color": "Class"})
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = df[numeric_cols + ["disruption"]].corr()
    fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto")
    fig2.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Time-Series View")
    ts_col = st.selectbox("Feature for time series", numeric_cols, index=2)
    agg = df.groupby("date")[ts_col].mean().reset_index()
    fig3 = px.line(agg, x="date", y=ts_col, labels={"date": "Date", ts_col: ts_col})
    fig3.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig3, use_container_width=True)
