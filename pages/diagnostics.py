import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.data_generator import get_feature_columns
from models.predictor_model import SupplyChainPredictor


def render():
    st.title("Model Diagnostics")
    st.markdown("Evaluate accuracy, calibration, and feature importance.")

    if "df" not in st.session_state:
        st.warning("Please go to Dashboard first to load the data.")
        return

    df = st.session_state["df"]

    model_name = st.selectbox("Select Model", ["Gradient Boosting", "Random Forest", "Logistic Regression"])

    cache_key = "model_" + model_name
    if cache_key not in st.session_state:
        with st.spinner("Training " + model_name + "..."):
            m = SupplyChainPredictor(model_name=model_name)
            m.train(df, get_feature_columns())
            st.session_state[cache_key] = m

    model = st.session_state[cache_key]
    metrics = model.metrics

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    cr = metrics["classification_report"]
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col2.metric("Avg Precision", f"{metrics['avg_precision']:.4f}")

    class_1 = cr.get("1", cr.get(1, {}))
    col3.metric("Precision (disruption)", f"{class_1.get('precision', 0):.3f}")
    col4.metric("Recall (disruption)", f"{class_1.get('recall', 0):.3f}")

    st.markdown("---")
    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        labels = ["Normal (0)", "Disrupted (1)"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            text=cm, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=60),
                          xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

    with diag_c2:
        st.subheader("Calibration Curve")
        try:
            frac_pos, mean_pred = model.get_calibration_data()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      name="Perfect calibration",
                                      line=dict(dash="dash", color="gray")))
            fig2.add_trace(go.Scatter(x=mean_pred, y=frac_pos,
                                      mode="lines+markers", name=model_name,
                                      line=dict(color="#4361ee", width=2),
                                      marker=dict(size=8)))
            fig2.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=40),
                                xaxis_title="Mean Predicted Probability",
                                yaxis_title="Fraction of Positives")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning("Calibration plot unavailable: " + str(e))

    st.markdown("---")
    st.subheader("Feature Importance")
    imp = model.feature_importance
    imp_df = pd.DataFrame({"Feature": list(imp.keys()), "Importance": list(imp.values())})
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)
    fig3 = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Blues")
    fig3.update_layout(height=400, margin=dict(l=20, r=20, t=10, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Classification Report")
    rows = []
    for k, v in cr.items():
        if isinstance(v, dict):
            rows.append({"Class": str(k),
                         "Precision": round(v.get("precision", 0), 3),
                         "Recall": round(v.get("recall", 0), 3),
                         "F1-Score": round(v.get("f1-score", 0), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
