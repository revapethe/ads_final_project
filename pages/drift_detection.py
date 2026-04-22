import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_generator import get_feature_columns
from models.drift_detector import DriftDetector


def render():
    st.title("Drift Detection")
    st.markdown("Detect dataset shift using adversarial validation and Population Stability Index.")

    if "df" not in st.session_state or "model" not in st.session_state:
        st.warning("Please go to Dashboard first to load the data.")
        return

    df = st.session_state["df"]
    model = st.session_state["model"]
    feat_cols = get_feature_columns()

    st.subheader("Configure Train / Eval Windows")
    c1, c2 = st.columns(2)
    with c1:
        train_pct = st.slider("Training window (% of data)", 30, 80, 60)
    with c2:
        eval_size = st.slider("Eval window (most recent N days)", 30, 180, 90)

    split_idx = int(len(df) * train_pct / 100)
    df_train = df.iloc[:split_idx]
    df_eval = df.tail(eval_size)

    st.info(f"Training: {len(df_train)} records | Evaluation: {len(df_eval)} records")

    if st.button("Run Drift Analysis", type="primary"):
        detector = DriftDetector()

        with st.spinner("Running adversarial validation..."):
            adv = detector.adversarial_validation(df_train, df_eval, feat_cols)

        with st.spinner("Computing PSI..."):
            psi = detector.compute_psi(df_train, df_eval, feat_cols)

        st.markdown("---")
        st.subheader("Adversarial Validation")
        auc = adv["adversarial_auc"]
        st.markdown("**Classifier AUC (train vs eval):** " + str(auc) + "  |  Status: **" + adv["status"] + "**")

        if adv["status"] == "STABLE":
            st.success("AUC is close to 0.5 - training and evaluation distributions are similar.")
        elif adv["status"] == "WARNING":
            st.warning("AUC above 0.65 - moderate distribution shift detected.")
        else:
            st.error("AUC above 0.80 - significant drift detected. Model predictions may be unreliable.")

        st.markdown("**Top drifted features:**")
        top_imp = {k: adv["feature_importance"][k] for k in adv["top_drifted_features"]}
        fig = px.bar(x=list(top_imp.keys()), y=list(top_imp.values()),
                     labels={"x": "Feature", "y": "Importance"},
                     color=list(top_imp.values()), color_continuous_scale="Reds")
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=60), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Population Stability Index (PSI)")
        overall_psi = psi["overall_psi"]
        st.markdown("**Overall PSI:** " + str(overall_psi) + "  |  Status: **" + psi["status"] + "**")

        psi_df = pd.DataFrame({
            "Feature": list(psi["psi_per_feature"].keys()),
            "PSI": list(psi["psi_per_feature"].values()),
        }).sort_values("PSI", ascending=False)

        def psi_label(v):
            if v >= 0.25:
                return "CRITICAL"
            if v >= 0.10:
                return "WARNING"
            return "STABLE"

        psi_df["Status"] = psi_df["PSI"].apply(psi_label)
        color_map = {"STABLE": "#2a9d8f", "WARNING": "#f4a261", "CRITICAL": "#e63946"}
        fig2 = px.bar(psi_df, x="Feature", y="PSI", color="Status", color_discrete_map=color_map)
        fig2.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="0.25 Critical")
        fig2.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="0.10 Warning")
        fig2.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=80))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Counterfactual Confound Test")
        st.markdown("Remove time-of-year features and check if predictions collapse.")

        suspect = st.multiselect("Suspect confounder columns", feat_cols,
                                 default=["month", "quarter", "is_holiday_qtr", "day_of_week"])

        if st.button("Run Counterfactual Test"):
            cf = detector.counterfactual_test(df_eval, feat_cols, suspect, model)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Mean prob (full features)", f"{cf['mean_prob_full']:.3f}")
                st.metric("Mean prob (confounder removed)", f"{cf['mean_prob_reduced']:.3f}")
            with col_b:
                st.metric("Relative drop", f"{cf['relative_drop_pct']:.1f}%")
                if cf["confound_detected"]:
                    st.error("Confound detected - model may be tracking seasonality, not real disruptions.")
                else:
                    st.success("No strong confound detected.")
