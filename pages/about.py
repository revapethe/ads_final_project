import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st


def render():
    st.title("About This Project")

    st.markdown("""
## Supply Chain Collapse Predictor

**Domain:** Logistics and Operations Research

**Goal:** Predict supply chain disruptions early using historical logistics data, inventory signals, and external risk indicators.

---

### Architecture Overview

| Layer | Component | Technology |
|-------|-----------|-----------|
| Data | Synthetic generation plus temporal features | Python / NumPy / Pandas |
| Model | Disruption probability estimator | Scikit-learn |
| Calibration | Isotonic calibration | CalibratedClassifierCV |
| Imbalance handling | SMOTE oversampling | imbalanced-learn |
| Drift detection | Adversarial validation plus PSI | Scikit-learn |
| Frontend | Interactive dashboard | Streamlit and Plotly |

---

### Project Phases

**Phase 1 - Problem Definition**
Modern supply chains exhibit time dependence, network propagation, and rare-event dynamics. This pipeline models temporal evolution and uncertainty simultaneously.

**Phase 2 - Uniqueness**
Combines stochastic reasoning, causal inference, adversarial validation, and counterfactual stress testing. A second module watches the model itself.

**Phase 3 - Pre-Technical Evaluation**
Silent failure mode: high-confidence predictions driven by seasonal demand variation rather than genuine operational stress. Detected via counterfactual tests.

**Phase 4 - Data and Ethics**
Rare disruptions (3 to 4%) require synthetic oversampling (SMOTE). All predictions include calibrated uncertainty estimates to prevent automation bias.

**Phase 5 - Implementation**
Minimum viable pipeline: data ingestion, temporal features, prediction, drift detection, and diagnostic visualizations. Runs on a standard laptop.

**Phase 6 - Quality**
Explicit causal assumptions, distribution shift detection, synthetic stress testing, and calibration testing.

---

### Key Concepts

**Calibration:** If the model says 70% chance of disruption, disruptions should actually occur roughly 70% of the time on that prediction class.

**Adversarial Validation:** Train a classifier to tell training data from evaluation data. AUC near 0.5 means stable. AUC above 0.75 means significant drift.

**PSI (Population Stability Index):** Per-feature distribution shift measure. Below 0.1 is stable. Above 0.25 is significant shift.

**Strategic Delegation:** The model detects and estimates. The human analyst interprets and decides.

---

### Tech Stack

- Python 3.10+
- streamlit - frontend
- scikit-learn - ML models and calibration
- imbalanced-learn - SMOTE oversampling
- pandas and numpy - data handling
- plotly - interactive charts
    """)
