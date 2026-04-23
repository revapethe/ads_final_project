# Supply Chain Collapse Predictor

> An AI-powered early warning system that predicts supply chain disruptions before they propagate — using historical logistics data, inventory signals, and external risk indicators.

---

## The Problem

Modern supply chains operate as complex, interconnected systems. A missed delivery at a Tier-3 component manufacturer can cascade into a stockout at a retail location thousands of miles away in under two weeks. Organizations like Amazon and Walmart process millions of shipments daily, and small disturbances can escalate into large-scale financial consequences.

The challenge is not whether disruptions happen. They happen constantly. The challenge is whether they can be seen coming — and whether the systems built to detect them can actually be trusted.

This project investigates whether supply chain disruptions can be predicted early enough for decision-makers to act — before the shelves go empty.

---

## What This Application Does

- Ingests 730 days of supply chain records and constructs temporal features that capture trajectory, not just current state
- Trains a calibrated disruption-probability model that outputs honest confidence scores, not just predictions
- Runs a separate drift detection module that monitors whether the model's assumptions still hold in the real world
- Exposes a counterfactual confound test to detect whether the model is tracking seasonal patterns instead of genuine operational stress
- Persists all results to a local cache so the dashboard loads instantly on every run after the first
- Records every user prediction and reflects it back on the dashboard

---

## The Silent Failure Mode

The most dangerous failure this system is designed to catch is not model error. It is misplaced confidence.

A model trained on historical logistics data will learn that October correlates with elevated risk — because October is a high-demand month. It will raise its disruption flag. The flag is technically coherent. And the flag may be wrong, because the model cannot easily distinguish between genuine operational stress and routine seasonal fluctuation that pattern-matches to historical crisis signatures.

The counterfactual test catches this: remove the time-of-year features and watch whether the predictions collapse. If they do, the model was tracking the season, not the mechanism. A prediction that dissolves when you remove month and quarter was never real evidence.

---

## Application Pages

| Page | What It Does |
|------|-------------|
| Dashboard | Executive overview with KPI metrics, 90-day risk timeline, alerts, and recent user predictions loaded from cache |
| Data Explorer | Browse and filter the dataset, view feature distributions by disruption status, correlation matrix, and time series |
| Predict Disruption | Enter current supply chain conditions manually and get an instant risk probability with gauge chart output |
| Drift Detection | Run adversarial validation, PSI analysis per feature, and counterfactual confound testing |
| Model Diagnostics | Compare all three models with confusion matrix, calibration curve, and feature importance |
| About | Full project description and key concept definitions |

---

## How the Model Works

### Data Generation

The dataset is synthetically generated using NumPy with a fixed random seed of 42, which means results are identical and reproducible on every run. For each of 730 days the generator produces:

- Shipment delay in hours with seasonal amplification
- Inventory level as a percentage of capacity
- Lead time in days
- Demand index with holiday season peaks
- Transport stress score
- Weather risk score
- Supplier reliability score

Approximately 4% of days are injected with disruption events by amplifying risk features: delays multiply 2.5 to 5 times, inventory drops to 10 to 40% of normal, and supplier reliability falls to 30 to 60% of baseline.

Temporal features are then added including 7-day rolling averages and 3-day velocity measures for key signals.

### Training Pipeline

```
Raw features
    -> StandardScaler
    -> SMOTE oversampling (handles 4% class imbalance)
    -> Base classifier (Gradient Boosting / Random Forest / Logistic Regression)
    -> CalibratedClassifierCV with isotonic regression
    -> Calibrated probability output P(disruption)
```

### Calibration

Raw classifier scores do not always correspond to true observed frequencies. Isotonic calibration ensures that when the model outputs 70%, disruptions actually occur roughly 70% of the time on that prediction class. A model that is accurate but not calibrated produces confidence scores that are decorative.

### Drift Detection

Two mechanisms monitor whether the model is still reliable:

**Adversarial Validation** — Label training data as class 0 and evaluation data as class 1. Train a classifier to distinguish them. AUC near 0.5 means distributions are similar. AUC above 0.75 means significant drift has been detected and predictions may be unreliable.

**Population Stability Index** — Measures per-feature distribution shift between a reference window and current data. Below 0.1 is stable. Above 0.25 indicates significant shift requiring investigation or retraining.



## Dependencies

```
streamlit>=1.32.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
```

---

## Sample Prediction Scenarios

| Scenario | Shipment Delay | Inventory | Supplier Reliability | Expected Output |
|----------|---------------|-----------|----------------------|-----------------|
| Normal operations | 8 hours | 65% | 78 | LOW risk, 10 to 20% |
| Stressed network | 40 hours | 30% | 50 | MEDIUM risk, 40 to 55% |
| Crisis conditions | 90 hours | 10% | 15 | HIGH risk, above 80% |

---

## Project Phases

**Phase 1 - Problem Definition**
Supply chains exhibit time dependence, network propagation, nonlinear interactions, and rare-event dynamics. Traditional regression assumes independence between observations, which does not hold in real logistics systems.

**Phase 2 - Uniqueness and Differentiation**
The system integrates stochastic reasoning, causal inference, adversarial validation, and counterfactual stress testing. A distinguishing feature is adversarial validation for detecting dataset shift, which tests whether learned patterns remain valid under changing conditions.

**Phase 3 - Pre-Technical Evaluation**
Model performance is evaluated against operational ground truth. The central focus is the identification of silent failure modes, specifically high-confidence predictions driven by seasonal correlation rather than genuine disruption mechanism.

**Phase 4 - Data Ethics and Synthetic Quality**
Synthetic disruption scenarios are validated by comparing statistical distributions and predictive behavior against real-world baselines. Predictions are presented with uncertainty estimates to prevent automation bias.

**Phase 5 - Implementation Realism**
The minimum viable pipeline includes data ingestion, temporal feature construction, disruption prediction, drift detection, and diagnostic visualization. Runs on a standard laptop without specialized hardware.

**Phase 6 - Quality and Portfolio Readiness**
The system represents a deployable prototype for predictive risk monitoring. Emphasis is on computational skepticism through explicit causal assumptions, distribution shift detection, and transparent failure analysis.

