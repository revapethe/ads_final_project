import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False


class SupplyChainPredictor:

    MODEL_OPTIONS = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }

    def __init__(self, model_name="Gradient Boosting"):
        self.model_name = model_name
        self.pipeline = None
        self.feature_cols = None
        self.is_trained = False
        self.metrics = {}
        self.feature_importance = {}
        self._y_test = None
        self._y_prob = None

    def train(self, df, feature_cols, target_col="disruption"):
        self.feature_cols = feature_cols
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if HAS_SMOTE:
            try:
                sm = SMOTE(random_state=42, k_neighbors=5)
                X_res, y_res = sm.fit_resample(X_train, y_train)
            except Exception:
                X_res, y_res = X_train, y_train
        else:
            X_res, y_res = X_train, y_train

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)
        X_test_scaled = scaler.transform(X_test)

        base_model = self.MODEL_OPTIONS[self.model_name]
        calibrated = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
        calibrated.fit(X_scaled, y_res)

        self.pipeline = (scaler, calibrated)
        self.is_trained = True

        y_pred = calibrated.predict(X_test_scaled)
        y_prob = calibrated.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        self.metrics = {
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "avg_precision": round(average_precision_score(y_test, y_prob), 4),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        try:
            base_est = calibrated.calibrated_classifiers_[0].estimator
            importances = base_est.feature_importances_
            self.feature_importance = dict(zip(feature_cols, importances))
        except Exception:
            try:
                coef = np.abs(calibrated.calibrated_classifiers_[0].estimator.coef_[0])
                self.feature_importance = dict(zip(feature_cols, coef))
            except Exception:
                self.feature_importance = {col: 0 for col in feature_cols}

        self._y_test = y_test
        self._y_prob = y_prob
        return self.metrics

    def predict(self, record):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        scaler, model = self.pipeline
        X = pd.DataFrame([record])[self.feature_cols].fillna(0)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        label = "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.35 else "LOW"
        return {"probability": float(prob), "risk_level": label}

    def predict_batch(self, df):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        scaler, model = self.pipeline
        X = df[self.feature_cols].fillna(0)
        X_scaled = scaler.transform(X)
        return model.predict_proba(X_scaled)[:, 1]

    def get_calibration_data(self):
        return calibration_curve(self._y_test, self._y_prob, n_bins=10)
