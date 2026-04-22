import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class DriftDetector:

    def __init__(self):
        self.results = {}

    def adversarial_validation(self, df_train, df_eval, feature_cols):
        X_train = df_train[feature_cols].copy()
        X_train["_split"] = 0
        X_eval = df_eval[feature_cols].copy()
        X_eval["_split"] = 1
        combined = pd.concat([X_train, X_eval], ignore_index=True).fillna(0)

        X = combined.drop("_split", axis=1)
        y = combined["_split"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        auc_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")
        mean_auc = float(np.mean(auc_scores))

        clf.fit(X_scaled, y)
        importance = dict(zip(feature_cols, clf.feature_importances_))
        top_drifted = sorted(importance, key=importance.get, reverse=True)[:5]

        status = "CRITICAL" if mean_auc > 0.80 else "WARNING" if mean_auc > 0.65 else "STABLE"

        result = {
            "adversarial_auc": round(mean_auc, 4),
            "status": status,
            "top_drifted_features": top_drifted,
            "feature_importance": importance,
        }
        self.results["adversarial"] = result
        return result

    def compute_psi(self, df_ref, df_cur, feature_cols, n_bins=10):
        psi_scores = {}
        for col in feature_cols:
            ref_vals = df_ref[col].dropna().values
            cur_vals = df_cur[col].dropna().values
            psi_scores[col] = round(self._psi(ref_vals, cur_vals, n_bins), 4)

        overall_psi = float(np.mean(list(psi_scores.values())))
        status = "CRITICAL" if overall_psi > 0.25 else "WARNING" if overall_psi > 0.10 else "STABLE"

        result = {
            "psi_per_feature": psi_scores,
            "overall_psi": round(overall_psi, 4),
            "status": status,
        }
        self.results["psi"] = result
        return result

    @staticmethod
    def _psi(ref, cur, n_bins):
        bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        ref_pct = np.histogram(ref, bins=bins)[0] / len(ref)
        cur_pct = np.histogram(cur, bins=bins)[0] / len(cur)
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def counterfactual_test(self, df, feature_cols, suspect_cols, predictor):
        y_full = predictor.predict_batch(df)
        reduced_cols = [c for c in feature_cols if c not in suspect_cols]
        original_cols = predictor.feature_cols
        predictor.feature_cols = reduced_cols
        y_reduced = predictor.predict_batch(df)
        predictor.feature_cols = original_cols

        drop = float(np.mean(y_full) - np.mean(y_reduced))
        relative_drop = drop / (np.mean(y_full) + 1e-9)

        return {
            "mean_prob_full": round(float(np.mean(y_full)), 4),
            "mean_prob_reduced": round(float(np.mean(y_reduced)), 4),
            "absolute_drop": round(drop, 4),
            "relative_drop_pct": round(relative_drop * 100, 1),
            "confound_detected": abs(relative_drop) > 0.25,
        }
