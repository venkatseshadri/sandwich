"""
Sandwich Signal API

Loads trained models and emits probability vectors for new market snapshots.

Usage:
    engine = SandwichSignalEngine()
    signal = engine.predict(features_dict)
    # signal = {"timestamp": "...", "probabilities": {label: prob, ...}, "model_metadata": {...}}
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd


class SandwichSignalEngine:
    """
    Loads trained models and emits probability vectors for new market snapshots.

    Usage:
        engine = SandwichSignalEngine()
        signal = engine.predict(features_dict)
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"
        self.data_dir = Path(data_dir)

        self.models = {}
        self.medians = {}
        self.feature_cols = None
        self.metadata = {}

        with open(self.data_dir / "step05_feature_columns.json") as f:
            self.feature_cols = json.load(f)

        for label in ["wont_crash_60", "wont_rip_60"]:
            model_path = self.data_dir / f"step05_model_{label}.pkl"
            medians_path = self.data_dir / f"step05_medians_{label}.json"
            eval_path = self.data_dir / f"step05_eval_{label}.json"

            with open(model_path, "rb") as f:
                self.models[label] = pickle.load(f)
            with open(medians_path) as f:
                self.medians[label] = json.load(f)
            with open(eval_path) as f:
                self.metadata[label] = json.load(f)

    def _prepare_row(self, features_dict, label):
        """
        Build a single-row DataFrame for prediction.
        Returns (DataFrame, list_of_imputed_feature_names).
        """
        row = {}
        imputed = []
        med = self.medians[label]
        for fc in self.feature_cols:
            val = features_dict.get(fc)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[fc] = med.get(fc, 0.0)
                imputed.append(fc)
            else:
                row[fc] = val
        return pd.DataFrame([row]), imputed

    def predict(self, features_dict, timestamp=None):
        """Predict single snapshot. Returns signal dict."""
        probs = {}
        imputed_per_label = {}
        for label in ["wont_crash_60", "wont_rip_60"]:
            row_df, imputed = self._prepare_row(features_dict, label)
            prob = float(self.models[label].predict_proba(row_df)[:, 1][0])
            probs[label] = round(prob, 4)
            imputed_per_label[label] = imputed

        imputed_features = imputed_per_label["wont_crash_60"]

        sample_meta = self.metadata.get("wont_crash_60", {})
        return {
            "timestamp": timestamp,
            "probabilities": probs,
            "model_metadata": {
                "trained_on_n": sample_meta.get("actual_train_n", 0),
                "test_n": sample_meta.get("test_n", 0),
                "untrustworthy": sample_meta.get("untrustworthy", True),
                "min_trusted_train": sample_meta.get("min_trusted_train"),
                "features_imputed": imputed_features,
                "n_features_imputed": len(imputed_features),
                "n_features_total": len(self.feature_cols),
            },
        }

    def predict_batch(self, features_df):
        """Predict batch. Returns DataFrame with probability columns."""
        result = features_df.copy()
        for label in ["wont_crash_60", "wont_rip_60"]:
            med = self.medians[label]
            X = result[self.feature_cols].fillna(pd.Series(med)).fillna(0)
            result[f"prob_{label}"] = self.models[label].predict_proba(X)[:, 1]
        return result
