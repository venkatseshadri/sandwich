"""
Sandwich Step 5 — Model Training

Trains GradientBoostingClassifier with calibration per label.
Time-series walk-forward split. Outputs model pickles + metadata.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "step03_features_and_labels.parquet"

GBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}

LABEL_COLS = ["wont_crash_60", "wont_rip_60"]

FEATURE_COLS = [
    "trend_ema_5",
    "trend_ema_20",
    "trend_ema_50",
    "trend_spot_vs_ema5",
    "trend_spot_vs_ema20",
    "trend_spot_vs_ema50",
    "trend_ema5_slope",
    "trend_ema20_slope",
    "trend_higher_high_20",
    "trend_lower_low_20",
    "trend_above_vwap",
    "trend_vwap_distance",
    "mom_rsi_14",
    "mom_rsi_14_slope",
    "mom_roc_5",
    "mom_roc_15",
    "mom_roc_60",
    "mom_consecutive_up",
    "vol_atr_14",
    "vol_atr_14_pct",
    "vol_realized_30",
    "vol_vix",
    "vol_vix_change_1d",
    "time_minute_of_day",
    "time_minutes_since_open",
    "time_minutes_to_close",
    "time_day_of_week",
    "expiry_dte",
    "expiry_is_dte0",
    "expiry_is_dte_le2",
    "opt_atm_iv",
    "opt_atm_pe_iv",
    "opt_iv_skew",
    "opt_pcr_oi",
    "opt_total_oi_change_5m",
    "opt_max_pain_offset",
    "opt_atm_ce_oi_change_5m",
    "opt_atm_pe_oi_change_5m",
    "regime_vix_low",
    "regime_vix_mid",
    "regime_vix_high",
]


def load_dataset():
    df = pd.read_parquet(FEATURES_PATH)
    for lc in LABEL_COLS:
        df = df[df[lc] != "IN_FLIGHT"].copy()
    return df


def prepare_xy(df, label_col):
    X_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[X_cols].copy()
    y_raw = df[label_col]
    y = (y_raw == "TRUE").astype(int)
    timestamps = df["timestamp"]

    # Impute NaN with column median (from training split)
    medians = X.median()
    X = X.fillna(medians)

    return X, y, timestamps, X_cols, medians


def time_split(X, y, timestamps, train_frac=0.7):
    n = len(X)
    split_idx = int(n * train_frac)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    split_ts = timestamps.iloc[split_idx] if split_idx < n else None
    return X_train, X_test, y_train, y_test, split_ts


def train_one(X_train, y_train):
    if y_train.nunique() < 2:
        print("  Cannot train: only 1 class in training set")
        return None
    gbm = GradientBoostingClassifier(**GBM_PARAMS)
    model = CalibratedClassifierCV(gbm, method="isotonic", cv=3)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

    result = {
        "test_n": len(X_test),
        "test_pos_rate": round(float(y_test.mean()), 4),
    }

    if model is None or len(y_test) < 2 or y_test.nunique() < 2:
        result.update(
            {"logloss": None, "brier": None, "auc": None, "top_decile_precision": None}
        )
        return result

    y_prob = model.predict_proba(X_test)[:, 1]

    result["logloss"] = round(log_loss(y_test, y_prob), 4)
    result["brier"] = round(brier_score_loss(y_test, y_prob), 4)
    result["auc"] = round(roc_auc_score(y_test, y_prob), 4)

    # Top decile precision
    cutoff = np.quantile(y_prob, 0.9)
    top_mask = y_prob >= cutoff
    if top_mask.sum() > 0:
        result["top_decile_precision"] = round(y_test[top_mask].mean(), 4)
    else:
        result["top_decile_precision"] = None

    return result


def train_all_labels(df, label_cols):
    results = {}
    for lc in label_cols:
        print(f"\n{'=' * 60}")
        print(f"Training: {lc}")
        print(f"{'=' * 60}")

        X, y, timestamps, X_cols, medians = prepare_xy(df, lc)
        print(f"Features: {len(X_cols)} columns, {len(X)} rows")
        print(f"Positive rate: {y.mean():.2%}")

        X_train, X_test, y_train, y_test, split_ts = time_split(X, y, timestamps)
        print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
        print(f"Split: {split_ts}")

        model = train_one(X_train, y_train)
        ev = evaluate(model, X_test, y_test)

        # Persist
        model_path = DATA_DIR / f"step05_model_{lc}.pkl"
        medians_path = DATA_DIR / f"step05_medians_{lc}.json"
        eval_path = DATA_DIR / f"step05_eval_{lc}.json"

        if model is not None:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        with open(medians_path, "w") as f:
            json.dump({k: float(v) for k, v in medians.items()}, f)

        ev["untrustworthy"] = True
        with open(eval_path, "w") as f:
            json.dump(ev, f)

        results[lc] = ev
        print(f"Eval: {json.dumps(ev)}")

    # Feature columns order
    with open(DATA_DIR / "step05_feature_columns.json", "w") as f:
        json.dump(X_cols, f)

    return results


def calibration_check(model, X_test, y_test, label_name):
    if model is None or len(X_test) < 10:
        print("Cannot run calibration check")
        return
    y_prob = model.predict_proba(X_test)[:, 1]
    deciles = np.quantile(y_prob, np.linspace(0, 1, 11))
    print(f"\nCalibration check — {label_name}")
    print(f"{'decile':<8} {'n':<6} {'pred_rate':<12} {'obs_rate':<12}")
    for i in range(10):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == 9:
            mask = y_prob >= lo
        n = mask.sum()
        if n == 0:
            continue
        pred_mean = y_prob[mask].mean()
        obs_mean = y_test[mask].mean()
        print(f"{i + 1:<8} {n:<6} {pred_mean:<12.4f} {obs_mean:<12.4f}")


def main():
    print("=" * 60)
    print("Sandwich Step 5 — Model Training")
    print("=" * 60)

    df = load_dataset()
    print(f"Loaded: {len(df):,} rows")
    print(
        f"UNTRUSTWORTHY WARNING: models trained on {len(df)} samples. Metrics below are NOT inference-valid. Pipeline correctness is what's being verified.\n"
    )

    date_col = "date_feat" if "date_feat" in df.columns else "date"
    train_n = int(len(df) * 0.7)
    print(f"Total rows: {len(df)}")
    print(
        f"Train: {train_n} rows, dates {df[date_col].iloc[0]} to {df[date_col].iloc[train_n - 1]}"
    )
    print(
        f"Test: {len(df) - train_n} rows, dates {df[date_col].iloc[train_n]} to {df[date_col].iloc[-1]}"
    )

    results = train_all_labels(df, LABEL_COLS)

    # Calibration check for wont_crash_60
    X, y, timestamps, X_cols, medians = prepare_xy(df, "wont_crash_60")
    X_train, X_test, y_train, y_test, _ = time_split(X, y, timestamps)
    with open(DATA_DIR / "step05_model_wont_crash_60.pkl", "rb") as f:
        model = pickle.load(f)
    calibration_check(model, X_test, y_test, "wont_crash_60")

    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
