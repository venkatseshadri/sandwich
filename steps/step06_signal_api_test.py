"""
Sandwich Step 6 — Signal API Test Harness

Runs the SandwichSignalEngine against historical test data and produces
a trade log + summary statistics.
"""

from pathlib import Path
import sys
import json

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signal_api import SandwichSignalEngine

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "step03_features_and_labels.parquet"


def main():
    print("=" * 60)
    print("Sandwich Step 6 — Signal API Test")
    print("=" * 60)

    # Load data
    df = pd.read_parquet(FEATURES_PATH)
    for lc in ["wont_crash_60", "wont_rip_60"]:
        df = df[df[lc] != "IN_FLIGHT"].copy()

    # Time split (match Step 5)
    n_train = int(len(df) * 0.7)
    test_df = df.iloc[n_train:].copy()
    print(f"Test set: {len(test_df):,} rows")

    # Instantiate engine
    engine = SandwichSignalEngine()

    # Smoke test: single prediction
    feature_cols = engine.feature_cols
    sample_row = test_df.iloc[0]
    features_dict = {
        fc: float(sample_row[fc]) if not pd.isna(sample_row[fc]) else None
        for fc in feature_cols
    }
    signal = engine.predict(
        features_dict,
        timestamp=str(
            sample_row.get("timestamp_feat", sample_row.get("timestamp", ""))
        ),
    )
    print(f"\nAPI smoke test:")
    print(f"  prediction: {json.dumps(signal['probabilities'])}")
    print(f"  untrustworthy: {signal['model_metadata']['untrustworthy']}")

    # Batch predict
    print(f"\nRunning batch predict on {len(test_df)} rows...")
    pred_df = engine.predict_batch(test_df)

    # Trade log: rows where prob > 0.7
    summary = {}
    for lc in ["wont_crash_60", "wont_rip_60"]:
        prob_col = f"prob_{lc}"
        signal_mask = pred_df[prob_col] > 0.7
        n_fired = signal_mask.sum()
        n_true = (pred_df.loc[signal_mask, lc] == "TRUE").sum() if n_fired > 0 else 0

        base_n_true = (pred_df[lc] == "TRUE").sum()
        base_n = len(pred_df)
        base_rate = base_n_true / base_n if base_n > 0 else 0
        hit_rate = n_true / n_fired if n_fired > 0 else 0

        print(f"\n{lc}:")
        print(f"  Fired (prob > 0.70): {n_fired}")
        print(f"  Actually TRUE: {n_true}")
        print(f"  Hit rate: {hit_rate:.2%}")
        print(f"  Base rate: {base_rate:.2%}")
        summary[lc] = {
            "n_fired": int(n_fired),
            "n_true": int(n_true),
            "hit_rate": round(float(hit_rate), 4),
            "base_rate": round(float(base_rate), 4),
            "total_rows": int(len(pred_df)),
        }

    # Build trade log for wont_crash_60
    log_cols = [
        "timestamp_feat",
        "spot",
        "vol_vix",
        "expiry_dte",
        f"prob_wont_crash_60",
    ]
    ts_col = "timestamp_feat" if "timestamp_feat" in pred_df.columns else "timestamp"
    spot_col = "spot_feat" if "spot_feat" in pred_df.columns else "spot"
    log_cols_actual = [ts_col, spot_col, "vol_vix", "expiry_dte", "prob_wont_crash_60"]
    log_cols_actual = [c for c in log_cols_actual if c in pred_df.columns]

    log_mask = pred_df["prob_wont_crash_60"] > 0.7
    if log_mask.sum() > 0:
        trade_log = pred_df.loc[log_mask, log_cols_actual].copy()
        trade_log["actual_label"] = pred_df.loc[log_mask, "wont_crash_60"].values
        trade_log.to_csv(DATA_DIR / "step06_signal_log.csv", index=False)
        print(f"\nTrade log saved: {len(trade_log)} signals > 0.70")
    else:
        pd.DataFrame(
            columns=[
                "timestamp",
                "spot",
                "vol_vix",
                "expiry_dte",
                "prob",
                "actual_label",
            ]
        ).to_csv(DATA_DIR / "step06_signal_log.csv", index=False)
        print(f"\nTrade log saved: 0 signals > 0.70")

    with open(DATA_DIR / "step06_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: step06_summary.json")
    print("\nEnd-to-end check: sandwich.signal_api imported successfully")

    print("\nPOC STATUS:")
    print("  Sandwich POC end-to-end runnable.")
    print(
        "  Models trained on insufficient data; edge metrics are not inference-valid."
    )
    print("  Pipeline architecture is verified.")
    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()
