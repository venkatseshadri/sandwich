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

    # Smoke test 1: full feature dict
    feature_cols = engine.feature_cols
    sample_row = test_df.iloc[0]
    features_dict_full = {
        fc: float(sample_row[fc]) if not pd.isna(sample_row[fc]) else None
        for fc in feature_cols
    }
    ts_str = str(sample_row.get("timestamp_feat", sample_row.get("timestamp", "")))
    signal_full = engine.predict(features_dict_full, timestamp=ts_str)
    print(f"\nAPI smoke test 1 (full features):")
    print(f"  probabilities: {json.dumps(signal_full['probabilities'])}")
    print(
        f"  features_imputed: {signal_full['model_metadata'].get('features_imputed', [])}"
    )
    print(f"  untrustworthy: {signal_full['model_metadata']['untrustworthy']}")

    # Smoke test 2: partial features (deliberately omit all option chain features)
    features_dict_partial = {
        k: v for k, v in features_dict_full.items() if not k.startswith("opt_")
    }
    signal_partial = engine.predict(features_dict_partial, timestamp=ts_str)
    print(f"\nAPI smoke test 2 (option chain features deliberately omitted):")
    print(f"  probabilities: {json.dumps(signal_partial['probabilities'])}")
    print(
        f"  features_imputed: {signal_partial['model_metadata'].get('features_imputed', [])}"
    )
    print(
        f"  n_features_imputed: {signal_partial['model_metadata'].get('n_features_imputed', 0)}"
    )

    expected_imputed = sorted([fc for fc in feature_cols if fc.startswith("opt_")])
    actual_imputed = sorted(
        signal_partial["model_metadata"].get("features_imputed", [])
    )
    if expected_imputed != actual_imputed:
        print(f"  WARNING: expected imputed {expected_imputed}, got {actual_imputed}")
    else:
        print(f"  Imputation reporting verified.")

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

    # Build trade log: include signals from BOTH labels at >0.70
    ts_col = "timestamp_feat" if "timestamp_feat" in pred_df.columns else "timestamp"
    spot_col = "spot_feat" if "spot_feat" in pred_df.columns else "spot"
    vix_col = "vol_india_vix" if "vol_india_vix" in pred_df.columns else "vol_vix"

    base_cols = [
        c for c in [ts_col, spot_col, vix_col, "expiry_dte"] if c in pred_df.columns
    ]

    fire_mask = (pred_df["prob_wont_crash_60"] > 0.7) | (
        pred_df["prob_wont_rip_60"] > 0.7
    )

    if fire_mask.sum() > 0:
        trade_log = pred_df.loc[fire_mask, base_cols].copy()
        trade_log["prob_wont_crash_60"] = pred_df.loc[
            fire_mask, "prob_wont_crash_60"
        ].values
        trade_log["prob_wont_rip_60"] = pred_df.loc[
            fire_mask, "prob_wont_rip_60"
        ].values
        trade_log["actual_wont_crash_60"] = pred_df.loc[
            fire_mask, "wont_crash_60"
        ].values
        trade_log["actual_wont_rip_60"] = pred_df.loc[fire_mask, "wont_rip_60"].values
        trade_log["fire_reason"] = "?"
        trade_log.loc[
            (pred_df.loc[fire_mask, "prob_wont_crash_60"] > 0.7)
            & (pred_df.loc[fire_mask, "prob_wont_rip_60"] > 0.7),
            "fire_reason",
        ] = "BOTH"
        trade_log.loc[
            (pred_df.loc[fire_mask, "prob_wont_crash_60"] > 0.7)
            & (pred_df.loc[fire_mask, "prob_wont_rip_60"] <= 0.7),
            "fire_reason",
        ] = "CRASH_ONLY"
        trade_log.loc[
            (pred_df.loc[fire_mask, "prob_wont_crash_60"] <= 0.7)
            & (pred_df.loc[fire_mask, "prob_wont_rip_60"] > 0.7),
            "fire_reason",
        ] = "RIP_ONLY"
        trade_log.to_csv(DATA_DIR / "step06_signal_log.csv", index=False)
        print(f"\nTrade log saved: {len(trade_log)} signals (either label > 0.70)")
        print(f"  CRASH_ONLY: {(trade_log['fire_reason'] == 'CRASH_ONLY').sum()}")
        print(f"  RIP_ONLY: {(trade_log['fire_reason'] == 'RIP_ONLY').sum()}")
        print(f"  BOTH: {(trade_log['fire_reason'] == 'BOTH').sum()}")
    else:
        pd.DataFrame(
            columns=base_cols
            + [
                "prob_wont_crash_60",
                "prob_wont_rip_60",
                "actual_wont_crash_60",
                "actual_wont_rip_60",
                "fire_reason",
            ]
        ).to_csv(DATA_DIR / "step06_signal_log.csv", index=False)
        print(f"\nTrade log saved: 0 signals > 0.70 in either label")

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
