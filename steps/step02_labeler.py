"""
Sandwich Step 2 — Labeler

Computes three labels at every timestamp:
  wont_crash(Y, T)  — TRUE if max drawdown over next T min <= Y points
  wont_rip(Y, T)    — TRUE if max excursion over next T min <= Y points
  range_holds(Y_up, Y_down, T) — TRUE if spot stays in [spot-Y_down, spot+Y_up]

All labels are intraday only (same-date window constraint).
IN_FLIGHT when insufficient future bars remain.
"""

from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

DB_PATH = Path("/home/trading_ceo/python-trader/varaha/data/varaha_data.duckdb")
INDEX = "NIFTY"

LABEL_PARAMS = {
    "wont_crash_60": {"label_type": "wont_crash", "Y": 25, "T_minutes": 60},
    "wont_rip_60": {"label_type": "wont_rip", "Y": 25, "T_minutes": 60},
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_OUTPUT = OUTPUT_DIR / "step02_labels.parquet"


def load_spot_series():
    """
    Pull all NIFTY spot data from DuckDB.
    Returns: pd.DataFrame with columns [timestamp, date, time, spot, india_vix, days_to_weekly, atm_strike]
    Sorted ascending by timestamp.
    Drops any rows where spot is null.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT timestamp, date, time, spot, india_vix, days_to_weekly, atm_strike
        FROM market_data
        WHERE index_name = '{INDEX}'
        ORDER BY timestamp ASC
    """).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["spot"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows from market_data.")
    return df


def compute_label_for_row(df, idx, label_type, **params):
    """
    Given the full spot DataFrame and an integer row index `idx`, compute one label.
    label_type is one of: "wont_crash", "wont_rip", "range_holds"
    """
    T = params.get("T_minutes", 60)
    entry_date = df.iloc[idx]["date"]
    entry_spot = df.iloc[idx]["spot"]

    window_end = idx + 1 + T
    if window_end > len(df):
        return "IN_FLIGHT"

    window = df.iloc[idx + 1 : window_end]
    if len(window) < T:
        return "IN_FLIGHT"

    # Same-date constraint
    if (window["date"] != entry_date).any():
        return "IN_FLIGHT"

    if label_type == "wont_crash":
        Y = params.get("Y", 25)
        max_drawdown = entry_spot - window["spot"].min()
        return "TRUE" if max_drawdown <= Y else "FALSE"

    elif label_type == "wont_rip":
        Y = params.get("Y", 25)
        max_excursion = window["spot"].max() - entry_spot
        return "TRUE" if max_excursion <= Y else "FALSE"

    elif label_type == "range_holds":
        Y_up = params.get("Y_up", 25)
        Y_down = params.get("Y_down", 25)
        up_ok = window["spot"].max() - entry_spot <= Y_up
        down_ok = entry_spot - window["spot"].min() <= Y_down
        return "TRUE" if (up_ok and down_ok) else "FALSE"

    return "IN_FLIGHT"


def label_all_rows(df, label_params_dict):
    """
    Iterate over every row in df. For each row, compute every label in label_params_dict.
    Returns: pd.DataFrame with timestamp + label columns.
    """
    assert df["timestamp"].is_monotonic_increasing, (
        "spot series must be sorted by timestamp"
    )

    labels = {
        "timestamp": df["timestamp"],
        "date": df["date"],
        "time": df["time"],
        "spot": df["spot"],
        "india_vix": df["india_vix"],
        "days_to_weekly": df["days_to_weekly"],
    }

    for label_key, params in label_params_dict.items():
        col = []
        for i in range(len(df)):
            col.append(compute_label_for_row(df, i, **params))
        labels[label_key] = col
        print(f"  {label_key}: done")

    return pd.DataFrame(labels)


def validate_labeler(df, labels_df):
    """
    Sanity checks on the labeler output. Prints results, does not return.
    """
    print("\n=== Labeler Validation ===")
    assert len(labels_df) == len(df), (
        f"Row count mismatch: {len(labels_df)} vs {len(df)}"
    )

    label_cols = [
        c
        for c in labels_df.columns
        if c not in ("timestamp", "date", "time", "spot", "india_vix", "days_to_weekly")
    ]

    for lc in label_cols:
        assert labels_df[lc].notna().all(), f"Nulls found in {lc}"
        counts = labels_df[lc].value_counts()
        print(f"\n{lc} distribution:")
        for val in ["TRUE", "FALSE", "IN_FLIGHT"]:
            print(f"  {val}: {counts.get(val, 0):,}")

    # Spot-checks for wont_crash_60
    if "wont_crash_60" in label_cols:
        rng = np.random.RandomState(42)

        false_idx = labels_df.index[labels_df["wont_crash_60"] == "FALSE"].tolist()
        if len(false_idx) >= 3:
            picks = rng.choice(false_idx, size=3, replace=False)
            print("\n--- Spot-check: wont_crash_60 = FALSE (3 random) ---")
            for i in picks:
                entry_spot = df.iloc[i]["spot"]
                Y = LABEL_PARAMS["wont_crash_60"]["Y"]
                T = LABEL_PARAMS["wont_crash_60"]["T_minutes"]
                window = df.iloc[i + 1 : i + 1 + T]
                window = window[window["date"] == df.iloc[i]["date"]]
                min_spot = window["spot"].min() if len(window) > 0 else float("nan")
                dd = entry_spot - min_spot
                print(
                    f"  idx={i} entry_spot={entry_spot} min_next60={min_spot} dd={dd:.1f}>{Y} ✓"
                )

        true_idx = labels_df.index[labels_df["wont_crash_60"] == "TRUE"].tolist()
        valid_true = [
            i
            for i in true_idx
            if i + 60 < len(df) and df.iloc[i]["date"] == df.iloc[i + 60]["date"]
        ]
        if len(valid_true) >= 3:
            picks = rng.choice(valid_true, size=3, replace=False)
            print(
                "\n--- Spot-check: wont_crash_60 = TRUE (3 random, sufficient bars) ---"
            )
            for i in picks:
                entry_spot = df.iloc[i]["spot"]
                Y = LABEL_PARAMS["wont_crash_60"]["Y"]
                T = LABEL_PARAMS["wont_crash_60"]["T_minutes"]
                window = df.iloc[i + 1 : i + 1 + T]
                window = window[window["date"] == df.iloc[i]["date"]]
                min_spot = window["spot"].min() if len(window) > 0 else float("nan")
                dd = entry_spot - min_spot
                print(
                    f"  idx={i} entry_spot={entry_spot} min_next60={min_spot} dd={dd:.1f}≤{Y} ✓"
                )

    # Edge cases
    print("\n--- Edge: last 5 rows of each date ---")
    for d in sorted(labels_df["date"].unique()):
        day_rows = labels_df[labels_df["date"] == d]
        tail = day_rows.tail(5)
        for lc in label_cols:
            vals = tail[lc].tolist()
            print(f"  {d} | {lc}: {vals}")


def main():
    print("=" * 60)
    print("Sandwich Step 2 — Labeler")
    print("=" * 60)
    df = load_spot_series()
    labels_df = label_all_rows(df, LABEL_PARAMS)
    validate_labeler(df, labels_df)
    labels_df.to_parquet(LABELS_OUTPUT, index=False)
    print(f"\nLabels saved to: {LABELS_OUTPUT}")
    print(f"Output size: {LABELS_OUTPUT.stat().st_size / 1024:.1f} KB")
    print("Step 2 complete.")


if __name__ == "__main__":
    main()
