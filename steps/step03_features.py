"""
Sandwich Step 3 — Feature Panel (R2: 15-feature lean balanced set)

15 features across 9 families. No family has more than 4 features.
Multi-TF trend colors replace raw EMA values.
"""

from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

np.random.seed(42)

DB_PATH = Path("/home/trading_ceo/python-trader/varaha/data/varaha_data.duckdb")
INDEX = "NIFTY"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = OUTPUT_DIR / "step02_labels.parquet"
FEATURES_OUTPUT = OUTPUT_DIR / "step03_features_and_labels.parquet"
CORR_MATRIX = OUTPUT_DIR / "step03_correlation_matrix.csv"
CORR_PAIRS = OUTPUT_DIR / "step03_correlation_pairs.csv"


def load_market_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT timestamp, date, time, spot, india_vix, days_to_weekly, atm_strike
        FROM market_data
        WHERE index_name = '{INDEX}'
        ORDER BY timestamp ASC
    """).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].is_monotonic_increasing
    print(f"market_data: {len(df):,} rows")
    return df


def load_option_snapshots():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT timestamp, date, strike_offset, option_type, oi, iv
        FROM option_snapshots
        WHERE tsym LIKE 'NIFTY%'
          AND expiry_label = 'weekly'
        ORDER BY timestamp ASC
    """).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"option_snapshots (weekly): {len(df):,} rows")
    return df


def compute_trend_multitf_features(df):
    """
    Adds 4 features: trend_1m, trend_5m, trend_1h, trend_1d.
    For 1m: direct EMA20 on spot + slope over 20 bars.
    For 5m/1h: resample bars, compute EMA20, slope, forward-fill.
    For 1d: daily close direction + 5-day slope.
    """
    df = df.set_index("timestamp")

    # trend_1m (directly on 1-min bars)
    e20_1m = df["spot"].ewm(span=20, adjust=False).mean()
    slope_1m = e20_1m.diff(20)
    df["trend_1m"] = 0
    mask_up = (df["spot"] > e20_1m) & (slope_1m > 0)
    mask_down = (df["spot"] < e20_1m) & (slope_1m < 0)
    df.loc[mask_up, "trend_1m"] = 1
    df.loc[mask_down, "trend_1m"] = -1

    # trend_5m: resample to 5-min bars
    spot_5m = df["spot"].resample("5min").last().dropna()
    e20_5m = spot_5m.ewm(span=20, adjust=False).mean()
    slope_5m = e20_5m.diff(20)
    t5_vals = pd.Series(0, index=spot_5m.index)
    t5_vals[(spot_5m > e20_5m) & (slope_5m > 0)] = 1
    t5_vals[(spot_5m < e20_5m) & (slope_5m < 0)] = -1
    df["trend_5m"] = t5_vals.reindex(df.index, method="ffill").fillna(0).astype(int)

    # trend_1h: resample to 60-min bars
    spot_1h = df["spot"].resample("60min").last().dropna()
    e20_1h = spot_1h.ewm(span=20, adjust=False).mean()
    slope_1h = e20_1h.diff(20)
    t1h_vals = pd.Series(0, index=spot_1h.index)
    t1h_vals[(spot_1h > e20_1h) & (slope_1h > 0)] = 1
    t1h_vals[(spot_1h < e20_1h) & (slope_1h < 0)] = -1
    df["trend_1h"] = t1h_vals.reindex(df.index, method="ffill").fillna(0).astype(int)

    df = df.reset_index()

    # trend_1d: requires 5+ prior trading days for non-zero output
    daily_close = df.groupby("date")["spot"].last()
    daily_dir = (daily_close > daily_close.shift(1)).astype(int) - (
        daily_close < daily_close.shift(1)
    ).astype(int)

    # 5-day rolling mean of direction — only meaningful with 5+ prior days
    # Use min_periods=5 (NOT 1) to enforce data sufficiency
    daily_slope = daily_dir.rolling(5, min_periods=5).mean()

    t1d_map = {}
    sorted_dates = sorted(df["date"].unique())
    for date_idx, d in enumerate(sorted_dates):
        # Need at least 5 prior days (i.e., this is the 6th day or later)
        if date_idx < 5:
            t1d_map[d] = 0
            continue

        dir_val = daily_dir.get(d, 0)
        slope_val = daily_slope.get(d, np.nan)

        if pd.isna(slope_val):
            t1d_map[d] = 0
        elif dir_val > 0 and slope_val > 0:
            t1d_map[d] = 1
        elif dir_val < 0 and slope_val < 0:
            t1d_map[d] = -1
        else:
            t1d_map[d] = 0

    df["trend_1d"] = df["date"].map(t1d_map).fillna(0).astype(int)

    print(f"trend features: trend_1m, trend_5m, trend_1h, trend_1d")
    return df


def compute_momentum_features(df):
    """Adds 1 feature: mom_rsi_14. Standard Wilder RSI on 1-min spot."""
    spot = df["spot"].astype(float)
    delta = spot.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["mom_rsi_14"] = 100 - (100 / (1 + rs))
    return df


def compute_volatility_features(df):
    """Adds 2 features: vol_atr_pct, vol_india_vix."""
    spot = df["spot"].astype(float)
    tr = spot.diff().abs()
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol_atr_pct"] = atr14 / spot
    df["vol_india_vix"] = df["india_vix"].astype(float)
    return df


def compute_volume_features(df):
    """
    Adds 1 feature: vwap_distance.

    Note: market_data has no volume column. We use an intraday running mean
    of spot as a VWAP proxy (each minute weighted equally). This is NOT a
    true VWAP. The feature name 'vwap_distance' is retained for clarity at
    the model layer, but the underlying computation is intraday_running_mean.
    """
    df = df.copy()
    spot = df["spot"].astype(float)
    intraday_running_mean = (
        spot.groupby(df["date"]).expanding().mean().droplevel(0).sort_index()
    )
    df["vwap_distance"] = (spot - intraday_running_mean) / spot
    return df


def compute_option_features(market_df, opts_df):
    """Adds 3 features: opt_atm_iv, opt_pcr_oi, opt_atm_oi_change_5m."""
    import time

    t0 = time.time()

    mdf = market_df.copy()
    mdf["opt_atm_iv"] = np.nan
    mdf["opt_pcr_oi"] = np.nan
    mdf["opt_atm_oi_change_5m"] = np.nan

    odf = opts_df.copy()
    odf["ts_str"] = odf["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    mdf["ts_str"] = mdf["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    ts_groups = {ts: grp for ts, grp in odf.groupby("ts_str")}
    mdf_ts = mdf["ts_str"].tolist()

    atm_ce_oi_map = {}
    for ts, grp in ts_groups.items():
        atm_ce = grp[(grp["strike_offset"] == 0) & (grp["option_type"] == "CE")]
        atm_ce_oi_map[ts] = atm_ce["oi"].sum() if len(atm_ce) > 0 else 0

    for i, ts in enumerate(mdf_ts):
        if ts not in ts_groups:
            continue
        chain = ts_groups[ts]
        in_range = chain[chain["strike_offset"].abs() <= 5]

        # atm_iv
        atm_ce = chain[(chain["strike_offset"] == 0) & (chain["option_type"] == "CE")]
        if len(atm_ce) > 0:
            mdf.at[i, "opt_atm_iv"] = atm_ce["iv"].iloc[0]

        # pcr_oi
        pe_oi = in_range[in_range["option_type"] == "PE"]["oi"].sum()
        ce_oi = in_range[in_range["option_type"] == "CE"]["oi"].sum()
        if ce_oi > 0:
            mdf.at[i, "opt_pcr_oi"] = pe_oi / ce_oi

        # atm_oi_change_5m (CE only)
        if i >= 5:
            ts_5m_ago = mdf_ts[i - 5]
            oi_now = atm_ce_oi_map.get(ts, 0)
            oi_prev = atm_ce_oi_map.get(ts_5m_ago, 0)
            mdf.at[i, "opt_atm_oi_change_5m"] = oi_now - oi_prev

    mdf = mdf.drop(columns=["ts_str"])
    elapsed = time.time() - t0
    print(
        f"option features: opt_atm_iv, opt_pcr_oi, opt_atm_oi_change_5m ({elapsed:.1f}s)"
    )
    return mdf


def compute_time_features(df):
    """Adds 1 feature: time_minutes_since_open."""
    times = pd.to_datetime(df["time"], format="%H:%M:%S")
    minutes = times.dt.hour * 60 + times.dt.minute
    df["time_minutes_since_open"] = minutes - 555
    return df


def compute_expiry_features(df):
    """Adds 1 feature: expiry_dte."""
    df["expiry_dte"] = df["days_to_weekly"].astype(float)
    return df


def compute_structure_features(df):
    """
    Adds 2 features: sr_dist_to_yday_high, sr_dist_to_yday_low.

    For each row at date D with current spot S:
        sr_dist_to_yday_high = (prev_day_high - S) / S
        sr_dist_to_yday_low  = (S - prev_day_low)  / S

    Both vary minute-by-minute as spot moves.
    First trading date in the dataset has both features = NaN.
    """
    df = df.copy()

    daily_max = df.groupby("date")["spot"].max()
    daily_min = df.groupby("date")["spot"].min()
    prev_max = daily_max.shift(1)
    prev_min = daily_min.shift(1)

    # Map prev-day's high/low onto every row of the current day
    df["_yday_high"] = df["date"].map(prev_max)
    df["_yday_low"] = df["date"].map(prev_min)

    # Compute distance using CURRENT row's spot, not day-opening spot
    df["sr_dist_to_yday_high"] = (df["_yday_high"] - df["spot"]) / df["spot"]
    df["sr_dist_to_yday_low"] = (df["spot"] - df["_yday_low"]) / df["spot"]

    df = df.drop(columns=["_yday_high", "_yday_low"])

    print(f"structure features: sr_dist_to_yday_high, sr_dist_to_yday_low")
    return df


def assemble_feature_panel():
    mdf = load_market_data()
    odf = load_option_snapshots()
    mdf = compute_trend_multitf_features(mdf)
    mdf = compute_momentum_features(mdf)
    mdf = compute_volatility_features(mdf)
    mdf = compute_volume_features(mdf)
    mdf = compute_option_features(mdf, odf)
    mdf = compute_time_features(mdf)
    mdf = compute_expiry_features(mdf)
    mdf = compute_structure_features(mdf)
    return mdf


def join_with_labels(features_df, labels_path):
    labels_df = pd.read_parquet(labels_path)
    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"])
    merged = features_df.merge(
        labels_df, on="timestamp", how="inner", suffixes=("_feat", "_label")
    )
    print(f"Joined: {len(merged):,} rows")
    return merged


def correlation_report(features_df):
    feature_cols = [
        "trend_1m",
        "trend_5m",
        "trend_1h",
        "trend_1d",
        "mom_rsi_14",
        "vol_atr_pct",
        "vol_india_vix",
        "vwap_distance",
        "opt_atm_iv",
        "opt_pcr_oi",
        "opt_atm_oi_change_5m",
        "time_minutes_since_open",
        "expiry_dte",
        "sr_dist_to_yday_high",
        "sr_dist_to_yday_low",
    ]
    avail = [c for c in feature_cols if c in features_df.columns]
    corr = features_df[avail].select_dtypes(include=[np.number]).corr()
    corr.to_csv(CORR_MATRIX)
    print(f"Correlation matrix saved: {CORR_MATRIX}")

    pairs = []
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            c = corr.iloc[i, j]
            if abs(c) > 0.85:
                pairs.append(
                    {"feat_a": avail[i], "feat_b": avail[j], "correlation": round(c, 4)}
                )

    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values(
            "correlation", key=abs, ascending=False
        )
        pairs_df.to_csv(CORR_PAIRS, index=False)
        print(f"Correlation pairs (|corr| > 0.85): {len(pairs_df)}")
        print(pairs_df.to_string())
    else:
        print("No feature pairs with |corr| > 0.85")
        pd.DataFrame(columns=["feat_a", "feat_b", "correlation"]).to_csv(
            CORR_PAIRS, index=False
        )


def main():
    print("=" * 60)
    print("Sandwich Step 3 — Feature Panel (R2: 15-feature lean)")
    print("=" * 60)

    # Diagnostic: option chain coverage
    con = duckdb.connect(str(DB_PATH), read_only=True)
    mkt_ts = con.execute(
        f"SELECT COUNT(DISTINCT timestamp) FROM market_data WHERE index_name='{INDEX}'"
    ).fetchone()[0]
    opt_total = con.execute("SELECT COUNT(*) FROM option_snapshots").fetchone()[0]
    opt_filt = con.execute(
        "SELECT COUNT(*) FROM option_snapshots WHERE tsym LIKE 'NIFTY%' AND expiry_label='weekly'"
    ).fetchone()[0]
    opt_dt = con.execute(
        "SELECT COUNT(DISTINCT timestamp) FROM option_snapshots WHERE tsym LIKE 'NIFTY%' AND expiry_label='weekly'"
    ).fetchone()[0]
    con.close()
    print(f"\nOption chain coverage: {opt_total:,} total rows, {opt_filt:,} filtered")
    print(f"Distinct timestamps: market_data={mkt_ts:,}  option_snapshots={opt_dt:,}")
    print(f"Ratio filtered_rows/distinct_ts: {opt_filt / opt_dt:.1f}")

    features_df = assemble_feature_panel()
    feature_cols = [
        "trend_1m",
        "trend_5m",
        "trend_1h",
        "trend_1d",
        "mom_rsi_14",
        "vol_atr_pct",
        "vol_india_vix",
        "vwap_distance",
        "opt_atm_iv",
        "opt_pcr_oi",
        "opt_atm_oi_change_5m",
        "time_minutes_since_open",
        "expiry_dte",
        "sr_dist_to_yday_high",
        "sr_dist_to_yday_low",
    ]
    avail = [c for c in feature_cols if c in features_df.columns]
    assert len(avail) == 15, f"Expected 15 features, got {len(avail)}: {avail}"
    print(f"Features: {len(avail)} — {avail}")

    nulls = features_df[avail].isna().sum().sort_values(ascending=False)
    print("\nNull counts per feature:")
    print(nulls.to_string())

    print("\nFeature distribution (5 samples):")
    sample_cols = [
        "trend_1m",
        "mom_rsi_14",
        "vol_atr_pct",
        "vwap_distance",
        "expiry_dte",
    ]
    print(features_df[sample_cols].describe().to_string())

    joined = join_with_labels(features_df, LABELS_PATH)
    joined.to_parquet(FEATURES_OUTPUT, index=False)
    print(f"\nSaved: {FEATURES_OUTPUT}")
    print(f"Size: {FEATURES_OUTPUT.stat().st_size / 1024:.1f} KB")

    correlation_report(joined)
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
