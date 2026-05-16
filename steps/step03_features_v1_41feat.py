"""
Sandwich Step 3 — Feature Panel

Computes 41 features across 7 families at each timestamp:
  TREND (12), MOMENTUM (6), VOLATILITY (5), TIME (4),
  EXPIRY CONTEXT (3), OPTION CHAIN (8), REGIME FLAGS (3)
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
FEATURES_OUTPUT = OUTPUT_DIR / "step03_features_and_labels_v1_41feat.parquet"
CORR_MATRIX = OUTPUT_DIR / "step03_correlation_matrix.csv"
CORR_PAIRS = OUTPUT_DIR / "step03_correlation_pairs.csv"


def load_market_data():
    """Returns DataFrame of all NIFTY market_data, ascending by timestamp."""
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
    """Returns DataFrame of NIFTY option_snapshots (weekly expiry), ascending by timestamp."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT timestamp, date, expiry_label, expiry_date, strike_offset,
               option_type, ltp, volume, oi, iv
        FROM option_snapshots
        WHERE tsym LIKE 'NIFTY%'
          AND expiry_label = 'weekly'
        ORDER BY timestamp ASC
    """).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"option_snapshots (weekly): {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Family 1: TREND (12 features)
# ---------------------------------------------------------------------------


def compute_trend_features(df):
    spot = df["spot"].astype(float)
    df["trend_ema_5"] = spot.ewm(span=5, adjust=False).mean()
    df["trend_ema_20"] = spot.ewm(span=20, adjust=False).mean()
    df["trend_ema_50"] = spot.ewm(span=50, adjust=False).mean()
    df["trend_spot_vs_ema5"] = (spot - df["trend_ema_5"]) / spot
    df["trend_spot_vs_ema20"] = (spot - df["trend_ema_20"]) / spot
    df["trend_spot_vs_ema50"] = (spot - df["trend_ema_50"]) / spot
    df["trend_ema5_slope"] = (df["trend_ema_5"] - df["trend_ema_5"].shift(5)) / df[
        "trend_ema_5"
    ].shift(5)
    df["trend_ema20_slope"] = (df["trend_ema_20"] - df["trend_ema_20"].shift(20)) / df[
        "trend_ema_20"
    ].shift(20)
    df["trend_higher_high_20"] = (
        spot > spot.rolling(20, min_periods=1).max().shift(1)
    ).astype(int)
    df["trend_lower_low_20"] = (
        spot < spot.rolling(20, min_periods=1).min().shift(1)
    ).astype(int)

    # VWAP: intraday, reset at start of each date. Uniform volume proxy (1 per minute).
    vwap = spot.groupby(df["date"]).expanding().mean().droplevel(0).sort_index()
    df["vwap"] = vwap
    df["trend_above_vwap"] = (spot > df["vwap"]).astype(int)
    df["trend_vwap_distance"] = (spot - df["vwap"]) / spot
    return df


# ---------------------------------------------------------------------------
# Family 2: MOMENTUM (6 features)
# ---------------------------------------------------------------------------


def compute_momentum_features(df):
    spot = df["spot"].astype(float)
    # RSI(14) Wilder
    delta = spot.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["mom_rsi_14"] = 100 - (100 / (1 + rs))
    df["mom_rsi_14_slope"] = df["mom_rsi_14"].diff(3)
    df["mom_roc_5"] = (spot - spot.shift(5)) / spot.shift(5)
    df["mom_roc_15"] = (spot - spot.shift(15)) / spot.shift(15)
    df["mom_roc_60"] = (spot - spot.shift(60)) / spot.shift(60)

    # Consecutive-up bars
    up = (spot > spot.shift(1)).astype(int)
    cons = up.groupby((up == 0).cumsum()).cumsum()
    df["mom_consecutive_up"] = cons.clip(upper=10).astype(float)
    return df


# ---------------------------------------------------------------------------
# Family 3: VOLATILITY (5 features)
# ---------------------------------------------------------------------------


def compute_volatility_features(df):
    spot = df["spot"].astype(float)
    # ATR(14) proxy: abs(spot.diff()) since no high/low in schema
    tr = spot.diff().abs()
    df["vol_atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol_atr_14_pct"] = df["vol_atr_14"] / spot
    # Realized vol 30-min, annualized
    df["vol_realized_30"] = spot.pct_change().rolling(30).std() * np.sqrt(375 * 252)
    df["vol_vix"] = df["india_vix"].astype(float)
    # VIX change from ~1 day ago (375 bars)
    df["vol_vix_change_1d"] = df["vol_vix"] - df["vol_vix"].shift(375)
    return df


# ---------------------------------------------------------------------------
# Family 4: TIME (4 features)
# ---------------------------------------------------------------------------


def compute_time_features(df):
    times = pd.to_datetime(df["time"], format="%H:%M:%S")
    minutes = times.dt.hour * 60 + times.dt.minute
    df["time_minute_of_day"] = minutes
    df["time_minutes_since_open"] = minutes - 555
    df["time_minutes_to_close"] = 930 - minutes
    df["time_day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# Family 5: EXPIRY CONTEXT (3 features)
# ---------------------------------------------------------------------------


def compute_expiry_features(df):
    dte = df["days_to_weekly"].astype(float)
    df["expiry_dte"] = dte
    df["expiry_is_dte0"] = (dte == 0).astype(int)
    df["expiry_is_dte_le2"] = (dte <= 2).astype(int)
    return df


# ---------------------------------------------------------------------------
# Family 6: OPTION CHAIN (8 features)
# ---------------------------------------------------------------------------


def compute_option_features(market_df, opts_df):
    mdf = market_df.copy()
    odf = opts_df.copy()

    # Aggregate option chain per timestamp
    for feat in [
        "opt_atm_iv",
        "opt_atm_pe_iv",
        "opt_iv_skew",
        "opt_pcr_oi",
        "opt_total_oi_change_5m",
        "opt_max_pain_offset",
        "opt_atm_ce_oi_change_5m",
        "opt_atm_pe_oi_change_5m",
    ]:
        mdf[feat] = np.nan

    # Build a dict of chain snapshots by timestamp
    odf["ts_str"] = odf["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    mdf["ts_str"] = mdf["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Group option snapshots by timestamp
    ts_groups = {ts: grp for ts, grp in odf.groupby("ts_str")}
    mdf_ts = mdf["ts_str"].tolist()

    # Pre-compute OI totals per timestamp for 5-min lookback
    oi_totals = {}
    for ts, grp in ts_groups.items():
        strikes_in_range = grp[grp["strike_offset"].abs() <= 5]
        oi_totals[ts] = strikes_in_range["oi"].sum()

    for i, ts in enumerate(mdf_ts):
        if ts not in ts_groups:
            continue
        chain = ts_groups[ts]

        # ATM IVs
        atm_ce = chain[(chain["strike_offset"] == 0) & (chain["option_type"] == "CE")]
        atm_pe = chain[(chain["strike_offset"] == 0) & (chain["option_type"] == "PE")]
        if len(atm_ce) > 0:
            mdf.at[i, "opt_atm_iv"] = atm_ce["iv"].iloc[0]
        if len(atm_pe) > 0:
            mdf.at[i, "opt_atm_pe_iv"] = atm_pe["iv"].iloc[0]

        # IV Skew
        if not pd.isna(mdf.at[i, "opt_atm_iv"]) and not pd.isna(
            mdf.at[i, "opt_atm_pe_iv"]
        ):
            mdf.at[i, "opt_iv_skew"] = (
                mdf.at[i, "opt_atm_pe_iv"] - mdf.at[i, "opt_atm_iv"]
            )

        # PCR OI
        in_range = chain[chain["strike_offset"].abs() <= 5]
        pe_oi = in_range[in_range["option_type"] == "PE"]["oi"].sum()
        ce_oi = in_range[in_range["option_type"] == "CE"]["oi"].sum()
        if ce_oi > 0:
            mdf.at[i, "opt_pcr_oi"] = pe_oi / ce_oi

        # OI change 5m
        if i >= 5:
            ts_5m_ago = mdf_ts[i - 5]
            oi_now = oi_totals.get(ts, 0)
            oi_prev = oi_totals.get(ts_5m_ago, 0)
            mdf.at[i, "opt_total_oi_change_5m"] = oi_now - oi_prev

            if ts_5m_ago in ts_groups:
                prev_chain = ts_groups[ts_5m_ago]
                ce_now = atm_ce["oi"].sum() if len(atm_ce) > 0 else 0
                pe_now = atm_pe["oi"].sum() if len(atm_pe) > 0 else 0
                prev_ce = prev_chain[
                    (prev_chain["strike_offset"] == 0)
                    & (prev_chain["option_type"] == "CE")
                ]["oi"].sum()
                prev_pe = prev_chain[
                    (prev_chain["strike_offset"] == 0)
                    & (prev_chain["option_type"] == "PE")
                ]["oi"].sum()
                mdf.at[i, "opt_atm_ce_oi_change_5m"] = ce_now - prev_ce
                mdf.at[i, "opt_atm_pe_oi_change_5m"] = pe_now - prev_pe

        # Max Pain
        strikes = chain[chain["strike_offset"].abs() <= 5]
        pain_by_strike = {}
        for _, row in mdf.iterrows():
            pass  # outside this loop
        strike_offsets = sorted(strikes["strike_offset"].unique())
        min_pain = float("inf")
        min_offset = np.nan
        atm = mdf.at[i, "opt_atm_iv"]
        for so in strike_offsets:
            sub = strikes[strikes["strike_offset"] == so]
            pe = sub[sub["option_type"] == "PE"]
            ce = sub[sub["option_type"] == "CE"]
            pain = pe["oi"].sum() + ce["oi"].sum()
            if pain < min_pain:
                min_pain = pain
                min_offset = so
        mdf.at[i, "opt_max_pain_offset"] = min_offset

    mdf = mdf.drop(columns=["ts_str"])
    return mdf


# ---------------------------------------------------------------------------
# Family 7: REGIME FLAGS (3 features)
# ---------------------------------------------------------------------------


def compute_regime_features(df):
    vix = df["india_vix"].astype(float)
    df["regime_vix_low"] = (vix < 14).astype(int)
    df["regime_vix_mid"] = ((vix >= 14) & (vix < 20)).astype(int)
    df["regime_vix_high"] = (vix >= 20).astype(int)
    return df


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_feature_panel():
    mdf = load_market_data()
    odf = load_option_snapshots()
    mdf = compute_trend_features(mdf)
    mdf = compute_momentum_features(mdf)
    mdf = compute_volatility_features(mdf)
    mdf = compute_time_features(mdf)
    mdf = compute_expiry_features(mdf)
    mdf = compute_option_features(mdf, odf)
    mdf = compute_regime_features(mdf)
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
        c
        for c in features_df.columns
        if c.startswith(
            ("trend_", "mom_", "vol_", "time_m", "expiry_", "opt_", "regime_")
        )
    ]
    corr = features_df[feature_cols].select_dtypes(include=[np.number]).corr()
    corr.to_csv(CORR_MATRIX)
    print(f"Correlation matrix saved: {CORR_MATRIX}")

    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c = corr.iloc[i, j]
            if abs(c) > 0.85:
                pairs.append(
                    {
                        "feat_a": feature_cols[i],
                        "feat_b": feature_cols[j],
                        "correlation": round(c, 4),
                    }
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
    print("Sandwich Step 3 — Feature Panel")
    print("=" * 60)
    features_df = assemble_feature_panel()
    feature_cols = [
        c
        for c in features_df.columns
        if c.startswith(
            ("trend_", "mom_", "vol_", "time_m", "expiry_", "opt_", "regime_")
        )
    ]
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    # Null counts
    nulls = features_df[feature_cols].isna().sum().sort_values(ascending=False)
    print("\nNull counts per feature (top 10):")
    print(nulls.head(10).to_string())

    # Distribution sanity
    print("\nFeature distribution (5 samples):")
    sample_cols = [
        "trend_ema_20",
        "mom_rsi_14",
        "vol_atr_14",
        "expiry_dte",
        "opt_pcr_oi",
    ]
    available = [c for c in sample_cols if c in features_df.columns]
    if available:
        print(features_df[available].describe().to_string())

    # VWAP note
    print("\nVWAP source: uniform 1-per-minute volume proxy (intraday mean per date)")

    joined = join_with_labels(features_df, LABELS_PATH)
    joined.to_parquet(FEATURES_OUTPUT, index=False)
    print(f"\nSaved: {FEATURES_OUTPUT}")
    print(f"Size: {FEATURES_OUTPUT.stat().st_size / 1024:.1f} KB")

    correlation_report(joined)
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
