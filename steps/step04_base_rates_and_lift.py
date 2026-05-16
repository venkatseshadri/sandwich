"""
Sandwich Step 4 — Base Rates & Feature Lift

Computes base rates, feature lift tables, and contextual breakdowns
by time-of-day, VIX regime, and DTE for all three labels.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "step03_features_and_labels.parquet"

LABEL_COLS = ["wont_crash_60", "wont_rip_60"]

FEATURE_COLS = [
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


def load_dataset():
    """Reads step03_features_and_labels.parquet, filters out IN_FLIGHT rows."""
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded: {len(df):,} rows")
    for lc in LABEL_COLS:
        n_before = len(df)
        df = df[df[lc] != "IN_FLIGHT"].copy()
        print(f"  Dropped IN_FLIGHT for {lc}: {n_before - len(df):,} rows")
    print(f"Final: {len(df):,} rows")
    return df


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportion p with sample size n."""
    if n == 0:
        return (np.nan, np.nan)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (center - margin, center + margin)


def compute_base_rates(df, label_cols):
    """For each label, compute base rate with Wilson CI."""
    rows = []
    for lc in label_cols:
        n_total = len(df[df[lc].notna()])
        n_true = (df[lc] == "TRUE").sum()
        base_rate = n_true / n_total if n_total > 0 else np.nan
        ci_low, ci_high = wilson_ci(base_rate, n_total)
        rows.append(
            {
                "label": lc,
                "base_rate": round(base_rate, 4),
                "n": n_total,
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
            }
        )
    result = pd.DataFrame(rows)
    print("\n=== Base Rates ===")
    print(result.to_string())
    return result


def bucket_feature(series, n_buckets=4):
    """
    Bucket a numeric feature into quantile buckets.

    Cases:
    - Few distinct values (<=n_buckets): use each value as its own bucket (treat as categorical)
    - Normal case: qcut with duplicates="drop"; log if fewer buckets result than requested
    - qcut fails: return NaN series, do NOT silently coerce to strings
    """
    name = getattr(series, "name", "<unnamed>")
    s = series.dropna()

    if len(s) == 0:
        return pd.Series([np.nan] * len(series), index=series.index)

    n_unique = s.nunique()
    if n_unique <= n_buckets:
        return series.astype(str)

    try:
        result, edges = pd.qcut(s, n_buckets, retbins=True, duplicates="drop")
        actual = len(edges) - 1
        if actual < n_buckets:
            print(
                f"  Note: {name} bucketed into {actual} buckets (requested {n_buckets})"
            )
        return result.reindex(series.index)
    except ValueError as e:
        print(f"  Could not bucket {name}: {e}")
        return pd.Series([np.nan] * len(series), index=series.index)


def compute_lift_table(
    df, feature_col, label_col, base_rate, min_bucket_n=20, precomputed_buckets=None
):
    """
    For each bucket of feature_col, compute P(label==TRUE) and lift.
    If precomputed_buckets is provided, use it instead of bucketing here
    (ensures the same buckets are used across multiple labels for fair comparison).
    """
    valid = df[[feature_col, label_col]].dropna()
    if len(valid) == 0:
        return pd.DataFrame()

    if precomputed_buckets is not None:
        buckets = precomputed_buckets.reindex(valid.index)
    else:
        buckets = bucket_feature(valid[feature_col])

    unique_buckets = sorted(buckets.dropna().unique())

    rows = []
    for b in unique_buckets:
        mask = buckets == b
        n = mask.sum()
        if n < min_bucket_n:
            continue
        n_true = (valid.loc[mask, label_col] == "TRUE").sum()
        p_true = n_true / n
        lift = p_true / base_rate if base_rate > 0 else np.nan
        rows.append(
            {"bucket": b, "n": n, "p_true": round(p_true, 4), "lift": round(lift, 4)}
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def rank_features_by_lift(df, feature_cols, label_col, base_rate):
    """For each feature, compute max-min lift spread across buckets."""
    rows = []
    for fc in feature_cols:
        # Bucket the feature once on the full df (not inside compute_lift_table)
        feature_buckets = bucket_feature(df[fc])
        lt = compute_lift_table(
            df, fc, label_col, base_rate, precomputed_buckets=feature_buckets
        )
        if lt.empty or len(lt) < 2:
            continue
        max_lift = lt["lift"].max()
        min_lift = lt["lift"].min()
        rows.append(
            {
                "feature": fc,
                "max_lift": round(max_lift, 4),
                "min_lift": round(min_lift, 4),
                "lift_spread": round(max_lift - min_lift, 4),
                "max_bucket_n": lt["n"].max(),
            }
        )
    result = pd.DataFrame(rows).sort_values("lift_spread", ascending=False)
    return result


def time_of_day_breakdown(df, label_col, bin_minutes=30):
    """Break label by 30-minute buckets of time_minutes_since_open."""
    df = df.dropna(subset=["time_minutes_since_open", label_col])
    tod = df["time_minutes_since_open"]
    bins = np.arange(0, 376, bin_minutes)
    labels = [f"{(b + 555) // 60}:{(b + 555) % 60:02d}" for b in bins[:-1]]
    bucket = pd.cut(tod, bins=bins, labels=labels, right=False)
    rows = []
    for b in labels:
        mask = bucket == b
        n = mask.sum()
        if n < 5:
            continue
        n_true = (df.loc[mask, label_col] == "TRUE").sum()
        rows.append({"tod_bucket": b, "n": n, "p_true": round(n_true / n, 4)})
    return pd.DataFrame(rows)


def vix_regime_breakdown(df, label_col):
    """Break label by VIX regime."""
    df = df.dropna(subset=["vol_india_vix", label_col])
    vix = df["vol_india_vix"]
    regimes = []
    for _, v in vix.items():
        if v < 14:
            regimes.append("low")
        elif v < 20:
            regimes.append("mid")
        else:
            regimes.append("high")
    regimes_s = pd.Series(regimes, index=df.index)
    rows = []
    for r in ["low", "mid", "high"]:
        mask = regimes_s == r
        n = mask.sum()
        if n < 5:
            continue
        n_true = (df.loc[mask, label_col] == "TRUE").sum()
        rows.append({"regime": r, "n": n, "p_true": round(n_true / n, 4)})
    return pd.DataFrame(rows)


def dte_breakdown(df, label_col):
    """Break label by expiry_dte."""
    df = df.dropna(subset=["expiry_dte", label_col])
    rows = []
    for dte_val in sorted(df["expiry_dte"].dropna().unique()):
        mask = df["expiry_dte"] == dte_val
        n = mask.sum()
        n_true = (df.loc[mask, label_col] == "TRUE").sum()
        rows.append({"dte": int(dte_val), "n": n, "p_true": round(n_true / n, 4)})
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Sandwich Step 4 — Base Rates & Feature Lift")
    print("=" * 60)

    df = load_dataset()
    n_before = len(df)
    # Keep only rows with at least some features
    df = df.dropna(subset=["vol_india_vix", "expiry_dte"])
    print(f"Dropped NaN vol_india_vix/expiry_dte: {n_before - len(df):,} rows")
    base_rates = compute_base_rates(df, LABEL_COLS)
    base_rates.to_csv(DATA_DIR / "step04_base_rates.csv", index=False)

    date_col = "date_feat" if "date_feat" in df.columns else "date"
    print(f"\nDate range: {df[date_col].min()} to {df[date_col].max()}")
    print(
        f"VIX range: {df['vol_india_vix'].min():.1f} to {df['vol_india_vix'].max():.1f}"
    )
    print(
        f"INSUFFICIENT DATA WARNING: {len(df)} samples across {df[date_col].nunique()} dates. "
    )
    print(
        f"Results are code-validation only. Real inference requires 30+ trading days.\n"
    )

    for lc in LABEL_COLS:
        br = base_rates[base_rates["label"] == lc]["base_rate"].iloc[0]
        print(f"\n{'=' * 60}")
        print(f"Label: {lc} (base rate: {br:.2%})")
        print(f"{'=' * 60}")

        # Feature lift ranking
        features_avail = [c for c in FEATURE_COLS if c in df.columns]
        lift_rank = rank_features_by_lift(df, features_avail, lc, br)
        print(f"\nTop 5 features by lift spread:")
        print(
            lift_rank.head(5)[
                ["feature", "max_lift", "min_lift", "lift_spread"]
            ].to_string()
        )
        lift_rank.to_csv(DATA_DIR / f"step04_lift_{lc}.csv", index=False)

        # Time of day
        tod = time_of_day_breakdown(df, lc)
        print(f"\nTime-of-day breakdown (top 5):")
        print(tod.sort_values("n", ascending=False).head(5).to_string())
        tod.to_csv(DATA_DIR / f"step04_tod_{lc}.csv", index=False)

        # VIX regime
        regime = vix_regime_breakdown(df, lc)
        print(f"\nVIX regime breakdown:")
        print(regime.to_string())
        regime.to_csv(DATA_DIR / f"step04_regime_{lc}.csv", index=False)

        # DTE
        dte = dte_breakdown(df, lc)
        print(f"\nDTE breakdown:")
        print(dte.to_string())
        dte.to_csv(DATA_DIR / f"step04_dte_{lc}.csv", index=False)

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
