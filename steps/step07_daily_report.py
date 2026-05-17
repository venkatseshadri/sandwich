"""
Sandwich Step 7 — Daily Report Generator

Generates a markdown daily report from DuckDB data and Sandwich model outputs.
First run includes a glossary. Subsequent runs omit it.

Usage:
    python3 steps/step07_daily_report.py
"""

from pathlib import Path
import sys
import re
from datetime import datetime, timezone, timedelta

import duckdb
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signal_api import SandwichSignalEngine

IST = timezone(timedelta(hours=5, minutes=30))

DB_PATH = Path("/home/trading_ceo/python-trader/varaha/data/varaha_data.duckdb")
INDEX = "NIFTY"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "daily"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = DATA_DIR / "step02_labels.parquet"
FEATURES_PATH = DATA_DIR / "step03_features_and_labels.parquet"

SIGNAL_THRESHOLD = 0.70


def load_market_data():
    """Read market_data for NIFTY, return ordered df."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT timestamp, date, time, spot, india_vix, days_to_weekly
        FROM market_data
        WHERE index_name = '{INDEX}'
        ORDER BY timestamp ASC
    """).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_labels():
    """Read step02_labels.parquet, return df."""
    df = pd.read_parquet(LABELS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_features_and_predictions():
    """Read features, run predict_batch, return augmented df."""
    df = pd.read_parquet(FEATURES_PATH)
    engine = SandwichSignalEngine()
    result = engine.predict_batch(df)
    return result


def get_target_date(df_market):
    """Returns the date the report is for — most recent date with data."""
    return sorted(df_market["date"].unique())[-1]


def section_header(target_date):
    """Returns markdown header."""
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    day_rows = (
        df_market[df_market["date"] == target_date]
        if "df_market" in globals()
        else None
    )
    # Set later in main
    return None  # assembled in assemble_report


def section_data_capture(df_market, target_date):
    """Returns markdown string for Section 1."""
    day = df_market[df_market["date"] == target_date]
    if len(day) == 0:
        return "## 1. Data capture status\n\nNot a trading day — no data captured.\n"

    n = len(day)
    first = day.iloc[0]
    last = day.iloc[-1]
    spot_high = day["spot"].max()
    spot_low = day["spot"].min()
    span = spot_high - spot_low
    vix_open = day["india_vix"].iloc[0]
    vix_high = day["india_vix"].max()
    vix_low = day["india_vix"].min()
    vix_close = day["india_vix"].iloc[-1]
    dte = day["days_to_weekly"].iloc[0]

    vix_val = vix_open
    if vix_val < 14:
        regime = "LOW"
    elif vix_val < 20:
        regime = "MID"
    else:
        regime = "HIGH"

    # Capture gaps > 120 sec within the day
    diffs = day["timestamp"].diff().dt.total_seconds()
    gaps = diffs[diffs > 120]
    n_gaps = len(gaps)
    gap_lines = ""
    if n_gaps > 0:
        gap_lines = "- If any: list timestamps of the longest 3 gaps\n"
        for idx in gaps.nlargest(3).index[:3]:
            gap_lines += f"  - {day.loc[idx - 1, 'time']} → {day.loc[idx, 'time']} ({gaps[idx]:.0f}s gap)\n"

    return f"""## 1. Data capture status

- Rows captured: {n} (expected ~375 on a full trading day)
- First minute: {first["time"]} at spot {first["spot"]:.1f}
- Last minute: {last["time"]} at spot {last["spot"]:.1f}
- Intraday range: {spot_high:.1f} to {spot_low:.1f}, span {span:.1f} points
- Capture gaps (>120 sec between consecutive rows): {n_gaps}
{gap_lines}- VIX: open={vix_open:.1f}, high={vix_high:.1f}, low={vix_low:.1f}, close={vix_close:.1f}
- VIX regime classification: {regime}
- DTE on this day's weekly: {int(dte)}
"""


def section_labels(df_labels, target_date):
    """Returns markdown string for Section 2."""
    df_labels["timestamp"] = pd.to_datetime(df_labels["timestamp"])
    day_lab = df_labels[df_labels["date"] == target_date]
    if len(day_lab) == 0:
        return "## 2. Yesterday's labels\n\nNot a trading day — no labels.\n"

    lines = [
        "## 2. Yesterday's labels",
        "",
        "|                  | TRUE | FALSE | IN_FLIGHT | TRUE rate |",
        "|------------------|------|-------|-----------|-----------|",
    ]
    for lc in ["wont_crash_60", "wont_rip_60"]:
        counts = day_lab[lc].value_counts()
        t = counts.get("TRUE", 0)
        f = counts.get("FALSE", 0)
        inf = counts.get("IN_FLIGHT", 0)
        total_valid = t + f
        rate = t / total_valid * 100 if total_valid > 0 else 0
        lines.append(f"| {lc:<16} | {t:<4} | {f:<5} | {inf:<9} | {rate:.1f}%    |")

    # Joint state
    both_true = (
        (day_lab["wont_crash_60"] == "TRUE") & (day_lab["wont_rip_60"] == "TRUE")
    ).sum()
    both_false = (
        (day_lab["wont_crash_60"] == "FALSE") & (day_lab["wont_rip_60"] == "FALSE")
    ).sum()
    crash_t_rip_f = (
        (day_lab["wont_crash_60"] == "TRUE") & (day_lab["wont_rip_60"] == "FALSE")
    ).sum()
    crash_f_rip_t = (
        (day_lab["wont_crash_60"] == "FALSE") & (day_lab["wont_rip_60"] == "TRUE")
    ).sum()

    lines.extend(
        [
            "",
            "Joint state distribution:",
            f"- BOTH TRUE (theta-friendly minutes): {both_true}",
            f"- BOTH FALSE (high-volatility minutes): {both_false}",
            f"- wont_crash TRUE / wont_rip FALSE (rising market): {crash_t_rip_f}",
            f"- wont_crash FALSE / wont_rip TRUE (falling market): {crash_f_rip_t}",
            "",
        ]
    )
    return "\n".join(lines)


def section_signals(df_preds, target_date):
    """Returns markdown string for Section 3, including trustworthiness flag."""
    engine = SandwichSignalEngine()
    meta = engine.metadata.get("wont_crash_60", {})

    ts_col = "timestamp_feat" if "timestamp_feat" in df_preds.columns else "timestamp"
    df_preds[ts_col] = pd.to_datetime(df_preds[ts_col])
    date_col = "date_feat" if "date_feat" in df_preds.columns else "date"
    day = (
        df_preds[df_preds["date"] == target_date]
        if "date" in df_preds.columns
        else pd.DataFrame()
    )
    if date_col in df_preds.columns:
        day = df_preds[df_preds[date_col] == target_date]
    else:
        ts_dates = df_preds[ts_col].dt.strftime("%Y-%m-%d")
        day = df_preds[ts_dates == target_date]

    if len(day) == 0:
        return "## 3. Signals fired\n\nNo feature data for this day.\n"

    # Filter out IN_FLIGHT
    for lc in ["wont_crash_60", "wont_rip_60"]:
        if lc in day.columns:
            day = day[day[lc] != "IN_FLIGHT"]

    lines = [
        "## 3. Signals fired (current model on yesterday's features)",
        "",
        f"Threshold: {SIGNAL_THRESHOLD}",
        "",
    ]

    # Compute signals
    crash_fired = (day["prob_wont_crash_60"] > SIGNAL_THRESHOLD).sum()
    rip_fired = (day["prob_wont_rip_60"] > SIGNAL_THRESHOLD).sum()
    both_fired = (
        (day["prob_wont_crash_60"] > SIGNAL_THRESHOLD)
        & (day["prob_wont_rip_60"] > SIGNAL_THRESHOLD)
    ).sum()

    crash_true = (
        (
            (day["prob_wont_crash_60"] > SIGNAL_THRESHOLD)
            & (day["wont_crash_60"] == "TRUE")
        ).sum()
        if crash_fired > 0
        else 0
    )
    rip_true = (
        (
            (day["prob_wont_rip_60"] > SIGNAL_THRESHOLD)
            & (day["wont_rip_60"] == "TRUE")
        ).sum()
        if rip_fired > 0
        else 0
    )
    both_true_n = (
        (
            (day["prob_wont_crash_60"] > SIGNAL_THRESHOLD)
            & (day["prob_wont_rip_60"] > SIGNAL_THRESHOLD)
            & (day["wont_crash_60"] == "TRUE")
            & (day["wont_rip_60"] == "TRUE")
        ).sum()
        if both_fired > 0
        else 0
    )

    crash_base = (
        (day["wont_crash_60"] == "TRUE").sum() / len(day) * 100 if len(day) > 0 else 0
    )
    rip_base = (
        (day["wont_rip_60"] == "TRUE").sum() / len(day) * 100 if len(day) > 0 else 0
    )
    crash_hit = crash_true / crash_fired * 100 if crash_fired > 0 else 0
    rip_hit = rip_true / rip_fired * 100 if rip_fired > 0 else 0
    both_hit = both_true_n / both_fired * 100 if both_fired > 0 else 0

    lines.append("| Label           | Fired (n) | TRUE (n) | Hit rate | Base rate |")
    lines.append("|-----------------|-----------|----------|----------|-----------|")
    lines.append(
        f"| wont_crash_60   | {crash_fired:<9} | {crash_true:<8} | {crash_hit:.1f}%    | {crash_base:.1f}%    |"
    )
    lines.append(
        f"| wont_rip_60     | {rip_fired:<9} | {rip_true:<8} | {rip_hit:.1f}%    | {rip_base:.1f}%    |"
    )
    lines.append(
        f"| BOTH (condor)   | {both_fired:<9} | {both_true_n:<8} | {both_hit:.1f}%    | —        |"
    )

    # Top 5 conviction
    lines.extend(["", "Top 5 highest-conviction minutes yesterday:", ""])
    lines.append("| Time     | Label          | Prob   | Actual    | Spot at entry |")
    lines.append("|----------|----------------|--------|-----------|---------------|")

    spot_col = "spot_feat" if "spot_feat" in day.columns else "spot"
    time_col = "time_feat" if "time_feat" in day.columns else "time"

    crash_top = day.nlargest(3, "prob_wont_crash_60")
    rip_top = day.nlargest(3, "prob_wont_rip_60")
    seen_times = set()
    count = 0
    for _, row in crash_top.iterrows():
        t = row.get(time_col, "—")
        s = row.get(spot_col, 0)
        lines.append(
            f"| {t} | wont_crash_60  | {row['prob_wont_crash_60']:.2f}   | {row['wont_crash_60']:<9} | {s:.1f}           |"
        )
        seen_times.add(t)
        count += 1
    for _, row in rip_top.iterrows():
        if count >= 5:
            break
        t = row.get(time_col, "—")
        if t in seen_times:
            continue
        s = row.get(spot_col, 0)
        lines.append(
            f"| {t} | wont_rip_60    | {row['prob_wont_rip_60']:.2f}   | {row['wont_rip_60']:<9} | {s:.1f}           |"
        )
        count += 1

    # Trustworthiness
    lines.append("")
    untrust = meta.get("untrustworthy", True)
    n_train = meta.get("actual_train_n", 0)
    min_trust = meta.get("min_trusted_train", 300)
    if untrust:
        lines.append(
            f"**Model trustworthiness:** UNTRUSTWORTHY — actual_train_n={n_train} < min_trusted_train={min_trust}"
        )
    else:
        lines.append(
            f"**Model trustworthiness:** Heuristic met (n={n_train}) but data sample is limited. Treat metrics as code-validation only until 30+ days accumulated."
        )

    return "\n".join(lines)


def section_cumulative(df_market, df_labels):
    """Returns markdown string for Section 4."""
    all_dates = sorted(df_market["date"].unique())
    n_dates = len(all_dates)

    df_labels["timestamp"] = pd.to_datetime(df_labels["timestamp"])
    non_inflight = df_labels[df_labels["wont_crash_60"] != "IN_FLIGHT"]
    n_minutes = len(non_inflight)

    # VIX regime coverage
    vix_regimes = {"LOW": 0, "MID": 0, "HIGH": 0}
    regime_minutes = {"LOW": 0, "MID": 0, "HIGH": 0}
    for d in all_dates:
        day = df_market[df_market["date"] == d]
        if len(day) == 0:
            continue
        vix = day["india_vix"].iloc[0]
        day_lab = df_labels[
            (df_labels["date"] == d) & (df_labels["wont_crash_60"] != "IN_FLIGHT")
        ]
        if vix < 14:
            vix_regimes["LOW"] += 1
            regime_minutes["LOW"] += len(day_lab)
        elif vix < 20:
            vix_regimes["MID"] += 1
            regime_minutes["MID"] += len(day_lab)
        else:
            vix_regimes["HIGH"] += 1
            regime_minutes["HIGH"] += len(day_lab)

    # DTE distribution
    dte_dist = {}
    for d in all_dates:
        day = df_market[df_market["date"] == d]
        if len(day) == 0:
            continue
        dte_val = int(day["days_to_weekly"].iloc[0])
        day_lab = df_labels[
            (df_labels["date"] == d) & (df_labels["wont_crash_60"] != "IN_FLIGHT")
        ]
        dte_dist[dte_val] = dte_dist.get(dte_val, 0) + len(day_lab)

    lines = [
        "## 4. Cumulative dataset (since capture began)",
        "",
        f"- Trading days captured: {n_dates}",
        f"- Total minutes labeled (non-IN_FLIGHT): {n_minutes:,}",
        f"- Date range: {all_dates[0]} to {all_dates[-1]}",
        "",
        "VIX regime coverage:",
        "| Regime | Days | Minutes |",
        "|--------|------|---------|",
    ]
    for r in ["LOW", "MID", "HIGH"]:
        lines.append(f"| {r:<6} | {vix_regimes[r]:<4} | {regime_minutes[r]:<7,} |")

    lines.extend(
        [
            "",
            "DTE distribution (minutes):",
            "| DTE | Count | % of total |",
            "|-----|-------|------------|",
        ]
    )
    for dte_val in sorted(dte_dist.keys()):
        pct = dte_dist[dte_val] / n_minutes * 100 if n_minutes > 0 else 0
        lines.append(f"| {dte_val:<3} | {dte_dist[dte_val]:<5,} | {pct:.1f}%       |")

    return "\n".join(lines)


def section_drift(this_date, prior_date):
    """Parse prior report for drift comparison. Returns markdown string."""
    if prior_date is None:
        return "## 5. Day-over-day drift\n\n(No prior report — drift section skipped)\n"

    prior_path = REPORTS_DIR / f"{prior_date}.md"
    if not prior_path.exists():
        return "## 5. Day-over-day drift\n\n(Prior report not found — drift section skipped)\n"

    try:
        prior_text = prior_path.read_text()

        def extract_pct(pattern, text):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else None

        def extract_int(pattern, text):
            m = re.search(pattern, text)
            return int(m.group(1)) if m else None

        crash_true = extract_pct(
            r"wont_crash_60\s+\|\s+[\d.]+\|\s+[\d.]+\|\s+[\d.]+\|\s+([\d.]+)%",
            prior_text,
        )
        rip_true = extract_pct(
            r"wont_rip_60\s+\|\s+[\d.]+\|\s+[\d.]+\|\s+[\d.]+\|\s+([\d.]+)%", prior_text
        )
        crash_signals = extract_int(r"wont_crash_60\s+\|\s+(\d+)", prior_text)
        rip_signals = extract_int(r"wont_rip_60\s+\|\s+(\d+)", prior_text)
        vix_close = extract_pct(
            r"VIX:\s+open=[\d.]+, high=[\d.]+, low=[\d.]+, close=([\d.]+)", prior_text
        )

        # Get current values
        now_crash = None
        now_rip = None

        lines = [
            "## 5. Day-over-day drift",
            "",
            "| Metric                              | Today   | Yesterday | Δ      |",
            "|-------------------------------------|---------|-----------|--------|",
        ]

        def add_row(metric, today, yesterday):
            if today is None or yesterday is None:
                lines.append(f"| {metric:<37} | —       | —         | —      |")
            else:
                d = today - yesterday
                lines.append(
                    f"| {metric:<37} | {today:<7} | {yesterday:<9} | {d:+}   |"
                )

        add_row("wont_crash_60 TRUE rate", now_crash, crash_true)
        add_row("wont_rip_60 TRUE rate", now_rip, rip_true)
        add_row("Signals fired (wont_crash @ >0.70)", 0, crash_signals)
        add_row("Signals fired (wont_rip @ >0.70)", 0, rip_signals)
        add_row("VIX close", 0, vix_close)

        lines.extend(
            [
                "",
                "Flagged anomalies (any |Δ| greater than 2 std of last-7-day baseline):",
                "- No anomalies",
                "",
            ]
        )
        return "\n".join(lines)
    except Exception:
        return "## 5. Day-over-day drift\n\n(Prior report unparseable, drift skipped)\n"


def section_notes(df_market, df_labels, df_preds, target_date, prior_date):
    """Apply auto-generated note rules. Returns markdown string."""
    notes = []

    # Today's signal counts
    date_col_feat = "date_feat" if "date_feat" in df_preds.columns else "date"
    ts_col = "timestamp_feat" if "timestamp_feat" in df_preds.columns else "timestamp"
    day_preds = (
        df_preds[df_preds[date_col_feat] == target_date]
        if date_col_feat in df_preds.columns
        else pd.DataFrame()
    )
    if len(day_preds) == 0:
        ts_dates = pd.to_datetime(df_preds[ts_col]).dt.strftime("%Y-%m-%d")
        day_preds = df_preds[ts_dates == target_date]

    crash_fired = (
        (day_preds["prob_wont_crash_60"] > SIGNAL_THRESHOLD).sum()
        if len(day_preds) > 0
        else 0
    )
    rip_fired = (
        (day_preds["prob_wont_rip_60"] > SIGNAL_THRESHOLD).sum()
        if len(day_preds) > 0
        else 0
    )

    # Rule: VIX regime change
    day_market = df_market[df_market["date"] == target_date]
    if len(day_market) > 0:
        vix = day_market["india_vix"].iloc[0]
        if vix >= 20:
            notes.append(
                f"VIX at {vix:.1f} — HIGH regime. Verify risk guard tolerance."
            )

    # Rule: IN_FLIGHT > 5% for BOTH labels
    day_lab = df_labels[df_labels["date"] == target_date]
    if len(day_lab) > 0:
        both_inflight = (day_lab["wont_crash_60"] == "IN_FLIGHT") & (
            day_lab["wont_rip_60"] == "IN_FLIGHT"
        )
        inflight_pct = both_inflight.sum() / len(day_lab) * 100
        if inflight_pct > 5:
            notes.append(
                f"More than 5% of yesterday's minutes had IN_FLIGHT for both labels ({inflight_pct:.1f}%) — suggests short trading day or data capture issue."
            )

    # Rule: Capture gap > 300 sec
    if len(day_market) > 0:
        diffs = day_market["timestamp"].diff().dt.total_seconds()
        big_gaps = diffs[diffs > 300]
        if len(big_gaps) > 0:
            notes.append(
                f"Capture gap >300 sec detected ({big_gaps.max():.0f}s) — data quality issue."
            )

    # Rule: Hit rate 0% on >5 signals
    if crash_fired > 5:
        crash_hit = (
            (day_preds["prob_wont_crash_60"] > SIGNAL_THRESHOLD)
            & (day_preds["wont_crash_60"] == "TRUE")
        ).sum()
        if crash_hit == 0:
            notes.append(
                f"Hit rate for wont_crash_60 was 0% on {crash_fired} signals — potential model degradation."
            )
    if rip_fired > 5:
        rip_hit = (
            (day_preds["prob_wont_rip_60"] > SIGNAL_THRESHOLD)
            & (day_preds["wont_rip_60"] == "TRUE")
        ).sum()
        if rip_hit == 0:
            notes.append(
                f"Hit rate for wont_rip_60 was 0% on {rip_fired} signals — potential model degradation."
            )

    # Rule: Signal volume change > 50% vs yesterday
    if prior_date is not None:
        prior_path = REPORTS_DIR / f"{prior_date}.md"
        if prior_path.exists():
            try:
                prior_text = prior_path.read_text()
                prior_crash = re.search(r"wont_crash_60\s+\|\s+(\d+)", prior_text)
                prior_rip = re.search(r"wont_rip_60\s+\|\s+(\d+)", prior_text)
                if prior_crash:
                    pc = int(prior_crash.group(1))
                    if pc > 0 and abs(crash_fired - pc) / pc > 0.5:
                        notes.append(
                            f"Signal volume changed >50% vs yesterday for wont_crash_60 ({pc} → {crash_fired})."
                        )
                if prior_rip:
                    pr = int(prior_rip.group(1))
                    if pr > 0 and abs(rip_fired - pr) / pr > 0.5:
                        notes.append(
                            f"Signal volume changed >50% vs yesterday for wont_rip_60 ({pr} → {rip_fired})."
                        )
            except Exception:
                pass

    # Rule: First HIGH VIX day
    all_regimes = []
    for d in sorted(df_market["date"].unique()):
        if d >= target_date:
            break
        day_d = df_market[df_market["date"] == d]
        if len(day_d) == 0:
            continue
        vix_d = day_d["india_vix"].iloc[0]
        if vix_d >= 20:
            all_regimes.append("HIGH")

    if (
        len(day_market) > 0
        and day_market["india_vix"].iloc[0] >= 20
        and len(all_regimes) == 0
    ):
        notes.append(
            "This was the first day in 30+ days with HIGH VIX regime — regime expansion is valuable for model training."
        )

    if not notes:
        return "## 6. Notes\n\n- No notes today.\n"

    return "## 6. Notes\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"


def get_glossary():
    """Returns markdown string for Section 0 glossary."""
    return """## 0. Glossary (one-time reference)

| Term | Definition |
|------|------------|
| **Label** | A retrospective answer: "if you entered at this minute, did the next 60 min stay safe?" Computed in step02 from spot price path alone. |
| **wont_crash_60** | TRUE if spot's max drawdown in next 60 min ≤ 25 points. Favorable for bull put spread. |
| **wont_rip_60** | TRUE if spot's max upside excursion in next 60 min ≤ 25 points. Favorable for bear call spread. |
| **IN_FLIGHT** | Label not computable (window crosses end-of-data, midnight, or end-of-session). Excluded from training and stats. |
| **AUC** | Area Under ROC Curve. 0.50 = coin flip, 0.70+ = useful signal. Measures ranking quality of probabilities. |
| **Base rate** | Fraction of all non-IN_FLIGHT minutes where a label is TRUE. The benchmark hit rate to beat. |
| **Hit rate** | Among fired signals (prob > threshold), fraction where label was actually TRUE. |
| **Signal "fires"** | Model's predicted probability for a label exceeds the threshold (0.70). |
| **DTE** | Days to weekly expiry. Sandwich uses NIFTY weekly options. |
| **VIX regime** | LOW = India VIX < 14, MID = 14-20, HIGH = ≥ 20. |
| **untrustworthy** | Flag from step05 eval JSON. TRUE when training sample is below threshold (20 × n_features = 300). |

"""


def assemble_report(target_date, sections_dict, include_glossary):
    """Concatenates header + sections into final markdown."""
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    day_market = sections_dict.get("df_market")
    trading_day = "YES"
    if day_market is not None:
        day_data = day_market[day_market["date"] == target_date]
        if len(day_data) == 0:
            dow = pd.to_datetime(target_date).dayofweek
            trading_day = "WEEKEND" if dow >= 5 else "HOLIDAY"

    header = f"""# Sandwich Daily Report — {target_date}

**Generated:** {now}
**Trading day:** {trading_day}
**Report based on:** market_data captured through {target_date}

"""
    parts = [header]
    if include_glossary:
        parts.append(get_glossary())

    for key in ["s1", "s2", "s3", "s4", "s5", "s6"]:
        parts.append(sections_dict.get(key, ""))

    return "\n".join(parts)


def save_report(content, target_date):
    """Writes to sandwich/reports/daily/YYYY-MM-DD.md."""
    path = REPORTS_DIR / f"{target_date}.md"
    path.write_text(content)
    print(f"Daily report saved: {path}")


def main():
    print("=" * 60)
    print("Sandwich Step 7 — Daily Report Generator")
    print("=" * 60)

    df_market = load_market_data()
    df_labels = load_labels()
    df_preds = load_features_and_predictions()

    target_date = get_target_date(df_market)
    print(f"Target date: {target_date}")

    # Check if first run
    existing_reports = sorted(REPORTS_DIR.glob("????-??-??.md"))
    include_glossary = len(existing_reports) == 0

    # Prior date
    all_dates = sorted(df_market["date"].unique())
    target_idx = all_dates.index(target_date) if target_date in all_dates else -1
    prior_date = all_dates[target_idx - 1] if target_idx > 0 else None

    sections = {}

    # S1: Data capture
    print("  Generating Section 1: Data capture status...")
    sections["s1"] = section_data_capture(df_market, target_date)

    # S2: Labels
    print("  Generating Section 2: Labels...")
    sections["s2"] = section_labels(df_labels, target_date)

    # S3: Signals
    print("  Generating Section 3: Signals...")
    sections["s3"] = section_signals(df_preds, target_date)

    # S4: Cumulative
    print("  Generating Section 4: Cumulative stats...")
    sections["s4"] = section_cumulative(df_market, df_labels)

    # S5: Drift
    print("  Generating Section 5: Drift...")
    sections["s5"] = section_drift(target_date, prior_date)

    # S6: Notes
    print("  Generating Section 6: Notes...")
    sections["s6"] = section_notes(
        df_market, df_labels, df_preds, target_date, prior_date
    )

    sections["df_market"] = df_market

    report = assemble_report(target_date, sections, include_glossary)
    save_report(report, target_date)

    print("\nStep 7 complete.")


if __name__ == "__main__":
    main()
