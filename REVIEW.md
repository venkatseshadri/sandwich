# Sandwich POC — Complete Implementation Review

**Commit:** `5fe426f` | **Date:** 2026-05-16 | **Branch:** `master`
**For Claude to revisit and verify.** All files below exist at `github.com/venkatseshadri/sandwich`.

## Data Source — v3.1 ONLY (v4 explicitly excluded)

**Sandwich POC reads exclusively from v3.1 DuckDB data.** The `market_data` and `option_snapshots` tables in `/home/trading_ceo/python-trader/varaha/data/varaha_data.duckdb` (populated by `data_capture_v3.1_duckdb.py`).

### What IS used
- `market_data` — 1-min NIFTY spot, VIX, DTE, ATM strike (columns: timestamp, date, time, spot, india_vix, days_to_weekly, atm_strike)
- `option_snapshots` — weekly expiry CE/PE chain (columns: timestamp, strike_offset, option_type, oi, iv, ltp)

### What is NOT used
- **v4 multi-timeframe aggregator** (`data_capture_v4_queue_aggregator.py`) — populates `market_data_multitf` table in a **separate** DuckDB file (`market_data_multitf.duckdb`). This table has SMA20/50/200, MACD, ADX+DI, Bollinger Bands, OBV, CMF, CCI across 5/15/30/60/240/1440-minute timeframes. **Explicitly out of scope for POC.** The spec states: _"market_data_multitf (in a separate varaha_data_multitf.duckdb, NOT in scope for this POC)."_
- **v4 multitf aggregator** (`data_capture_v4_multitf_aggregator.py`) — simplified batch version writing to the same `varaha_data.duckdb`. Also excluded.

### Why
The POC goal is to validate the pipeline architecture end-to-end with minimal data. Adding multi-timeframe features (SMA across 5m/15m/30m/1h/4h/1D) would:
1. Require a second DB connection and cross-DB joining logic
2. Inflate the feature count from 41 to ~80+ (duplicating trend/momentum families per timeframe)
3. Multiply the data sparsity problem — 8 days of 1-min data is marginal; 8 days of 1D bars is useless
4. Create maintenance coupling to the separate v4 DB schema

### Future integration path (NOT in scope)
When production accumulation has 30+ trading days:
1. Join `market_data_multitf` on `(timestamp, index_name)` for higher-TF context
2. Add multi-TF features: `stf_5m_ema20`, `stf_15m_sma50`, etc.
3. Run Steps 4-6 on the expanded feature set with proper walk-forward validation

## Architecture

Sandwich is a 6-step pipeline that mines paper-trade data to produce probabilistic entry signals for NIFTY ATM credit spreads:

```
Data Capture (external) → Step 1 (inspect) → Step 2 (label) → Step 3 (features) → Step 4 (lift) → Step 5 (model) → Step 6 (API deliverable)
```

## Revision Status (Post-POC R1+R2+R3)

| Revision | Change | Status |
|----------|--------|--------|
| R1 | Drop range_holds label (iron condor not in playbook) | ✅ |
| R2 | 15-feature lean balanced panel (was 41) | ✅ |
| R3 | Document Wing Optimizer as future varaha scope | ✅ |

## Feature families (R2: 15 features, 9 families)

Reduced from 41 features to 15, balanced across families, to prevent implicit family-count bias in the model. Family count no longer correlates with family importance. The remaining 15 features each carry a clear hypothesis.

| # | Feature | Family | Computation |
|---|---------|--------|-------------|
| 1 | `trend_1m` | Trend | -1/0/+1: spot vs EMA20 + EMA20 slope on 1-min |
| 2 | `trend_5m` | Trend | -1/0/+1: same logic resampled to 5-min bars |
| 3 | `trend_1h` | Trend | -1/0/+1: same logic resampled to 60-min bars |
| 4 | `trend_1d` | Trend | -1/0/+1: daily close direction + 5-day slope |
| 5 | `mom_rsi_14` | Momentum | Wilder RSI(14) on 1-min spot |
| 6 | `vol_atr_pct` | Volatility | ATR(14) / spot, using abs(spot.diff()) as TR |
| 7 | `vol_india_vix` | Volatility | India VIX as captured |
| 8 | `vwap_distance` | Volume | (spot - intraday_vwap) / spot. Uniform 1/min volume proxy |
| 9 | `opt_atm_iv` | Options | IV of ATM CE from option_snapshots |
| 10 | `opt_pcr_oi` | Options | sum(PE OI for offsets -5..+5) / sum(CE OI same range) |
| 11 | `opt_atm_oi_change_5m` | Options | ATM CE OI delta over 5 minutes |
| 12 | `time_minutes_since_open` | Time | minute_of_day - 555 |
| 13 | `expiry_dte` | Expiry | days_to_weekly from market_data |
| 14 | `sr_dist_to_yday_high` | Structure | (yesterday_high - spot) / spot |
| 15 | `sr_dist_to_yday_low` | Structure | (spot - yesterday_low) / spot |

No family has more than 4 features. Multi-TF trend colors replace raw EMA stacks.

## Labels (R1: 2 labels, range_holds removed)

Two binary labels per timestamp, each with a 60-minute future window:

| Label | Meaning | Base rate | TRUE count |
|-------|---------|-----------|------------|
| `wont_crash_60` | Max drawdown in next 60 min ≤ 25 points | 37.6% | 955 |
| `wont_rip_60` | Max excursion in next 60 min ≤ 25 points | 48.4% | 1,229 |

IN_FLIGHT when insufficient future bars within the same trading date.

**Why 2 labels not 3:** Iron condor not in trading playbook. Two independent single-side signals (wont_crash for PE-side bull put, wont_rip for CE-side bear call) cover the trading universe. Both firing simultaneously composes into a condor at the strategy layer, not at the signal layer.

## Models

2 GradientBoostingClassifier models (sklearn), each wrapped in CalibratedClassifierCV (isotonic, 3-fold):

```
GBM_PARAMS = {n_estimators: 200, max_depth: 3, learning_rate: 0.05, subsample: 0.8, random_state: 42}
```

Time-series split: first 70% (May 4-11 morning) → train, last 30% (May 11 afternoon - May 15) → test.

**Results (untrustworthy — 8 trading days, single VIX regime):**

| Label | AUC | Top-decile hit | >0.70 signals |
|-------|-----|----------------|---------------|
| wont_crash_60 | 0.52 | 55.6% vs 43.8% base | 0 |
| wont_rip_60 | 0.39 | 26.0% vs 47.9% base | 0 |

AUC near coin-flip on 15 lean features — expected. No >0.70 signals generated.

### Latest correlation pairs (|corr| > 0.85)

Only 1 pair — vs 18 pairs in the 41-feature version. Features are deliberately orthogonal:

| feat_a | feat_b | correlation |
|--------|--------|-------------|
| sr_dist_to_yday_high | sr_dist_to_yday_low | -0.966 |

(Expected: they are symmetric measures from the same source.)

### Latest Step 4 lift rankings

Top 3 features by lift spread for wont_crash_60:
```
time_minutes_since_open  lift_spread=0.648
trend_5m                 lift_spread=0.642
opt_atm_iv               lift_spread=0.635
```

Top 3 for wont_rip_60:
```
vol_india_vix            lift_spread=0.589
opt_atm_iv               lift_spread=0.501
trend_1m                 lift_spread=0.438
```

## API contract

```python
from signal_api import SandwichSignalEngine
engine = SandwichSignalEngine()

# Single snapshot (15 features, 2 labels)
signal = engine.predict({
    "trend_1m": 1, "trend_5m": 0, "trend_1h": 1, "trend_1d": -1,
    "mom_rsi_14": 52.0, "vol_atr_pct": 0.00025, "vol_india_vix": 18.5,
    "vwap_distance": -0.001, "opt_atm_iv": 15.2, "opt_pcr_oi": 0.95,
    "opt_atm_oi_change_5m": 15000, "time_minutes_since_open": 75,
    "expiry_dte": 4, "sr_dist_to_yday_high": 0.005, "sr_dist_to_yday_low": 0.008,
}, timestamp="2026-05-15T11:05:00")
# → {"timestamp": "...", "probabilities": {"wont_crash_60": 0.407, "wont_rip_60": 0.431},
#     "model_metadata": {"untrustworthy": True}}

# Batch
pred_df = engine.predict_batch(features_dataframe)
```

## File inventory

```
sandwich/
├── README.md                          # Project overview, all steps logged
├── .gitignore                         # __pycache__ excluded
├── signal_api.py                      # SandwichSignalEngine class
├── __init__.py                        # Package marker (empty)
├── reports/
│   ├── step01_report.md               # DuckDB connect verification
│   ├── step02_report.md               # Labeler output + spot-checks
│   ├── step03_report.md               # Feature panel + correlation pairs
│   ├── step04_report.md               # Base rates, lift, breakdowns
│   ├── step05_report.md               # Model eval, calibration check
│   └── step06_report.md               # API test, POC closeout
├── steps/
│   ├── __init__.py
│   ├── step01_connect_and_inspect.py   # 5 functions, DuckDB read + plot
│   ├── step02_labeler.py              # 5 functions, labeler + validate
│   ├── step03_features.py             # 10 functions, 15 features (R2)
│   ├── step04_base_rates_and_lift.py  # 11 functions, 2-label lift + breakdowns
│   ├── step05_model.py                # 7 functions, 2-model GBM + calibration
│   └── step06_signal_api_test.py      # Test harness, smoke test
└── data/
    ├── step02_labels.parquet           # 3,039 × 8 columns, 57 KB
    ├── step03_features_and_labels.parquet  # joined dataset, 251 KB
    ├── step04_*.csv                    # 12 CSV files (2 labels)
    ├── step05_model_*.pkl              # 2 pickle files
    ├── step05_medians_*.json           # 2 JSON files
    ├── step05_eval_*.json              # 2 JSON files
    ├── step05_feature_columns.json     # Feature order (15 columns)
    ├── step06_signal_log.csv           # Signal log
    └── step06_summary.json            # Aggregate stats
```

## Design decisions

1. **No imports from antariksh.** All DB access via raw DuckDB SQL. Zero coupling.
2. **No try/except.** Errors surface visibly. Pipeline halts, doesn't silently skip.
3. **Plain print().** No logging framework. stdout is the reportable output.
4. **Constants at top of file.** No argparse, no CLI flags, no YAML configs.
5. **Surgical functions.** Each step has exactly the functions listed in the spec. No shared utils.py.
6. **Time-series split, never random.** Train on earlier dates, test on later dates.
7. **Same-date constraint on labels.** Labels never span overnight gaps.
8. **VWAP proxy.** `market_data` has no volume column. VWAP uses uniform 1-per-minute proxy.
9. **ATR proxy.** No high/low in schema. ATR(14) uses `abs(spot.diff())` as True Range.
10. **Feature NaNs imputed with median.** From training split, applied to test. Medians saved for predict-time use.

## What's missing (by design — for production phase)

- v4 multi-timeframe data (higher-TF trend context)
- More than 8 trading days of data (real inference needs 30+)
- VIX low (<14) and high (>20) regime samples
- Proper walk-forward cross-validation (single split in POC)
- Rolling model retraining
- Model serialization versioning
- Feature importance / SHAP analysis
- Real paper-trade outcomes as labels (currently labels are derived from spot path, not actual trade P&L)

## Wing Optimizer (future varaha scope, NOT Sandwich)

The Wing Optimizer is a separate planned module in the data capture layer (varaha). At each minute (or every 5 minutes), it evaluates the live option chain and computes optimal wing width for PE-side and CE-side credit spreads, optimizing return-on-capital subject to SPAN margin constraints.

Output columns (added to market_data):
- `best_pe_wing_short`, `best_pe_wing_long`, `best_pe_breakeven_cushion`
- `best_ce_wing_short`, `best_ce_wing_long`, `best_ce_breakeven_cushion`
- `best_ce_roc`, `best_pe_roc`

Sandwich would then consume these as additional features. Wing Optimizer does NOT depend on Sandwich; Sandwich consumes Wing Optimizer's output.

## To run the full pipeline

```bash
cd /home/trading_ceo/sandwich
python3 steps/step01_connect_and_inspect.py   # verify DB
python3 steps/step02_labeler.py               # compute labels
python3 steps/step03_features.py              # 15 features
python3 steps/step04_base_rates_and_lift.py   # lift tables
python3 steps/step05_model.py                 # train models
python3 steps/step06_signal_api_test.py       # API + trade log
```

## Latest pipeline execution (2026-05-16 ~16:00 IST)

```
Step 1:  3,039 rows, 27.5 MB DB, 60s cadence, zero nulls
Step 2:  2 labels, 955 TRUE/1,584 FALSE/500 IN_FLIGHT (wont_crash)
                       1,229 TRUE/1,310 FALSE/500 IN_FLIGHT (wont_rip)
Step 3:  15 features, 3,039 × 15 joined, 251 KB parquet, 1 correlation pair
Step 4:  base rates 37.6% (wont_crash) / 48.4% (wont_rip)
Step 5:  2 models, AUC 0.52 / 0.39, logloss 0.72 / 0.76, all untrustworthy
Step 6:  SandwichSignalEngine operational, 0 signals >0.70 in test set
```

## Commit history

```
5fe426f R3: document Wing Optimizer as future varaha scope
ad93b68 R2: lean 15-feature panel balanced across 9 families
b0dc222 R1: drop range_holds label, iron condor not in playbook
90a1283 docs: clarify v3.1-only data source, v4 explicitly excluded per spec
b0f3125 docs: add full implementation review for Claude revisit
b8f1264 Add .gitignore, remove __pycache__
590854a Step 6: signal API and POC closeout
b0f9294 Step 5: model training — GradientBoosting + calibration
02a03a0 Step 4: base rates and feature lift
464b72f Step 3: feature panel — 41 features across 7 families
0eabde2 Step 2: labeler — wont_crash, wont_rip, range_holds with IN_FLIGHT
08309fd Cleanup: consolidate docs, move Step 1 report to reports/
```
