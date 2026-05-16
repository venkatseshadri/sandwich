# Sandwich POC — Complete Implementation Review

**For Claude to revisit and verify.** All files below exist at `github.com/venkatseshadri/sandwich`.

## Architecture

Sandwich is a 6-step pipeline that mines paper-trade data to produce probabilistic entry signals for NIFTY ATM credit spreads:

```
Data Capture (external) → Step 1 (inspect) → Step 2 (label) → Step 3 (features) → Step 4 (lift) → Step 5 (model) → Step 6 (API deliverable)
```

## Data Source

**Only v3 / v3.1 DuckDB data.** The `market_data` table at `/home/trading_ceo/python-trader/varaha/data/varaha_data.duckdb`.
- `market_data` — 1-min spot, VIX, DTE, ATM strike (104 columns available, 7 used)
- `option_snapshots` — strike-wise OI, LTP, IV for weekly expiry options (joined on timestamp)

**v4 multi-timeframe data is NOT used.** The spec explicitly excludes it: `market_data_multitf` lives in a separate DB and is declared out of scope for the POC. The 41 features are all computed from 1-min `market_data` joined with `option_snapshots` — no higher-timeframe bars, no SMA50/200, no MACD, no OBV/CMF.

## Feature families

All 41 features at 1-minute resolution, computed deterministically in pandas:

| Family | Count | Features |
|--------|-------|----------|
| TREND | 12 | EMA5/20/50, spot vs EMA ratios, EMA slopes, higher_high/lower_low_20, above_VWAP, VWAP distance |
| MOMENTUM | 6 | RSI(14) Wilder, RSI slope, ROC(5/15/60), consecutive-up bars |
| VOLATILITY | 5 | ATR(14) proxy, ATR pct, realized vol 30m annualized, VIX, VIX change 1d |
| TIME | 4 | minute_of_day, minutes_since_open, minutes_to_close, day_of_week |
| EXPIRY | 3 | DTE, is_dte0, is_dte_le2 |
| OPTIONS | 8 | ATM CE/PE IV, IV skew, PCR OI, OI change 5m, max pain offset, ATM OI changes |
| REGIME | 3 | VIX low/mid/high flags (currently all "mid" — 15.9-19.0) |

## Labels

Three binary labels per timestamp, each with a 60-minute future window:

| Label | Meaning | Base rate | TRUE count |
|-------|---------|-----------|------------|
| `wont_crash_60` | Max drawdown in next 60 min ≤ 25 points | 37.6% | 955 |
| `wont_rip_60` | Max excursion in next 60 min ≤ 25 points | 48.4% | 1,229 |
| `range_60` | Spot stays in ±25 pt band for 60 min | 4.2% | 106 |

IN_FLIGHT when insufficient future bars within the same trading date.

## Models

3 GradientBoostingClassifier models (sklearn), each wrapped in CalibratedClassifierCV (isotonic, 3-fold):

```
GBM_PARAMS = {n_estimators: 200, max_depth: 3, learning_rate: 0.05, subsample: 0.8, random_state: 42}
```

Time-series split: first 70% (May 4-11 morning) → train, last 30% (May 11 afternoon - May 15) → test.

**Results (untrustworthy — 8 trading days, single VIX regime):**

| Label | AUC | Top-decile hit | >0.70 signals |
|-------|-----|----------------|---------------|
| wont_crash_60 | 0.54 | 46.8% vs 43.8% base | 156 |
| wont_rip_60 | 0.29 | 32.9% vs 47.9% base | 0 |
| range_60 | 0.50 | 6.1% vs 5.4% base | 0 |

## API contract

```python
from signal_api import SandwichSignalEngine
engine = SandwichSignalEngine()

# Single snapshot
signal = engine.predict({
    "trend_ema_5": 24000.0,
    "trend_ema_20": 23980.0,
    # ... all 41 features ...
}, timestamp="2026-05-15T11:05:00")
# → {"timestamp": "...", "probabilities": {"wont_crash_60": 0.372, ...},
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
│   ├── step03_features.py             # 12 functions, 41 features
│   ├── step04_base_rates_and_lift.py  # 12 functions, lift + breakdowns
│   ├── step05_model.py                # 7 functions, GBM + calibration
│   └── step06_signal_api_test.py      # Test harness, smoke test
└── data/
    ├── step02_labels.parquet           # 3,039 × 9 columns, 57.5 KB
    ├── step03_features_and_labels.parquet  # joined dataset, 784 KB
    ├── step04_*.csv                    # 16 CSV files
    ├── step05_model_*.pkl              # 3 pickle files (~50 KB each)
    ├── step05_medians_*.json           # 3 JSON files
    ├── step05_eval_*.json              # 3 JSON files
    ├── step05_feature_columns.json     # Feature order (41 columns)
    ├── step06_signal_log.csv           # 156 signals > 0.70
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

## To run the full pipeline

```bash
cd /home/trading_ceo/sandwich
python3 steps/step01_connect_and_inspect.py   # verify DB
python3 steps/step02_labeler.py               # compute labels
python3 steps/step03_features.py              # 41 features
python3 steps/step04_base_rates_and_lift.py   # lift tables
python3 steps/step05_model.py                 # train models
python3 steps/step06_signal_api_test.py       # API + trade log
```
