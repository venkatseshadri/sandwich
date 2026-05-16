# Step 3 Report — Feature Panel

## Files added
- sandwich/steps/step03_features.py
- sandwich/data/step03_features_and_labels.parquet
- sandwich/data/step03_correlation_matrix.csv
- sandwich/data/step03_correlation_pairs.csv

## Execution
Wall time: ~4 min
Rows in feature panel: 3,039
Total feature columns: 41

## Null counts per feature (top 10 by null count)
```
opt_iv_skew                398
opt_atm_iv                 382
opt_atm_pe_iv              381
vol_vix_change_1d          375
opt_atm_ce_oi_change_5m    370
opt_atm_pe_oi_change_5m    370
opt_total_oi_change_5m     365
opt_pcr_oi                 365
opt_max_pain_offset        365
mom_roc_60                  60
```
Most nulls are at the beginning of the series (not enough prior bars for lookback features: 375 bars for VIX change, 60 bars for ROC). Option IV is 0.0 in many snapshots — treat as NaN.

## Feature distribution sanity
```
       trend_ema_20   mom_rsi_14   vol_atr_14   expiry_dte   opt_pcr_oi
count    3039.000     3036.000     3038.000      3039.000      2674.000
mean     24019.30     49.05        6.18          3.45          0.96
min      23567.12     7.14         2.44          0.00          0.31
max      24425.24     97.63        24.44         6.00          4.27
```

## Correlation pairs with |corr| > 0.85
```
feat_a                     feat_b                    correlation
time_minute_of_day         time_minutes_since_open    1.0000
time_minutes_since_open    time_minutes_to_close     -1.0000
time_minute_of_day         time_minutes_to_close     -1.0000
vol_atr_14                 vol_atr_14_pct            0.9997
trend_ema_5                trend_ema_20              0.9973
trend_ema_20               trend_ema_50              0.9961
trend_ema_5                trend_ema_50              0.9888
trend_spot_vs_ema20        trend_spot_vs_ema50       0.9036
trend_spot_vs_ema20        mom_roc_15                0.8956
expiry_dte                 expiry_is_dte_le2        -0.8944
trend_spot_vs_ema5         mom_roc_5                 0.8749
trend_spot_vs_ema20        trend_ema5_slope          0.8698
trend_spot_vs_ema20        mom_rsi_14                0.8644
trend_ema_5                vol_vix                  -0.8598
trend_ema5_slope           mom_roc_5                 0.8560
vol_atr_14_pct             vol_realized_30           0.8543
trend_ema_20               vol_vix                  -0.8541
vol_atr_14                 vol_realized_30           0.8528
```
18 pairs with |corr| > 0.85. Easiest removals: time_minutes_since_open (linear transform of minute_of_day), vol_atr_14_pct (linear of vol_atr_14). The EMA correlation (0.99) is expected from 1-min data.

## VWAP source used
Computed via uniform 1-per-minute volume proxy (intraday expanding mean per date). Market_data schema has no volume column.

## Anomalies
- Option IV (opt_atm_iv, opt_atm_pe_iv) is 0.0 in many snapshots — Black-Scholes IV may not be computed in capture. PCR and OI data IS populated for most rows.
- Range of 2674 rows with valid PCR vs 3039 total — 365 rows (~12%) have no option chain data at that timestamp
- 18 feature pairs at |corr| > 0.85: time features are redundant (3 collinear), EMA crossovers highly correlated (expected on 1-min data), ATR/ATRpct essentially identical
- Mom_roc_60 has 60 nulls (~2%) — first 60 minutes of first trading day

## Status
Step 3 complete. Proceeding to Step 4.
