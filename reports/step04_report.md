# Step 4 Report — Base Rates & Feature Lift

## Files added
- sandwich/steps/step04_base_rates_and_lift.py
- sandwich/data/step04_base_rates.csv
- sandwich/data/step04_lift_wont_crash_60.csv, step04_lift_wont_rip_60.csv, step04_lift_range_60.csv
- sandwich/data/step04_tod_*.csv, step04_regime_*.csv, step04_dte_*.csv

## Sample size note
Current dataset: 2,539 labeled rows, after dropping 500 IN_FLIGHT.
Window: 2026-05-04 to 2026-05-15 (8 trading dates, 2 unique weeks).
VIX range covered: 15.9 to 19.0 (entirely mid regime — no low/high samples).

INSUFFICIENT DATA WARNING: Statistical findings below are reported for code 
validation only. Real inference requires accumulating more data across multiple 
regimes and expiries.

## Base rates (with sample sizes)
```
           label  base_rate     n  ci_low  ci_high
0  wont_crash_60     0.3761  2539  0.3575   0.3951
1    wont_rip_60     0.4840  2539  0.4646   0.5035
2       range_60     0.0417  2539  0.0346   0.0502
```
wont_crash: 37.6% base rate. wont_rip: 48.4%. range_60: 4.2% (25-point band is tight at VIX 16-19).

## Top 5 features by lift spread — wont_crash_60
```
                    feature  max_lift  min_lift  lift_spread
35      opt_max_pain_offset    1.4152    0.6331       0.7821
23       time_minute_of_day    1.3211    0.6730       0.6481
30               opt_atm_iv    1.4782    0.8432       0.6350
19           vol_atr_14_pct    1.3397    0.7368       0.6029
18               vol_atr_14    1.2986    0.7071       0.5915
```
Option chain features (max pain, ATM IV) show meaningful lift spread. Time-of-day and vol also separate.

## Top 5 features by lift spread — wont_rip_60
```
             feature  max_lift  min_lift  lift_spread
21           vol_vix    1.2621    0.6734       0.5887
30        opt_atm_iv    1.1689    0.6676       0.5013
19    vol_atr_14_pct    1.2657    0.7658       0.4999
18        vol_atr_14    1.2624    0.7789       0.4835
26  time_day_of_week    1.1481    0.7050       0.4431
```
VIX itself is the strongest lift feature for rip prediction. Volatility features dominate.

## Top 5 features by lift spread — range_60
```
                feature  max_lift  min_lift  lift_spread
20      vol_realized_30    2.3293    0.2295       2.0998
19       vol_atr_14_pct    2.3414    0.3399       2.0015
18           vol_atr_14    2.3414    0.3777       1.9637
35  opt_max_pain_offset    1.8822    0.0000       1.8822
26     time_day_of_week    2.0945    0.3794       1.7151
```
Volatility features have >2x lift spread for range — intuitive: lower vol = tighter range. 0.0 min_lift on max_pain_offset means some offset bucket has ZERO range_holds TRUE samples (bucket too small).

## Time-of-day breakdown for wont_crash_60
Largest buckets: 13:45 (n=266), 12:15 (n=265), 12:45 (n=263). P(true) varies 32-54% across buckets.

## VIX regime breakdown
Currently single-regime in dataset: entire window in "mid" (15.9-19.0). No low/high comparisons possible.

## DTE breakdown
DTE-3 has only 21.4% wont_crash (tight day?), DTE-1 at 45.4%. Variability likely driven by the specific market conditions on each expiry date rather than DTE itself.

## Anomalies
- ALL data falls in VIX "mid" regime — the regime breakdown decomposes cleanly but has no signal. Need VIX >20 or <14 data.
- range_60 has only 106 TRUE out of 2539 — small base rate makes lift estimates noisy
- DTE-3 shows anomalous low wont_crash rate (21.4%) — likely a single volatile trading day rather than a DTE effect

## Status
Step 4 complete. Proceeding to Step 5.
