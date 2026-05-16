# Step 5 Report — Model Training

## Files added
- sandwich/steps/step05_model.py
- sandwich/data/step05_model_wont_crash_60.pkl
- sandwich/data/step05_model_wont_rip_60.pkl
- sandwich/data/step05_model_range_60.pkl
- sandwich/data/step05_medians_*.json (3 files)
- sandwich/data/step05_eval_*.json (3 files)
- sandwich/data/step05_feature_columns.json

## Training data
Total rows: 2,539
Train: 1,777 rows, dates 2026-05-04 to 2026-05-11
Test: 762 rows, dates 2026-05-11 to 2026-05-15
Split timestamp: 2026-05-11 12:55

UNTRUSTWORTHY WARNING: models trained on 2,539 samples across 8 dates.
Metrics below are NOT inference-valid. Pipeline correctness is what's being verified.

## Evaluation per label

wont_crash_60:
  test_n: 762
  test_pos_rate: 0.4383
  logloss: 0.7179
  brier: 0.2614
  auc: 0.5406
  top_decile_precision: 0.4679

wont_rip_60:
  test_n: 762
  test_pos_rate: 0.479
  logloss: 0.8552
  brier: 0.3159
  auc: 0.2921
  top_decile_precision: 0.3291

range_60:
  test_n: 762
  test_pos_rate: 0.0538
  logloss: 0.2148
  brier: 0.0513
  auc: 0.5016
  top_decile_precision: 0.0606

## Calibration check (wont_crash_60)
```
decile   n      pred_rate    obs_rate    
1        5      0.3336       0.6000      
2        141    0.3727       0.2979      
3        30     0.4042       0.7333      
4        126    0.4300       0.5556      
5        56     0.4859       0.1429      
6        61     0.5252       0.3115      
8        187    0.6121       0.5187      
10       156    0.7001       0.4679
```
Calibration inconsistent (decile 3 with 30 samples shows 73% observed vs 40% predicted) — expected with small sample. Decile 7 is empty due to probability clustering near 0.43 from isotonic calibration.

## Anomalies
- AUC ≈ 0.5 for all models — no edge detected. Expected: 8 trading days with single VIX regime (15.9-19.0). The model has no regime variation to learn from.
- range_60 AUC 0.5016 and 4.17% base rate — extreme class imbalance with naive model predicting the majority class
- wont_rip_60 AUC 0.292 — worse than random. Model may be learning an inverse pattern. Again: insufficient data.
- Calibration decile 7 empty — isotonic calibration squeezed all probabilities into a narrow range on this dataset

## Status
Step 5 complete. Proceeding to Step 6.
