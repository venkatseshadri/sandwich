# Step 6 Report — Signal API

## Files added
- sandwich/signal_api.py
- sandwich/steps/step06_signal_api_test.py
- sandwich/data/step06_signal_log.csv
- sandwich/data/step06_summary.json

## API smoke test
```
{
  "wont_crash_60": 0.372,
  "wont_rip_60": 0.4468,
  "range_60": 0.0396
}
```
Single-dict prediction works. Returns calibrated probabilities + untrustworthy flag.

## Batch test on held-out data
Rows processed: 762

Signal fire counts:
  wont_crash_60 > 0.70: 156 times
  wont_rip_60 > 0.70:   0 times
  range_60 > 0.70:      0 times

Hit rates (where signal fired):
  wont_crash_60 fires that were actually TRUE: 73 / 156 = 46.79%
  base rate for comparison: 43.83%
  
  (wont_rip_60 and range_60 never breach 0.70 threshold — models produce low-confidence probabilities)

## End-to-end check
Imported sandwich.signal_api from outside the steps/ directory: OK

## POC STATUS
Sandwich POC end-to-end runnable. Models trained on insufficient data; 
edge metrics are not inference-valid. Pipeline architecture is verified.

## Pipeline summary
```
Step 1 (✓): DuckDB connect — 3,039 rows, 60s cadence, zero nulls
Step 2 (✓): Labeler — 3 labels, wont_crash/wont_rip/range_60, IN_FLIGHT handled
Step 3 (✓): Features — 41 features across 7 families, 18 correlations found
Step 4 (✓): Lift — base rates 37.6% / 48.4% / 4.2%, feature lift ranked
Step 5 (✓): Model — 3 GradientBoosting models, calibrated, time-split
Step 6 (✓): API — SandwichSignalEngine, dict + batch predict, trade log
```

## Next phase (NOT in scope here)
- Accumulate 2-3 months of capture data
- Re-run Steps 4-6 on richer data
- Validate edge with proper walk-forward
- Integration with Antariksh / Brahmand execution layer

## Anomalies
- wont_rip_60 and range_60 never produce >0.70 predictions — model calibration yields narrow probability ranges (AUC≈0.3-0.5)
- 156 wont_crash_60 signals at >0.70 with 46.79% hit rate vs 43.83% base rate — this is a slight edge but NOT statistically significant (156 signals, 3% lift)

## Status
POC complete. Awaiting Venkat review.
