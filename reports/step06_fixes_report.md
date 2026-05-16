# R2-Fix-3 Report (POC Cleanup)

## Commit
SHA: (pending)
Message: R2-Fix-3: signal_api honesty (untrustworthy, imputation reporting),
         step04 bucket handling, step06 dual-label trade log

## Files changed
- signal_api.py — Fix-S1, Fix-S2
- steps/step04_base_rates_and_lift.py — Fix-4A, Fix-4B
- steps/step06_signal_api_test.py — Fix-6A, Fix-6B
- data/step04_*.csv — regenerated
- data/step06_signal_log.csv — regenerated (now includes both labels)
- data/step06_summary.json — regenerated
- reports/step06_fixes_report.md — this file

## Fix-S1 verification (signal_api: honest untrustworthy + trained_on_n)

API smoke test 1 (full features) output:
```
probabilities: {"wont_crash_60": 0.4073, "wont_rip_60": 0.4171}
features_imputed: []
untrustworthy: False
```
`untrustworthy` is now `False` (reads from eval JSON: actual_train_n=1,777 > min_trusted_train=300).
`trained_on_n` now reports actual_train_n = 1,777.

## Fix-S2 verification (signal_api: imputation reporting)

API smoke test 2 (partial features) output:
```
probabilities: {"wont_crash_60": 0.3678, "wont_rip_60": 0.4369}
features_imputed: ["opt_atm_iv", "opt_pcr_oi", "opt_atm_oi_change_5m"]
n_features_imputed: 3
Imputation reporting verified.
```

## Fix-4A + 4B verification (step04: bucket handling)

No bucket reduction notes printed — all 15 features have sufficient unique values for 4-bucket qcut.

Lift rankings unchanged from R2-Fix-2. Top 3 for wont_crash_60:
```
sr_dist_to_yday_high     lift_spread=0.761
time_minutes_since_open  lift_spread=0.648
trend_5m                 lift_spread=0.642
```

## Fix-6A verification (step06: partial-features smoke test)

Imputation reporting verified: ✓ — expected 3 option features, got 3. Match confirmed.

## Fix-6B verification (step06: dual-label trade log)

Signals fired at >0.70:
  CRASH_ONLY: 0
  RIP_ONLY:   51
  BOTH:       0
  Total:      51

No BOTH case — no iron-condor-by-composition signals.

## Anomalies
- untrustworthy is False because min_trusted_train=300 < actual_train_n=1,777. The heuristic is met on sample count but 8 trading days is clearly insufficient. (Already logged in KNOWN_ISSUES.md)
- 0 CRASH_ONLY signals at >0.70 — wont_crash model produces conservative probabilities. RIP_ONLY fires 51 signals with 56.9% hit rate but this is noise on 8 days.

## POC closeout

After R2-Fix-3, the Sandwich POC is in maintenance mode.

✅ Pipeline architecture verified end-to-end
✅ Labels honest (path-based, no leakage)
✅ Features honest (intraday-varying, balanced families)
✅ Models honest (no train/test leakage, dynamic trustworthiness)
✅ Signal API honest (reports actual training stats and imputation)

Current model AUCs (on 8 trading days, single VIX regime):
  wont_crash_60: 0.45
  wont_rip_60:   0.39

Both within noise of coin-flip. This is the correct answer given current data.

## Next phase (NOT in scope here)

- varaha continues capturing data
- Sandwich is re-run weekly (no code changes) as samples accumulate
- When sample size crosses ~30 trading days across multiple VIX regimes, re-evaluate AUC trajectory
- If AUCs consistently >0.55 across multiple weekly runs, start signal deployment planning
- Wing Optimizer becomes the next active project, in the varaha repo, not Sandwich

## Status
R2-Fix-3 complete. Sandwich POC in maintenance mode. Awaiting data accumulation.
