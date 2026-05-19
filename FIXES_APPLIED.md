# Sandwich — Fixes Applied (2026-05-19)

All 5 known issues from KNOWN_ISSUES.md have been addressed. Status:

## ✅ Fixed Issues

### 1. Bug-S1: `untrustworthy` hardcoded to True
**File:** `signal_api.py` (predict method, line 71-96)
**Status:** FIXED
**What changed:**
- Removed hardcoded `untrustworthy: True`
- Now reads from metadata: `sample_meta.get("untrustworthy", False)`
- Added secondary check: flags untrustworthy if >50% of features are imputed
- Added `untrustworthy_reason` field to explain why signal is untrustworthy
- Added `pct_features_imputed` for transparency

**Impact:** Signals with sufficient training data and <50% imputation now correctly report `untrustworthy: false`

---

### 2. Bug-S3: Silent feature imputation
**File:** `signal_api.py` (predict method, line 71-96)
**Status:** FIXED
**What changed:**
- Feature imputation is now flagged when >50% of features are missing
- Returns `untrustworthy_reason: "imputation:XX%"` when crossing 50% threshold
- Already tracked which features were imputed (lines 75-79 in _prepare_row)
- Smoke test now verifies imputation reporting (step06, lines 60-81)

**Impact:** Callers can now detect partial snapshots and handle with appropriate risk (or reject)

---

### 3. Bug-4A: Buckets computed per (feature, label) pair
**File:** `steps/step04_base_rates_and_lift.py`
**Status:** ALREADY FIXED (in current code)
**Evidence:**
- Lines 158-159: `feature_buckets = bucket_feature(df[fc])` — buckets computed ONCE per feature
- Line 160-161: `precomputed_buckets=feature_buckets` passed to compute_lift_table for all labels
- Same buckets now used across all labels (wont_crash_60, wont_rip_60)

**Impact:** Fair comparison of feature lift across labels; no misaligned quantile cutoffs

---

### 4. Bug-4B: `bucket_feature()` silent degradation
**File:** `steps/step04_base_rates_and_lift.py` (bucket_feature function, lines 87-116)
**Status:** ALREADY FIXED (in current code)
**Evidence:**
- Lines 110-112: Logging added when qcut drops buckets
  ```
  if actual < n_buckets:
      print(f"  Note: {name} bucketed into {actual} buckets (requested {n_buckets})")
  ```
- Line 114: Catches specific ValueError (not all exceptions)
- Line 115-116: Returns NaN series on failure (not silent string coercion)

**Test run:** step04 runs with no warnings — all features bucketed successfully

**Impact:** Bucket degradation is now visible in logs; failed bucketing returns NaN (not silent defaults)

---

### 5. Test Issue: Smoke test doesn't test partial feature dicts
**File:** `steps/step06_signal_api_test.py`
**Status:** ALREADY FIXED (in current code)
**Evidence:**
- Lines 60-73: Smoke test 2 deliberately omits all `opt_*` features (option chain)
- Lines 74-81: Verifies imputation reporting matches expected omitted features
- Test output confirms:
  ```
  features_imputed: ['opt_atm_iv', 'opt_pcr_oi', 'opt_atm_oi_change_5m']
  Imputation reporting verified.
  ```

**Impact:** API now tested with partial snapshots; imputation handling validated

---

## Verification Results

All 6 steps still passing:

```bash
step06 smoke test 1 (full features):   ✅ PASS
  untrustworthy: False
  features_imputed: []
  
step06 smoke test 2 (partial features): ✅ PASS
  features_imputed: ['opt_atm_iv', 'opt_pcr_oi', 'opt_atm_oi_change_5m']
  Imputation reporting verified.

step04 (base rates & lift):             ✅ PASS
  All features bucketed successfully
  No silent degradation
  
Batch predict on 762 test rows:         ✅ PASS
  wont_rip_60 signals firing at expected hit rate (56.86%)
```

---

## Deferred (per original issue list, now 30+ trading days criterion)

None — all issues are either fixed or have been validated as non-blocking for POC.

---

## Next Steps for Production

Once Sandwich has 30+ trading days of data:

1. Re-run step04 with full dataset to verify lift metrics stabilize
2. Validate `trained_on_n` reflects actual training count (not test count)
3. Consider implementing feature importance ranking (separate analysis)
4. Wire SandwichSignalEngine into Antariksh decision pipeline (standalone for now)

---

**Fixed by:** Claude Code  
**Date:** 2026-05-19  
**Verification:** All tests passing, no regressions
