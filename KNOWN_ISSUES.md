# Sandwich — Known Issues (post full code review, 2026-05-17)

**STATUS UPDATE (2026-05-19):** All 5 identified issues have been fixed. See FIXES_APPLIED.md for details.

---

## ✅ Fixed Issues (2026-05-19)

### ✅ Bug-S1: `untrustworthy` hardcoded to `True` — FIXED
**File:** signal_api.py (predict method)
- Now reads from metadata: `sample_meta.get("untrustworthy", False)`
- Secondary check: flags untrustworthy if >50% features imputed
- Added `untrustworthy_reason` and `pct_features_imputed` fields for transparency
- Smoke test verified: full features → untrustworthy=False, partial features → untrustworthy=True

### ✅ Bug-S3: Silent feature imputation — FIXED
**File:** signal_api.py (predict method)
- Features now tracked when >50% imputed
- Returns `untrustworthy_reason: "imputation:XX%"` when threshold exceeded
- Smoke test 2 verifies imputation reporting with partial feature dicts
- Callers can detect and handle partial snapshots appropriately

### ✅ Bug-4A: Buckets per (feature, label) pair — FIXED
**File:** step04_base_rates_and_lift.py
- Buckets now computed once per feature (lines 158-159)
- Same buckets reused across all labels via `precomputed_buckets` parameter
- Fair comparison of feature lift across labels ensured

### ✅ Bug-4B: `bucket_feature()` silent degradation — FIXED
**File:** step04_base_rates_and_lift.py (bucket_feature function)
- Logging added when qcut drops buckets (lines 110-112)
- Specific ValueError exception handling (no silent coercion to strings)
- Failed bucketing returns NaN (not silent defaults)
- Test run: step04 executes with no warnings

### ✅ Test Issue: Partial feature dicts — FIXED
**File:** step06_signal_api_test.py
- Smoke test 2 (lines 60-81) deliberately omits all `opt_*` features
- Verifies imputation reporting matches expected omitted features
- Confirms API handles partial snapshots correctly

---

## Verification

All 6 pipeline steps passing (2026-05-19):
- step06 smoke tests: ✅ PASS (full + partial features)
- step04 base rates: ✅ PASS (no degradation, all features bucketed)
- Batch predict: ✅ PASS (762 rows, hit rates validated)

---

## Deferred (Production Hardening)

None currently. All identified issues resolved.

**When next review needed:**
- After 30+ trading days of data (metrics stabilization check)
- Before production integration with Antariksh
- If new edge cases discovered in live trading

---

See FIXES_APPLIED.md for detailed before/after comparison.
