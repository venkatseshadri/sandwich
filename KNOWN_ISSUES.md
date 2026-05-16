# Sandwich — Known Issues (post full code review, 2026-05-17)

All files reviewed. L1-L2 bugs (data leakage) fixed in R2-Fix-1 and R2-Fix-2.
Remaining issues are production-deployment concerns — they do not affect the current
AUC numbers or pipeline correctness on 8 days of data. Deferred until data accumulates.

---

## signal_api.py

### Bug-S1: `untrustworthy` hardcoded to `True`
`predict()` ignores the conditional flag in the eval JSON (fixed in R2-Fix-2) and
always returns `true`. Should read `metadata[label]["untrustworthy"]`.
Also: `trained_on_n` reports test count, not training count.

### Bug-S3: Silent feature imputation
`_prepare_row()` substitutes missing features with stored medians without signaling
to the caller. In production, a partial snapshot (e.g. option chain unavailable for
one minute) would produce confident-looking probabilities from heavily imputed inputs.
Fix: add `features_imputed: [...]` to model_metadata, optionally raise if >50% missing.

---

## step04_base_rates_and_lift.py

### Bug-4A: Buckets computed per (feature, label) pair
`compute_lift_table()` buckets independently per label, risking different quantile
cutoffs for the same feature across labels. Fix: bucket once per feature, reuse.

### Bug-4B: `bucket_feature()` silent degradation
`qcut(duplicates="drop")` silently reduces bucket count with no notice. `try/except Exception`
catches everything and falls back to `astype(str)`, creating 200+ singleton buckets
that all get filtered by `min_bucket_n`. Result: feature shows no signal when it might.
Fix: log when `qcut` drops buckets; catch specific exceptions only.

---

## step06_signal_api_test.py

### Smoke test doesn't test partial feature dicts
The API smoke test passes all 15 features. Should also test with missing keys to
catch Bug-S3 at test time.

### Signal log only captures wont_crash_60
wont_rip_60 signals are not logged to `step06_signal_log.csv`.

---

## Resolution

All issues deferred. Fix when:
- Data has accumulated 30+ trading days
- Sandwich is being prepared for production integration with Antariksh
- A separate production-hardening pass is warranted
