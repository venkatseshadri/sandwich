# Sandwich

Entry-signal research project. Produces probabilistic directional/path signals for downstream credit-spread strategies.

## Status

POC complete ✅ (2026-05-16). All 6 steps done.

## Layout

- `steps/` — one standalone `.py` script per step, numbered
- `reports/` — step report-back artifacts
- `data/` — parquet, CSV, JSON, pickle outputs
- (other folders added as the project earns them)

## How to run

Each step is a standalone Python script. From the repo root:

    python sandwich/steps/step01_connect_and_inspect.py
    python sandwich/steps/step02_labeler.py
    python sandwich/steps/step03_features.py
    python sandwich/steps/step04_base_rates_and_lift.py
    python sandwich/steps/step05_model.py
    python sandwich/steps/step06_signal_api_test.py

## Reports

- [Step 1 Report](reports/step01_report.md) — Connect & Inspect
- [Step 2 Report](reports/step02_report.md) — Labeler
- [Step 3 Report](reports/step03_report.md) — Feature Panel
- [Step 4 Report](reports/step04_report.md) — Base Rates & Lift
- [Step 5 Report](reports/step05_report.md) — Model Training
- [Step 6 Report](reports/step06_report.md) — Signal API

## Rules

- Sandwich does not import from or modify existing antariksh code.
- Each step is a single `.py` script with strict acceptance criteria.
- No Jupyter notebooks. Standalone scripts only.
- New steps are not started until the previous step is reported back and acknowledged.

## Steps log

- Step 1: Connect & Inspect — ✅ complete (2026-05-16)
- Step 2: Labeler — ✅ complete (2026-05-16)
- Step 3: Feature Panel — ✅ complete (2026-05-16)
- Step 4: Base Rates & Feature Lift — ✅ complete (2026-05-16)
- Step 5: Model Training — ✅ complete (2026-05-16)
- Step 6: Signal API — ✅ complete (2026-05-16)

## Daily reports

After R3-Daily, Sandwich generates a daily markdown report at:
`sandwich/reports/daily/YYYY-MM-DD.md`

Run manually with:
python3 steps/step07_daily_report.py

The first report includes a glossary. Subsequent reports omit it for brevity.

The report is designed for two readers:
1. The user, glancing on a phone in the morning (~30 sec to scan)
2. Claude, when the user pastes the report for review and trend detection

## Review status

All 7 source files independently reviewed (step01–step06 + signal_api).
No remaining L1/L2 bugs. [Known issues deferred](KNOWN_ISSUES.md) for production phase.

## Future scope (NOT in Sandwich)

The Wing Optimizer is a separate, planned module that lives in the data 
capture layer (varaha), not in Sandwich. Its job: at each minute (or 
every 5 minutes), evaluate the live option chain and compute the optimal 
wing width for both PE-side and CE-side credit spreads, optimizing return-
on-capital subject to SPAN margin constraints.

Wing Optimizer output (best_pe_wing_short, best_pe_wing_long, 
best_pe_breakeven_cushion, etc.) becomes additional columns in market_data, 
which Sandwich then consumes as input features in a future iteration.

Wing Optimizer does NOT depend on Sandwich. Sandwich consumes Wing 
Optimizer's output, not the other way around.
