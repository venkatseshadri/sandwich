# Sandwich

Entry-signal research project. Produces probabilistic directional/path signals for downstream credit-spread strategies.

## Status

Step 2: Labeler — in progress.

## Layout

- `steps/` — one standalone `.py` script per step, numbered
- `reports/` — step report-back artifacts
- `data/` — parquet, CSV, JSON, pickle outputs
- (other folders added as the project earns them)

## How to run

Each step is a standalone Python script. From the repo root:

    python sandwich/steps/step01_connect_and_inspect.py
    python sandwich/steps/step02_labeler.py

## Reports

- [Step 1 Report](reports/step01_report.md) — Connect & Inspect

## Rules

- Sandwich does not import from or modify existing antariksh code.
- Each step is a single `.py` script with strict acceptance criteria.
- No Jupyter notebooks. Standalone scripts only.
- New steps are not started until the previous step is reported back and acknowledged.

## Steps log

- Step 1: Connect & Inspect — ✅ complete (2026-05-16)
- Step 2: Labeler — in progress
- Step 3: Feature Panel — not started
- Step 4: Base Rates & Feature Lift — not started
- Step 5: Model Training — not started
- Step 6: Signal API — not started
