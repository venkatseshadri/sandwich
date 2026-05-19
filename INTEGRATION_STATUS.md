# Sandwich — Local Integration Status (2026-05-19)

## Location

```
/home/trading_ceo/sandwich/              ← Standalone project repo
├── signal_api.py                        ← Main API (SandwichSignalEngine class)
├── steps/                               ← 7-step pipeline (all runnable)
│   ├── step01_connect_and_inspect.py
│   ├── step02_labeler.py
│   ├── step03_features.py
│   ├── step04_base_rates_and_lift.py
│   ├── step05_model.py
│   ├── step06_signal_api_test.py
│   └── step07_daily_report.py
├── data/                                ← Models + medians (pickle/JSON)
│   ├── step05_model_wont_crash_60.pkl
│   ├── step05_model_wont_rip_60.pkl
│   ├── step05_medians_wont_crash_60.json
│   ├── step05_medians_wont_rip_60.json
│   ├── step05_eval_*.json
│   └── step05_feature_columns.json
├── reports/
│   ├── step0{1-7}_report.md             ← Per-step reports
│   └── daily/YYYY-MM-DD.md              ← Daily market reports
├── README.md
├── KNOWN_ISSUES.md
└── FIXES_APPLIED.md                     ← Just added (v2 of known issues)
```

---

## Codebase References

### Current Status: ZERO Integration
Sandwich is **completely standalone** — no imports in antariksh/ or brahmand/.

**Search results:**
```bash
grep -r "sandwich\|Sandwich\|SandwichSignal" /home/trading_ceo/antariksh/*.py
  → (no matches)
  
grep -r "sandwich\|Sandwich\|SandwichSignal" /home/trading_ceo/brahmand/*.py
  → (no matches)
```

**Why:** Per design rule in README.md: *"Sandwich does not import from or modify existing antariksh code."*

---

## What Sandwich Does

**Purpose:** Generates probabilistic trading signals for credit-spread strategies.

**Inputs:** 15 market features (Trend, Momentum, Vol, Greeks, Support/Resistance, etc.)

**Outputs:** Two 60-minute horizon probabilities:
- `wont_crash_60`: P(index won't crash in next 60 min)
- `wont_rip_60`: P(index won't rip/spike in next 60 min)

**Training:** 8 trading days, ~3,000 labeled snapshots per label

---

## How to Use Sandwich (Standalone)

### 1. Load and predict
```python
from sandwich.signal_api import SandwichSignalEngine

engine = SandwichSignalEngine()
features_dict = {
    "trend_1m": 0.5,
    "trend_5m": 0.45,
    "mom_rsi_14": 60.0,
    "vol_atr_pct": 1.2,
    # ... (13 more features)
}
signal = engine.predict(features_dict, timestamp="2026-05-19T14:30:00")

print(signal["probabilities"])
# {'wont_crash_60': 0.72, 'wont_rip_60': 0.65}

print(signal["model_metadata"]["untrustworthy"])
# False (reads metadata, not hardcoded)

print(signal["model_metadata"]["features_imputed"])
# [] (no imputation if all features provided)
```

### 2. Batch predict
```python
pred_df = engine.predict_batch(features_df)
# Returns DataFrame with prob_wont_crash_60 and prob_wont_rip_60 columns
```

### 3. Run daily report
```bash
cd /home/trading_ceo/sandwich
python3 steps/step07_daily_report.py
# Creates reports/daily/YYYY-MM-DD.md with market summary
```

---

## Integration Opportunity (Future)

Sandwich could feed into Antariksh decision chain:

### Option A: Risk Agent Tool
Add SandwichSignalTool to CrewAI risk_agent:
```python
# In tools/risk_tools.py
class SandwichSignalTool(BaseTool):
    name: str = "query_sandwich_signals"
    description: str = "Get probabilistic crash/rip signals for next 60 minutes"
    
    def _run(self, features_dict: dict) -> str:
        from sandwich.signal_api import SandwichSignalEngine
        engine = SandwichSignalEngine()
        signal = engine.predict(features_dict)
        return json.dumps(signal)
```

**Use case:** Risk agent queries Sandwich for sideways/crash risk → tightens SL if crash likely.

### Option B: Pattern Fusion
Combine with PatternAnalyzer (6-TF traffic light):
- PatternAnalyzer: historical regime (trending/sideways)
- Sandwich: forward-looking risk (crash/rip probability)
- Result: More confident SL/TP decisions

---

## Data Dependencies

Sandwich features source:
- **Trend (4 TFs):** market_data_multitf.duckdb (v4)
- **Momentum:** option_snapshots + Greeks (v3.1)
- **Vol (ATR, VIX):** market_data + option_snapshots
- **Greeks (IV, PCR, OI):** option_snapshots
- **Support/Resistance:** market_data

**Current status:** ✅ All data available in local varaha_data.duckdb and market_data_multitf.duckdb

---

## Known Limitations (POC)

1. **Data volume:** Only 8 trading days (→ high variance in metrics)
2. **Feature engineering:** 15 features; future iteration could include Wing Optimizer output
3. **Horizon:** Fixed 60 minutes; could parameterize for 15m/30m/4h
4. **Timezone:** Assumes IST (hardcoded in some reports)

---

## Next Steps if Integrating

1. **Month 1 (May-Jun):** Accumulate 30+ trading days → stabilize metrics
2. **Validation:** Run parallel backtests (Sandwich signals vs actual outcomes)
3. **Wiring:** If AUC validates, add SandwichSignalTool to CrewAI
4. **Learning:** Correlate Sandwich predictions with TSL outcomes → refine lock_ratio
5. **Production:** Replace POC models with production-trained models (June+)

---

**Current Status:** POC complete ✅, standalone ✅, ready for integration review
**Owner:** Research (separate from trading pipeline)
**Last Updated:** 2026-05-19 (all 5 issues fixed)
