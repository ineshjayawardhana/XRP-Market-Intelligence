# Market Conditions Research

This educational historical study adds **Market Conditions Research** to the dashboard sidebar. It does not train models, change prediction validation, modify news annotations or place trades.

The subsequent consistency audit is [audit.md](reports/market_conditions/audit.md), with detailed machine-readable checks and block-length sensitivity in [audit.json](reports/market_conditions/audit.json). Run `python audit_market_conditions.py` to repeat it. A rebuild of the study also refreshes the audit automatically.

The source is **43,967 stored hourly records**, expanded after the original **17,688-row** model dataset; 17,688 overlapping records agree and are not counted twice. One stored candle is partial, leaving **43,966 complete hourly candles**. Feature/known-target filtering leaves 43,582 hourly rows; prior-volatility warm-up leaves 42,142. The anchored 25-hour grid selects 1,671 origins: 350 reference, one boundary purge and 1,320 descriptive observations. No count correction was needed.

The descriptive observations **do overlap already evaluated ML tests**: 176 study intervals overlap the 175 production test intervals, and 176 overlap the historical experiment reserve. Their frequencies are not independent model validation and those periods cannot become untouched tests again. Unfinished registry reserves are now excluded from reference/evaluation/news results; the dashboard withholds stale saved results that conflict with a newly declared reserve. There are currently no active unfinished reserves.

The audit verified source timestamps, exact target returns, prior-only volatility thresholds and zero duplicate/overlapping study intervals. It also removed a silent independent-row bootstrap fallback and added 3/7/14-origin block sensitivity. Intervals remain approximate under long-lived or changing regimes; the charts do not fully eliminate dependence uncertainty.

```powershell
# Rebuild the descriptive study from existing local candles; no network or model fitting.
.\.venv\Scripts\python.exe market_conditions_study.py

# Launch the dashboard and choose Market Conditions Research.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

# Run automated tests.
.\.venv\Scripts\python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v
```

Study outputs: [report.md](reports/market_conditions/report.md), [report.json](reports/market_conditions/report.json), and `reports/market_conditions/observations.csv`. The dashboard reads these saved outputs without rebuilding the study every refresh. Missing files show a rebuild instruction; market/news/model views remain independent.

## Actual findings

The evaluation contains **1,320 non-overlapping 24-hour outcomes**, with origins from 1 January 2023 to 14 November 2026 UTC. Outcomes are UP at least +1%, DOWN at most −1%, and NEUTRAL between those boundaries. Counts are UP 442, DOWN 461 and NEUTRAL 417.

| Condition | Observations | UP | DOWN | NEUTRAL |
|---|---:|---:|---:|---:|
| Bullish | 531 | 32.8% | 40.9% | 26.4% |
| Bearish | 645 | 34.4% | 30.4% | 35.2% |
| Sideways | 144 | 31.9% | 33.3% | 34.7% |
| High volatility | 330 | 37.9% | 40.9% | 21.2% |

Bullish/bearish/sideways are mutually exclusive. High volatility overlaps those groups and its count must not be added to their total.

The trend names describe earlier moving averages, not what price must do next. In this sample, bullish conditions were followed by DOWN more often than UP. This is not a reversal recommendation or evidence for a tradable strategy. High-volatility observations had fewer NEUTRAL outcomes, but volatile conditions did not establish a reliable direction.

Examples of pointwise 95% moving-block intervals:

- Bullish → UP: **28.2%–36.8%**, estimate 32.8%.
- Bullish → DOWN: **36.6%–45.0%**, estimate 40.9%.
- High volatility → NEUTRAL: **17.2%–26.4%**, estimate 21.2%.

Year slices vary substantially: the observed UP frequency during high volatility was 32.9% in 2023, 46.7% in 2024, 39.7% in 2025 and 28.2% in 2026. These post-hoc year comparisons are descriptive and have unequal samples; they are not a validated conditional forecasting rule.

## Past-only regime definitions

- Bullish: `(EMA20 − EMA50) / current close > 0.001`.
- Bearish: the same gap is below −0.001.
- Sideways: the gap lies between those fixed boundaries.
- High volatility: current rolling 24-hour return volatility exceeds the 75th percentile of preceding hourly volatility observations over 2,160 hours (90 days), shifted one hour. At least 720 earlier observations are required. The threshold is never fitted using future or whole-sample volatility.

The study reuses existing causal indicator/feature calculations. The original public cache has one missing hour and one partial candle near 24 March 2023. Partial candles are excluded and all indicators, rolling windows and targets restart in contiguous segments. No candles, prices or volumes are interpolated. Regime labels use only information available at the origin; actual 24-hour returns are joined afterward as outcomes.

Origins follow the fixed global 25-hour grid used by earlier research. Their 24-hour outcomes do not overlap. Unknown future outcomes, gap-crossing windows and insufficient warm-up observations are excluded. Shared feature histories and persistent regimes still create dependence despite this spacing.

## Baselines and uncertainty

The reference period is 2022. Its 350 matured origins contain UP 109, DOWN 137 and NEUTRAL 104, so the fixed reference-majority baseline is DOWN. Reference target ends strictly precede the later evaluation. It is not reselected using evaluation outcomes or individual regime labels.

On the same 1,320 evaluation origins:

- Reference-majority historical match rate: **34.9%**, block interval **32.5%–37.0%**.
- Existing EMA/RSI technical-rule match rate: **32.4%**, block interval **30.0%–35.0%**.

These match rates compare historical descriptions; they do not revalidate either prediction model. Baseline results within every regime and year are provided in JSON and the dashboard.

The report supplies counts, empirical class frequencies, nominal Wilson intervals and 1,500-replicate, seeded, seven-origin moving-block intervals. Bootstrap blocks never bridge missing source/warm-up periods. Conditional regime frequencies are recomputed after resampling the chronological observations rather than resampling each regime as independent rows.

Wilson intervals assume independent Bernoulli observations and may understate uncertainty here. Blocks preserve some local dependence but cannot capture all long market cycles or structural breaks. Intervals are pointwise, without multiple-comparison correction, and are not confidence in a future event. Groups below 30 observations are flagged as small; the year filter helps reveal changing composition rather than prove an effect.

## News association audit

The existing Gold/Silver datasets are read without modification. Eligibility requires matching article identity, protected human decisions, explicit label availability, and acquired-content timing. Feature availability is the later relevant timestamp; publication dates, GDELT first-seen and filesystem dates do not substitute for when a label/context became available.

Only **one** record has enough timing evidence. It has **no matured, gap-free 24-hour outcome in the saved price dataset**, so there are **zero usable event/control pairs**. No measurable news association or event effect is claimed.

The event-study path uses the first completed candle at or after availability, within one hour. It never assigns an earlier entry price. It requires a complete next-24-hour window, removes overlapping event windows, and selects unique earlier same-trend controls within 90 days without matching on future returns. Gold/Silver provenance stays explicit. A minimum of 30 pairs across ten event days is required before reporting an exploratory pooled difference. That count is a research policy, not proof of statistical power or causation; further stratification and confounding checks would still be needed.

## Dashboard and preservation

The new view contains:

- An outcome-frequency chart with labeled classes and uncertainty whiskers.
- Counts, frequency intervals and small-sample flags.
- A UTC-year filter with SLST display timestamps.
- Same-observation baseline tables.
- An interactive historical price/regime timeline with distinct shapes and colors.
- Plain-English explanations, news eligibility and method limitations.

Existing market overview, live news, EMA/RSI/MACD charts, appearance modes, research tools and experimental-model probability gate remain available. The new section never turns historical frequencies into validated future probabilities.

Before edits, `app.py` was backed up under `data/backups/regime_20261116T0625296986402Z/`. The adjacent manifest records 107 existing data/model artifacts for byte-preservation checks. Model files, model validation/report pointers, human labels, SKIP decisions, news features and prior research artifacts were not edited. Only the dashboard navigation was extended to show this separate saved study.

The full automated suite passed **190 tests with zero failures**. Eight new tests cover prefix/future-data invariance, exact horizons, missing/partial candles, bootstrap gap boundaries, frequencies/baselines, reference chronology, honest news entry timing and the new Streamlit view with missing data and both themes. Automated UI checks passed; browser screenshot inspection was unavailable. The final preservation check found **zero changes among all 107 protected data/model files**. The existing model remains experimental with the same failed balanced-accuracy check.

After the audit changes, the full suite passed **196 tests**. Six additional tests cover interval overlap despite different origins, reference-boundary purging and active-reserve exclusion, evaluated-versus-unfinished reserve handling, refusal of an independent bootstrap fallback, raw/partial/duplicate inventory and exact boundary protection. Source datasets and model artifacts remain byte-identical.

All earlier data has already been exposed during research. This is a descriptive outcome study, not a new untouched predictive evaluation. Crypto market history, news timing and retrospective associations do not establish future financial performance.

This is for education only, not financial advice. Crypto is high risk.
