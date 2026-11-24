# XRP Market Intelligence Dashboard

Run from this project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Stop with Ctrl+C. The existing environment and dependencies are sufficient; no model training is needed to launch.

## Views

- **Overview:** Binance spot price, rolling 24-hour change/high/low/volume, a plain-English market-condition card with three reasons, and linked price/EMA, RSI and MACD charts.
- **News intelligence:** the existing real corpus and permitted publisher RSS items, with article links, source, separate publication/observation times, human versus provisional status and uncalibrated score types. Filters cover category, status and publisher. Recent category counts require both timestamps to be recent.
- **Advanced research:** saved Gold/Silver readiness and class counts, existing news-classifier metrics with provenance, on-demand 30/90/180-day historical outcome counts and same-event technical/combined comparisons. The existing headline classifier remains available in the headline lab.
- **Indicator guide:** beginner explanations of the indicators, timestamps, news labels and research limitations.

Use the sidebar to switch Dark/Light appearance, stop public news polling or request a refresh. The active view reruns every 30 seconds while the browser session is open. Quotes are cached for 25 seconds, hourly candles for 55 seconds, publisher feeds for five minutes and historical evaluations' price input for one hour. An hourly candle is not a tick-by-tick quote.

All visible times and chart axes use **Asia/Colombo, SLST (UTC+05:30)**. Stored data and research joins remain UTC.

## Current-condition rules

These are transparent technical heuristics, not trained forecasts:

- EMA20/EMA50 separation must exceed 0.1% of price and price must agree with the faster average to establish a trend vote.
- MACD histogram uses a small price-relative tolerance. RSI provides momentum context; RSI at or beyond 30/70 triggers caution.
- Volume uses the existing 1.8 times prior-20-hour-average rule. Unusually high opposing volume raises caution.
- News direction is considered only with at least three classified stories from two publishers, both published and observed within six hours. Provisional content categories are unverified context. Dominant speculative coverage raises caution.
- Aligned EMA/MACD and compatible RSI support Bullish/Bearish conditions. Conflicting or incomplete evidence gives Mixed / Insufficient Evidence. Balanced evidence gives Neutral.

The three reasons explain trend, momentum, and participation/news. They never instruct a trade. Price alone does not establish a trend.

## Data freshness and failure behavior

Quotes older than 90 seconds, exchange ticker timestamps older than 120 seconds, or the latest completed hourly candle older than 90 minutes limit the outlook. Future, incomplete, duplicate, unordered, gapped or invalid candle data is rejected. A failed market refresh also limits the outlook, even when a retained snapshot exists.

Last successful snapshots keep their original timestamps and appear with warnings. No replacement price is generated. Missing article publication times stay missing. A successful feed check or a newly observed archived article does not make its publication recent. News markers use first-available times and the preceding hourly price as an approximation; they do not imply causation.

Public RSS polling uses the existing robots-aware, rate-limited collector client with conditional requests and persistent cooldowns. Sources are Cointelegraph's XRP tag feed and The Daily Hodl's Ripple/XRP feed. Access failures are shown in News intelligence, while saved news remains available. No paywall or Google robots restrictions are bypassed.

Only `data/dashboard_cache/` is written by normal dashboard use. It holds last-good market snapshots, publisher items and publisher access state. Original CSVs, SQLite research history, Gold/Silver files, human decisions, backups and trained models are read without modification. Set `XRP_DASHBOARD_CACHE_DIR` to override this cache location. `XRP_DASHBOARD_OFFLINE=1` disables network requests and displays existing snapshots/archive data with appropriate warnings.

## What is and is not validated

The bundled `artifacts/news_tfidf.joblib` has three content classes and a recorded **88% classification accuracy on 50 synthetic test examples**. Its metadata does not record an evaluation timestamp or a chronological real-news test. This is displayed only as a synthetic teaching baseline. It is not real-news accuracy, price accuracy or calibrated prediction confidence. The current environment reports that this artifact was saved with scikit-learn 1.8.0 and is loaded with 1.9.1; a smoke inference succeeded, but version-compatible validation is still needed. No package or model was replaced.

The existing saved readiness report contains 400 sourced records, 8 Gold human annotations, 215 Silver provisional labels, 174 unresolved cases and 3 SKIP decisions. After duplicate handling there are 396 usable annotation candidates, not 396 training-ready examples. Gold has no Real_Adoption examples. There are 52 missing publication timestamps, source imbalance and insufficient chronological/class coverage. TF-IDF real-news training and DistilBERT training are currently blocked by the existing readiness checks; optional transformer dependencies are also absent.

There is now a separately trained, evaluated 24-hour direction model, described in [TREND_RESEARCH_README.md](TREND_RESEARCH_README.md). It achieved 0.3400 balanced accuracy on 175 non-overlapping held-out origins, below the predeclared 0.38 quality threshold. Its class output is experimental. The dashboard therefore still says:

> Predictive probabilities are not yet validated.

UP / DOWN / NEUTRAL use the existing observed-return definitions: at least +1%, at most -1%, or between those boundaries over 24 hours. These thresholds define outcomes, not forecast confidence. No forecast percentages are fabricated from indicators, headline scores or classifier accuracy.

`forecast_validation.py` now supports the v2 research contract with frozen data/model hashes, reproducible causal features and metrics, strict target-maturity purging, non-overlapping origins and quality comparisons against training-derived baselines. It distinguishes present-day historical replay computation from the simulated historical information cutoff. Fresh inference uses the exact evaluated model and passes through the display gate in memory; no manually written forecast file is needed. Older v1 contracts remain backwards compatible. See the trend guide for the complete current display policy and limitations.

Further work needs untouched chronological market periods and consistently timestamped news across time. Expand Gold examples without backdating their availability. Unreviewed AI labels never become verified evaluation ground truth. Market-direction training is a separate explicit workflow; this dashboard never starts it.

## Historical evaluation limitations

The advanced view fetches real closed Binance candles on demand and reuses the existing technical and combined backtests. A missing candle can no longer turn a longer interval into a claimed 24-hour return. Results show actual signal/outcome counts, sample periods and run time. They are exploratory historical observations, not a chronological out-of-sample ML evaluation.

Combined analysis may use AI-provisional categories. Its entry uses a prior hourly close (up to the existing two-hour join tolerance), and outcomes are 24 hours from that candle, not precisely from article publication. One event per signal/day reduces but does not eliminate overlap. Costs, slippage, availability of later article edits and regime shifts limit interpretation. Existing same-event comparison logic is retained; different sample populations must not be compared as if they were an experiment.

If `data/historical_news.csv` exists, the advanced view offers an opt-in to include it. Its existing loader excludes SYNTHETIC sources and generates provisional classifier labels; its dates must be genuine first-available timestamps. The dashboard does not independently verify user-supplied dates. The archive file is never rewritten.

For human batch review, keep using the separate interface:

```powershell
.\.venv\Scripts\python.exe -m streamlit run review_research_app.py --server.port 8502 --server.address 127.0.0.1
```

See [RESEARCH_REVIEW_README.md](RESEARCH_REVIEW_README.md) for updating research outputs after review.

## Verification of this upgrade

```powershell
.\.venv\Scripts\python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v
```

The final full-suite run passed **112 tests with zero failures**. Tests cover label protection, collection restrictions, chronology/leakage, research preparation, market freshness/fallbacks, chart time conversion, exact outcome horizons, selected historical windows, probability gating and Streamlit page interactions. Streamlit's UI harness exercises missing data, both appearance choices, all navigation views, a chart with mocked live data and the historical action. The overview, news and research pages also loaded against the actual local research files without exceptions. Test fixtures never enter the real dataset or artifacts.

The live connectivity check on 15 November 2026 succeeded for Binance's ticker, 887 completed hourly candles and both publisher RSS feeds (26 feed items cached separately). These feeds can return old articles; they are not treated as fresh market confirmation. The local Streamlit health endpoint returned HTTP 200. Browser screenshot inspection was unavailable because this environment exposed no browser; visual layout and mobile appearance still need a user browser check.

Before editing, the old `app.py` and `live_data.py` were copied to `data/backups/dashboard_20261115T1608428655517Z/`. The adjacent `protected_manifest.json` records hashes of 55 existing data/research/model artifacts. The preservation check confirmed all 55 were unchanged.

This is for education only, not financial advice. Crypto is high risk.

References: [Streamlit active-session fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment), [Binance public market data](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints).
