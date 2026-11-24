# Live XRP market display

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The existing environment already has `websockets`. For a fresh environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## What caused the delay

The original chart used only completed hourly candles, fetched through a 55-second cache. A completed hourly close can be almost an hour behind the current trade. Its price card was a separately cached 24-hour REST ticker, not a trade stream. Plotly's default bar auto-range and a future forming-candle close could also pad the right edge.

## New data paths

`streaming_market.py` subscribes to public Binance spot `xrpusdt@trade`, `kline_1m`, `kline_5m`, `kline_15m` and `kline_1h` streams using the market-data-only endpoint. It sends no account or trading requests and needs no API credentials. See the [official Binance stream specification](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md).

- Last-traded price uses the exchange trade timestamp, with a separate receipt timestamp.
- Each selectable chart uses native candles from its own timeframe, including the explicitly labeled forming bar. EMA20/50, RSI14 and MACD are calculated in that chart's candle units.
- Model/market-condition inputs use a separate native hourly frame. Forming bars, minute bars and partial-duration hours are excluded. Existing model feature logic, weights, quality thresholds, evaluation results and validation status are unchanged.
- Existing rolling 24-hour ticker statistics keep their own snapshot timestamp; refreshing a trade never refreshes that statistics clock.
- The two-second live fragment updates independently of the surrounding 30-second analysis fragment. Stable Plotly identity preserves chart interaction; turn off Follow latest to explore. Chart time uses Asia/Colombo, RSI is fixed at 0–100, and the following range ends only two seconds after the last actual plotted update. The existing longer hourly archive remains available.

## Failure behavior

The background worker bootstraps bounded REST history and reconnects with exponential backoff, capped at 60 seconds plus jitter. The WebSocket library handles ping/pong and normal disconnects, including the exchange's periodic connection retirement. A server-shutdown event requests reconnection. A quiet connection is recycled after 30 seconds without valid events; socket status becomes stale earlier.

REST `/trades?limit=1` supplies a genuine last-trade timestamp on fallback. Native REST klines provide chart history/backfill. Unhealthy channels are checked at a 20-second cadence and history is repaired periodically. Network operations have bounded timeouts. HTTP 429/418 responses activate a global REST cooldown respecting Retry-After (at least 60 seconds); restrictions are never bypassed. Request duration and cooldown can extend the fallback interval, which is why timestamps and stale indicators remain visible.

Freshness conventions: WebSocket trades ≤15 seconds old; REST trades ≤45 seconds old; local receipt ≤45 seconds old; selected chart update ≤45 seconds old and its interval must cover now or have ended within 45 seconds. A connected socket alone does not establish a fresh price/chart. Failed calls never rewrite saved event/receipt times. A fallback rolling ticker lacks an exact trade time and is explicitly identified as such. Missing candles are reported, never synthesized. Late events do not replace newer bars/trades or reopen a closed candle. Workers stop after five minutes with no dashboard access.

## Beginner observations and research

The market-condition card and its three reasons remain hourly technical heuristics. A separate observation panel lists upward evidence, downward evidence, RSI/volume cautions and conflicts. News is summarized using the existing eligible recent labels, with human/provisional provenance retained in News Intelligence. Content-category counts do not establish positive/negative price sentiment.

The comparison panel reads the existing audited descriptive regime study and checks unfinished ML reserves before showing outcomes. Similarity means the same historical hourly EMA regime only, not a claim that all conditions match. Those are **24-hour observed outcomes**, explicitly separate from the **1-hour experimental model**. Sample sizes, dependence-aware descriptive intervals, source period and limitations remain visible. No new forecasting evaluation is performed.

The one-hour model retains its recorded test accuracy of 44.26%, majority baseline 41.90%, technical baseline 32.69%, and all existing reliability restrictions. Predictive percentages remain withheld because the historical test does not establish current-market reliability. Live data freshness does not validate a model.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe verify_live_market.py
```

The second command performs a bounded real public connection check and prints results; it does not persist market prices or alter training data. Deterministic tests use fixtures for reconnection, outages, rate limits, clocks, model boundaries and Streamlit behavior. Public network availability is separate from fixture-test success.

Original source files and the protected-file SHA256 manifest were backed up under `data/backups/live_stream_20261116T135329/`. Streaming state is in memory; existing research CSVs, annotations, models, reports and model pointers are never written by the stream. News collection retains its existing separate opt-in/polling behavior and isolated live cache.

Verification on 16 November 2026: **227 tests passed**, zero failures/errors/skips. All **139 protected files were unchanged**. The public smoke check received trade and all four native kline streams; REST returned 600 rows per timeframe and the hourly model adapter retained 599 completed rows. The actual saved-model Streamlit check loaded Overview, all four timeframe selections, News Intelligence, Market Conditions Research, Advanced research and the guide with zero exceptions. Baseline comparison and audited historical outcomes were visible; all three predictive percentages displayed “Not validated”. Checks validate behavior and a point-in-time connection, not uninterrupted future connectivity or forecasting performance.

Verification on 16 November 2026: all **227 tests passed** (zero failures, errors or skips). A real public connection check received a trade and 1m/5m/15m/1h stream updates, and REST returned 600 bars per timeframe with no reported connection failure. The model boundary retained 599 completed native hourly bars and excluded the forming hour. All **139 protected data/model/report files** matched their original SHA256 hashes. Logs are under `tests/live_upgrade_full.log` and `tests/live_connection_check.json`; connectivity success is a point-in-time observation, not an uptime guarantee.

This is for education only, not financial advice. Crypto is high risk. No paper simulator, real accounts, orders, direct trading instructions or leverage features were added.
