# XRP Market Intelligence Dashboard

The current dashboard has beginner-friendly market/news views and a separate advanced research area. See [MARKET_INTELLIGENCE_README.md](MARKET_INTELLIGENCE_README.md) for current behavior, validation limits and the launch command. No retraining is required to open it. Existing collection, annotation and model research scripts remain available.

The separate [24-hour trend research pipeline](TREND_RESEARCH_README.md) now trains Logistic Regression and Random Forest on real historical market candles with purged chronological evaluation and calibration. Run `python train_trend_model.py --days 730`. The evaluated model remains experimental because its holdout balanced accuracy did not pass the documented display policy; forecast probabilities remain withheld. Opening the dashboard never starts training.

**Research / educational dashboard only, not financial advice. Crypto is high risk.**

This academic research prototype does not place orders, handle wallets, or predict an exact price.
It classifies news headlines and summarizes observed technical conditions and past 24-hour
outcomes. A signal is an unvalidated rule until independently backtested.

## Folder guide (English + Sinhala)

- `requirements.txt` — Python packages (`install karana libraries`).
- `model.py` — Generates 200 **SYNTHETIC** labeled XRP example headlines and trains TF-IDF + Logistic Regression. Prints accuracy, classification report and confusion matrix. Optional DistilBERT training included.
- `live_data.py` — Binance read-only API requests, one-hour candle features, RSI/EMAs/MACD, public RSS or authorized CryptoPanic headlines, SQLite archive, six-month backtest logic.
- `app.py` — Streamlit market intelligence dashboard, four navigation views, 30-second auto-refresh in the active browser session; charts, news and update timestamps are shown in **Sri Lanka time (SLST, UTC+05:30)**.
- `data/xrp_headlines_200.csv` — The 200 synthetic examples with `date,headline,source,label` (not real news).
- `data/historical_news.csv` — Optional, real timestamped news archive you must provide for a genuine combined 6-month backtest. NOT included by default.
- `artifacts/` — TF-IDF model and metrics generated when training.
- `requirements-bert.txt` — Optional large transformer package dependencies.
- `.env.example` — Optional CryptoPanic token and model backend.
- `Project_Proposal.pdf` — One-page lecturer proposal.

## Step-by-step: Windows 11 + PowerShell

1. Install **Python 3.11 or 3.12** if Python is not already installed. Open this extracted folder in VS Code.
2. In VS Code choose **Terminal > New Terminal**. The terminal should be in the folder that contains `model.py`.
3. Create virtual environment:

   ```powershell
   py -m venv .venv
   ```

4. Install packages (no PowerShell activation required):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

5. The existing classifier artifacts can be used as-is. The market dashboard also works when model files are absent. Do not retrain the production model until real-data readiness and chronological evaluation requirements are met. The bundled news classifier remains a synthetic teaching demonstration.

6. Start dashboard:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
   ```

   Streamlit should open `http://localhost:8501` in your browser. `Ctrl+C` stops it.

7. **Free no-key news:** The dashboard reads the existing real-news research corpus and polls permitted Cointelegraph and The Daily Hodl publisher RSS feeds. Source failures and archived publication dates are shown honestly. It does not bypass Google RSS robots restrictions.
8. **Legacy collectors:** GDELT, optional Google RSS and authorized CryptoPanic support remain in the collection code. The dashboard uses the publisher feeds above. Never commit `.env` or API credentials to GitHub.

## Optional: Upgrade to DistilBERT

Only after the first four files work:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-bert.txt
.\.venv\Scripts\python.exe model.py --train-distilbert
```

Set `MODEL_BACKEND=distilbert` inside `.env`, then restart Streamlit. This downloads
pretrained model weights and requires more RAM/compute. The first classifier is easier
for a beginner. A DistilBERT improvement is **not** guaranteed with 200 synthetic samples.

## Timezone: Sri Lanka (SLST)

The dashboard displays **Sri Lanka Standard Time, Asia/Colombo (UTC+05:30)** for:
- Price chart and RSI horizontal time axis, including Plotly hover times.
- News-event markers and their hover times on the price chart.
- First-seen and published dates in the news-feed table.
- Live ticker retrieval/last-updated text.

The Binance API, SQLite news records, CSV historical-news inputs, and all backtest
joins **remain in UTC** internally. The conversion happens only in `app.py` when
displaying information. This prevents 5-hour-30-minute offsets in historical tests.

## Six-month backtest — IMPORTANT

**Price history is available directly from public Binance APIs**, so the technical-only
baseline can be evaluated immediately. The combined news-price signal cannot be honestly
backtested without **real dated headlines from that period**. The live news archive grows
while the dashboard is open, but is not retrospectively six months long.

To provide genuine historical news, create `data/historical_news.csv` with the header:

```csv
date,headline,source
```

One row per REAL dated news item. Use UTC ISO 8601 timestamps, e.g.
`2026-08-01T12:00:00Z`. DO NOT copy synthetic training headlines into this file.
Dates should be when the article was actually available to readers.

In **Advanced research**, choose a history window and press `Evaluate historical outcomes`. The real corpus uses its first-available timestamps. If `historical_news.csv` exists, an optional checkbox includes that legacy archive. When no matured matched headlines exist, the combined analysis reports that no samples are available.

**Research choices:**
- Price series: XRP/USDT Binance spot, **closed 1-hour candles**.
- 24-hour return computed from the candle close to 24 subsequent hourly closes.
- Outcome: **UP ≥ +1%**, **DOWN ≤ -1%**, **NEUTRAL otherwise**.
- The legacy historical baseline uses RSI14, EMA20 and EMA50. The new current-condition heuristic also uses MACD, volume and eligible recent news; it is not the same rule as this baseline.
- Volume spike = closed hourly volume ≥ 1.8 × average of previous 20 closed candles.
- Strong bullish = Real_Adoption + bullish technical + volume spike.
- Fake pump warning = Hype_Speculation + RSI > 70.
- Bearish = Negative_FUD + bearish technical.
- Otherwise neutral.
- The news backtest joins articles to the *most recent completed candle* within two hours, never to a future candle. The entry-price approximation may be off by as much as an hour.
- Daily grouping limits correlated/overlapping signals, but does not eliminate dependence.
- Rule-win percentages and future UP/DOWN/NEUTRAL percentages are *descriptive historical frequencies*, not calibrated forecast probabilities.
- A displayed label confidence is classifier certainty, **not truth verification**.

## Real-world ML improvements for lecturer discussion

Collect several thousand REAL, manually verified and consistently labeled published
headlines with source/time metadata; separate news authenticity from news-topic categories;
use time-based train/test splits rather than random splits; train baseline and DistilBERT;
compare macro F1, calibration, false-pump precision and meaningful event backtests;
repeat with out-of-sample rolling windows; compare against simple price-only baselines.

## API reference

- Binance read-only endpoint: `https://data-api.binance.vision/api/v3/ticker/24hr?symbol=XRPUSDT`
- Binance 1h candles: `https://data-api.binance.vision/api/v3/klines?symbol=XRPUSDT&interval=1h&limit=100`
- CryptoPanic documentation: `https://cryptopanic.com/developers/api/`
- Streamlit fragments: `https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment`

## NEW: Real news training (upgrade)

For improved provisional classification, separate Silver/Gold datasets and a
standalone Streamlit batch review interface, see
[RESEARCH_REVIEW_README.md](RESEARCH_REVIEW_README.md).
Run `python research_workflow.py`, then
`streamlit run review_research_app.py --server.port 8502 --server.address 127.0.0.1`.

For resumable multi-source collection of 300–500 real headlines, see
[MULTISOURCE_NEWS_README.md](MULTISOURCE_NEWS_README.md). Run
`python collect_multisource_news.py --target 400 --days 84 --max-requests 65`.
The original 53-row dataset and reviews remain separate and preserved.

For automatic provisional classification, batch review reports and protected
chronological training preparation, see [BATCH_LABELING_README.md](BATCH_LABELING_README.md).
Run `python ai_labeling.py batch`, then `python prepare_real_training.py`.

For AI-assisted suggestions and keyboard review, see
[AI_LABELING_README.md](AI_LABELING_README.md). Suggestions stay separate from
human labels and are never used directly as evaluation ground truth.

Read **`REAL_DATA_UPGRADE_README.md`** for the step-by-step beginner guide.
Use `collect_real_news.py`, `label_real_news.py`, `train_real_model.py`,
`evaluate_real_signals.py`. Your synthetic baseline remains unchanged.
Choose the real trained classifier using `NEWS_TFIDF_MODEL=real` in `.env`.
Do not interpret headline accuracy as predictive financial performance.
