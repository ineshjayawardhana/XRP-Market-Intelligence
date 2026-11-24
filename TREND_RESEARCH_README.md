# Educational XRP 24-hour trend research

The dashboard now has a complete, separate market-direction training and inference pipeline. It does not train the news classifier or place trades. The actual model currently **fails the documented quality gate**, so the dashboard shows its current class as **experimental**, without reliable UP/DOWN/NEUTRAL probabilities.

## Commands

Run from the project folder in PowerShell. Existing installed libraries are reused; SciPy is now declared explicitly because the calibrator uses its optimizer.

```powershell
# Collect/resume public market history, then train and evaluate both candidates.
.\.venv\Scripts\python.exe train_trend_model.py --days 730

# Reproduce research from the saved candles without downloading again.
.\.venv\Scripts\python.exe train_trend_model.py --offline --days 730

# Start the existing dashboard with the new model integration.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

# Run all automated tests.
.\.venv\Scripts\python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v
```

Open http://127.0.0.1:8501. Opening the dashboard performs inference only; training requires the explicit training command. Each run creates a separate version directory. Avoid repeatedly tuning against the final holdout; later model changes need a new untouched period or a preregistered rolling evaluation.

## Actual market dataset and results

The first collection used **18 public Binance requests**, with no failed connections. It saved **17,688 completed hourly XRP/USDT spot candles**, from 8 November 2024 through 15 November 2026 UTC. This includes seven extra warm-up days. The collector checkpoints every page, resumes missing intervals, uses normal public access, and stops with a persisted cooldown on HTTP 429/418. It never generates replacement candles or interpolates gaps. If collection is interrupted, rerun the same command. A complete cached interval is not downloaded again.

After feature warm-up and excluding the unknown last 24-hour outcomes, the run has **17,497 usable hourly rows** and **700 supervised origins spaced 25 hours apart**. Spacing rotates through UTC hours, rather than testing only one hour of the day; no two supervised target windows overlap. Price-feature lookbacks can overlap and market regimes remain dependent, so this does not establish independent financial observations.

Final fit/calibration/holdout sizes are **480 / 45 / 175**. The held-out prediction origins run from **16 May 2026, 12:59:59.999 UTC to 13 November 2026, 18:59:59.999 UTC**. All 24-hour outcomes had matured before evaluation. Holdout outcome counts are **UP 45, DOWN 63, NEUTRAL 67**.

The current evaluated model version is `xrp24h_20261115T170231910490Z`, evaluated 15 November 2026 at 17:02:34 UTC (22:32:34 SLST). The saved JSON report is the authoritative source for full precision and later versions.

| Model/baseline | Accuracy | Balanced accuracy | Macro-F1 | Brier score | Log loss | Top-class ECE |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression, selected on development folds |0.3657|0.3400|0.3270|0.6674|1.1028|0.0275|
| Random Forest |0.3257|0.3138|0.2992|0.6658|1.0974|0.0120|
| Training-majority class |0.3600|0.3333|0.1765|1.2800|23.0679|0.6400|
| Training class-prior distribution |0.3600|0.3333|0.1765|0.6718|1.1070|0.0233|
| Existing technical-only rule |0.3257|0.3186|0.3171|1.3486|24.3037|0.6743|

Majority means the most common **training** class (DOWN), not the largest holdout class chosen with hindsight. Majority/rule outputs are deterministic one-hot vectors for scoring and are not calibrated forecasts; their extreme log loss/Brier values should not be interpreted as probability-model comparisons. The training-prior distribution is the meaningful probabilistic baseline. Multiclass Brier is the summed squared error across three outcomes, averaged over examples (range0–2).

Development walk-forward macro-F1 was0.3238 for Logistic Regression and0.2826 for Random Forest. The model family was frozen using development macro-F1, breaking ties with Brier score, **before** scoring the final holdout. Random Forest's lower holdout probability loss did not cause a post-hoc model switch.

Selected-model confusion matrix (rows actual, columns predicted):

| Actual / predicted |UP|DOWN|NEUTRAL|
|---|---:|---:|---:|
|UP|5|24|16|
|DOWN|5|30|28|
|NEUTRAL|6|32|29|

The model's UP recall is only0.1111. Good-looking aggregate calibration cannot repair poor class discrimination. Balanced accuracy0.3400 is below the predeclared0.38 minimum; **probability display remains blocked**. No accuracy value is a future price probability, and the small difference from majority accuracy does not demonstrate profitable or statistically significant skill.

## Features and targets

`trend_pipeline.feature_frame` reuses the existing RSI14, EMA20, EMA50 and MACD calculations. Model inputs include RSI; relative price/EMA distances and EMA gap; normalized MACD/signal/histogram;1-hour and24-hour volume changes; volume relative to the previous20-hour mean;1/6/12/24/72-hour historical returns;24/72-hour rolling volatility; candle range/body; and cyclic UTC-hour encoding. Absolute EMA/MACD values remain in the hourly dataset for research, while normalized versions enter the model.

All rolling calculations use the current completed candle and earlier candles. There is no centered rolling window, future fill, future news or target-derived input. Targets are looked up by exact timestamp,24 hours after the feature close:

- **UP:** actual close-to-close return >=+1%.
- **DOWN:** actual return <=-1%.
- **NEUTRAL:** otherwise.

Forming/future candles, invalid OHLC ranges, incorrect one-hour durations, duplicated/unordered timestamps and gaps are rejected. Unknown future outcomes are excluded. Missing feature evidence is dropped, never filled from the future.

## Chronological protocol

The latest25% of non-overlapping origins form the untouched holdout. Earlier development data supplies three expanding walk-forward tests of30 origins each for this run. Every fit uses an earlier training block and a later45-origin calibration block. At both boundaries, the earlier block's last **target end** must be strictly earlier than the next block's first feature timestamp. Training-origin separation is thus greater than the24-hour horizon. No random data split is used.

The Logistic Regression pipeline fits StandardScaler only on each training block, then Logistic Regression with fixed settings. Random Forest uses200 trees, depth6 and minimum leaf size12 with a fixed seed. Neither model family has a holdout hyperparameter search. A single positive temperature is optimized for log loss on the calibration block only, with classifier weights held fixed. Temperature scaling uses log probabilities and a softmax; it does not use holdout labels.

The exact final trained/calibrated model that produced the holdout scores is saved for inference. It is **not refitted on the holdout or all data**. Its last classifier-training origin is29 March2026; calibration ends15 May2026. This frozen-model age and later regime drift are limitations, even when current input candles are fresh.

Historical outputs are explicitly marked `retrospective_replay`. `available_at` is the simulated information cutoff, and `prediction_computed_at` is the actual present-day computation time. The pipeline never pretends to have recorded these predictions live in the past. Historical exchange-data vintages/revisions and real execution latency are not known.

## Probability-display policy

`forecast_validation.validated_research` verifies artifact SHA256 hashes, causal feature-code signature, recorded split/maturity periods, non-overlapping origins, observed outcomes against the frozen market prices, training-derived priors, and metrics recomputed from the **exact saved model's** holdout predictions. Model/software versions must match. This is reproducible trusted-local pipeline evidence; it is not an independent refit/audit of every optimization step or a prospective track record.

The policy requires all of these:

- At least100 non-overlapping holdout examples and at least10 examples per outcome.
- Balanced accuracy at least0.38.
- Accuracy at least the training-majority baseline and macro-F1 above it.
- Macro-F1 and balanced accuracy at least the existing technical rule on identical origins.
- Brier score and log loss better than the training-prior probabilistic baseline.
- Ten-bin top-class expected calibration error no greater than0.10.
- Development walk-forward macro-F1 above its training-majority baseline.

These are documented educational thresholds, not calibrated confidence values or a guarantee of future reliability. With only45 calibration and175 holdout observations, calibration estimates have substantial uncertainty, especially classwise. Passing the policy would allow research probabilities with limitations, not investment advice.

Live inference also requires valid completed hourly input no older than90 minutes, a forecast generated within30 minutes and a feature timestamp matching the latest completed candle. A failed gate yields an **experimental class only** with no probability percentages. Stale/corrupt/missing inputs or modified artifacts withhold the current signal as well. Every time remains UTC internally and appears in Sri Lankan time in the dashboard.

## News audit and future comparisons

The existing8 Gold and215 Silver labels remain untouched. Only **one Gold record** has sufficient explicit label/content availability timing, on15 November2026;222 records lack explicit label-availability evidence. Excerpts retrieved today cannot become historical features for September/October predictions, and publication or GDELT first-seen time cannot replace the time a label was actually known.

`trend_news.audit_news` checks identity, protected human decisions, SKIP/rejection, duplicates and separate acquired-content/annotation times. Features become available only after both content and label availability. Gold and Silver counts remain separate. Price returns, not news labels, are the direction targets.

News-enhanced comparisons require at least60 eligible feature days spanning90 days, at least5% prediction coverage and at least10 covered origins in each chronological third. These policy minima currently fail, so **no news-enhanced performance is claimed**. When supported, separate LR/RF+News candidates are evaluated on the same chronological protocol. News inputs are frozen and hashed with the run and checked against point-in-time counts; live new news is still included only after its actual availability. More consistently timestamped contemporaneous news collection and human review are required.

## Files and preservation

- `data/trend/xrp_hourly.csv`: incremental raw market cache; collector state and market-cache backups sit beside it.
- `artifacts/trend_runs/<version>/model.joblib`: selected trend classifier/calibrator, separate from the existing news classifier.
- Each run also contains frozen market data, causal hourly dataset, non-overlapping supervised origins, development/holdout prediction CSVs, report and validation manifest. Older runs are retained.
- `artifacts/trend_24h_report.json` and `trend_24h_validation.json`: pointers/current report used by the dashboard; versioned copies remain in the run directory.
- Dashboard inference writes no dataset labels and never triggers training. The original market/news views, charts, research tools and educational notice remain available.

Before changes, source backups and a 59-file data/model preservation manifest were created under `data/backups/trend_20261115T1653189337889Z/`. All 59 protected files matched their original hashes after implementation. The old news models and all human/SKIP/review datasets remain unchanged.

The full automated suite passed **144 tests with zero failures**, including 32 new trend/news/UI checks. It covers causal features, exact targets, purging, holdout-independent selection, calibration/metric correctness, artifact replay/tampering, incremental collection/cooldown and probability display. The actual local artifacts loaded successfully across all four dashboard pages in both themes; the experimental class was visible and probabilities stayed hidden. Browser screenshot inspection was unavailable in this environment. Tests use temporary synthetic fixtures only; their results are never included in the real market model report.

Documentation: [Binance public market data](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints), [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html).

This is for education only, not financial advice. Crypto is high risk.
