# Next 1 Hour — Research Outlook

The main dashboard now uses a separate **1-hour** classifier. The old 24-hour model, its metadata and comparisons remain in Advanced research and in their original files. News Intelligence and Market Conditions Research were not changed.

Run from the project folder:

```powershell
# Explicit training/collection. Completed frozen results are reused without retesting.
.\.venv\Scripts\python.exe train_one_hour.py

# Reuse the separate saved 1-hour data/result without network collection.
.\.venv\Scripts\python.exe train_one_hour.py --offline

# Regenerate the readable comparison and uncertainty report.
.\.venv\Scripts\python.exe one_hour_report.py

# Launch the updated dashboard.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

# Automated verification.
.\.venv\Scripts\python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v
```

Open http://127.0.0.1:8501. The active dashboard still refreshes every 30 seconds. Opening it performs inference only, never training.

## Actual results

The new source contains **29,889 real public Binance XRP/USDT hourly records**, from 1 June 2018 through 31 October 2021 UTC. It was collected in a separate folder without overwriting the existing historical caches. There are 87 absent hour timestamps, eight partial/non-standard bars and no duplicate opens. Missing prices were not filled. Contiguous-segment feature warm-up and exact next-hour outcome requirements reduce the eligible counts below calendar-hour counts.

The reserve dates (January–October 2021) were frozen before collection. This block lies before the earliest original research cache (November 2021). Development uses earlier data only, with three purged chronological validation folds and separate calibration. This is newly collected retrospective historical evidence, not a prospective track record.

| Candidate | Validation accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Logistic Regression |44.60%|43.66%|0.4112|
| Random Forest |45.44%|44.68%|0.4342|
| HistGradientBoosting |45.37%|44.70%|0.4367|
| ExtraTrees |45.01%|44.14%|0.4219|

HistGradientBoosting was selected using validation macro-F1, with Brier score as a tie-breaker. No model was selected using final-test scores. Only this selected model and the fixed baselines were evaluated on the final test.

| Final test, n=6,311 | Accuracy | Balanced accuracy | Macro-F1 | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Selected 1-hour model |44.26%|38.56%|0.3850|0.6214|1.0187|0.0173|
| Training-majority / training-prior class |41.90%|33.33%|0.1968|—|—|—|
| Training-prior probability distribution |41.90%|33.33%|0.1968|0.6667|1.0986|0.0853|
| Existing EMA/RSI technical rule |32.69%|32.77%|0.3168|—|—|—|

Majority class and class-prior probabilities come from classifier-training data, not test class counts. Majority/technical rule outputs are deterministic one-hot vectors, so their very large probability losses are not meaningful calibration comparisons. Compare probability quality against the training-prior distribution.

Per-class test counts are **UP 2,644; DOWN 2,539; NEUTRAL 1,128**. Recalls are 53.44%, 48.05% and **14.18%** respectively. Low Neutral recall remains a material limitation. Full precision/recall/F1, confusion matrix, original-versus-calibrated scores, split dates and fold thresholds are in [the generated report](reports/one_hour/report.md) and its JSON/CSV companions.

The selected model's accuracy exceeded majority by **2.36 percentage points** on this historical test. Approximate paired block intervals were +1.06 to +3.82 points using 24-hour blocks and +0.74 to +4.09 points using 168-hour blocks. Blocks never bridge source gaps. These are pointwise historical estimates under limited dependence assumptions, not financial guarantees. Comparisons to the old 24-hour model's accuracy would be invalid: horizon, threshold and test period differ.

## Target and threshold

The new target is `close(t+1 hour) / close(t) - 1`, using completed hourly candles and exact timestamp lookup within each contiguous source segment. The last unknown outcome in each segment is dropped. No 24-hour target label or old fitted model weight is reused.

The neutral-band rule was fixed before evaluation: take the **one-third quantile of absolute next-hour returns in classifier-training rows only**, then use symmetric boundaries. UP is return at or above the positive boundary; DOWN is at or below the negative boundary; otherwise NEUTRAL. This targets roughly one-third neutral training observations without tuning the rule on validation or test performance. It does not guarantee balanced future classes.

The final frozen threshold is **±0.207263432%**. Each validation fold derives its own threshold from its own earlier training block (approximately 0.21321%, 0.20407%, 0.20523%). Calibration/validation/test labels always use their corresponding training-derived threshold. The final test never chooses or adjusts its threshold. Dashboard boundary text is rounded for readability; JSON retains full precision.

## Leakage protection and calibration

Inputs are the existing 21 causal RSI, EMA, MACD, volume, return, volatility, candle-range/body and UTC-hour features. Historical 24-hour *past return/volatility features* can be valid 1-hour predictors; they are not the discarded 24-hour future target.

Indicators restart after absent/partial bars. At least 168 contiguous warm-up hours are required. No interpolation, future fill or centered window is used. Targets are constructed afresh from exact next closes.

Final training/calibration/test counts are **18,699 / 341 / 6,311**. Final training ends 30 November 2020; the surviving calibration observations are 7–21 December 2020. Final test origins run from 1 January 2021, 03:59:59.999 UTC to 31 October 2021, 22:59:59.999 UTC. Gaps/warm-up explain the reduced calibration count.

Earlier target ends must be strictly before the next partition's first feature timestamp, providing a one-hour purge. Adjacent hourly outcomes share a price endpoint but do not overlap return increments. Their features and volatility regimes remain dependent. Histogram boosting disables random early stopping. StandardScaler fits only training data. Fixed model settings avoid a large hyperparameter search.

Temperature scaling fits only the separate calibration block; model weights stay fixed. Final calibration is excluded from family selection. The saved model is the exact model tested and is not refitted on the final test or all data afterward. Historical outputs carry actual present-day computation times and are marked retrospective replay.

## Why current percentages remain hidden

The model passed the predefined **numerical historical** quality/calibration policy on the older reserve. That does not establish reliability in the current market. Live display additionally requires out-of-sample market evidence within 180 days, matching task/model/data hashes and the exact latest completed hourly candle. The current 2021 test fails the recent-market requirement, so the dashboard shows an **experimental class with a low-reliability warning**, not UP/DOWN/NEUTRAL percentages.

The target window starts at the latest completed hourly close and ends at the next completed hourly close. An intrahour refresh does not create a new rolling 60-minute forecast. Current tick price is displayed as context; only completed hourly data enters this model. Main timestamps and three current technical-condition explanations use SLST; explanations describe technical heuristics, not fabricated ML feature attributions.

## Preservation and verification

New datasets live under `data/trend_1h/`, new models under `artifacts/trend_1h_runs/`, and the new pointer is `artifacts/trend_1h_current.json`. The original news and 24-hour model artifacts, reports, human annotations, SKIP decisions and market-condition study remain unchanged. Repeating the training command reuses the frozen completed result. New model designs need a genuinely new untouched study; this test cannot be optimized repeatedly.

Backups and the 125-file preservation manifest are under `data/backups/one_hour_20261116T0722408670890Z/`. The frozen original training source was preserved before reader/inference-verification repairs. Explicit ISO8601 and round-trip numeric CSV parsing fixed mixed timestamp precision and exact tree-score replay without retraining or altering stored labels/scores.

All **209 automated tests passed**. New checks cover fresh 1-hour labels, training-only thresholds, past-only features, gap/partial filtering, purge boundaries, wrong-horizon rejection, stale inputs, old-evidence probability suppression, mixed ISO timestamps, saved-score replay, tampering and no-rescore behavior. Automated Streamlit tests preserve both appearance choices and earlier views; browser screenshot inspection remains unavailable.

The final SHA256 preservation check confirmed **all 125 protected files unchanged**. Actual local-artifact UI checks loaded the 1-hour main view, Advanced research with the archived 24-hour results, News Intelligence and Market Conditions Research with zero exceptions. A fresh public Binance-data inference produced an experimental class with probabilities correctly withheld; the original 24-hour integrity check still passes and its validation status remains false.

No automatic trading, order execution, BUY/SELL instructions or leverage features were added. This is for education only, not financial advice. Crypto is high risk.
