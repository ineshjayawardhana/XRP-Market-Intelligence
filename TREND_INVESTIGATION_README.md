# XRP model-performance investigation

This is a separate educational research workflow. It does not edit `app.py`, the deployed feature functions, production validation/report pointers, existing news or trend models, datasets, human annotations or SKIP decisions.

The completed comparison is [reports/trend_investigation/bb1462a533962b5f/comparison.md](reports/trend_investigation/bb1462a533962b5f/comparison.md). Full metrics, split boundaries, tuning trials, confusion matrices, class reports, calibration bins, uncertainty intervals and source-quality details are in the adjacent JSON and CSV files. The candidate model is isolated in `artifacts/trend_experiments/bb1462a533962b5f/`.

## Main findings

- The old Logistic Regression train/test macro-F1 was 0.4121/0.3270. The original Random Forest was reconstructed from its frozen settings and exactly matched its recorded test scores; its train/test macro-F1 was 0.7334/0.2992. This large gap is consistent with overfitting or changing regimes. It does not identify one cause with certainty.
- The Neutral share increased from 29.0% in old training to 38.3% in its test. UP decreased from 32.7% to 25.7%. Test volatility and class prevalence changed. Four training feature pairs had absolute correlation at least 0.95; marginal feature/return associations were weak. These are descriptive findings, not proof that nonlinear information is absent.
- The old model's 0.57 percentage-point accuracy gain over the training-majority baseline had an exploratory paired block interval from −7.43 to +9.14 points. Its probability calibration improved scoring without changing class rankings or repairing low UP recall.
- Expanded Logistic Regression won the new nested development comparison, with 43.33% accuracy, 43.44% balanced accuracy and 0.4320 macro-F1. The corresponding base-feature procedure had 37.78%, 37.87% and 0.3710. Both used identical origins and separate training/calibration partitions.
- On the newly collected historical reserve, the selected expanded model achieved **36.36% accuracy, 35.40% balanced accuracy and 0.3470 macro-F1**, across 176 observations. Its Brier score was **0.6775**, worse than the training-prior baseline's **0.6707**; its log loss was also worse. Top-class ECE was 0.0557, which does not establish useful discrimination.
- The model beat majority macro-F1 in all three outer folds and technical-rule macro-F1 in two. However, the reserve's balanced-accuracy gain over majority and macro-F1 gain over technical rules had block intervals crossing zero. **Robust, consistent improvement was not established.** The quality gate failed, so no replacement or validated probability release occurred.

New reserve baseline accuracy was 29.55% for the training-majority class and 31.82% for technical rules. These are same-period comparisons. The old 2026 test and new 2024 reserve are different periods: their accuracy values must not be interpreted as a direct before/after improvement.

## Why another historical reserve was needed

The original 175-outcome final test was already examined. New feature/model choices could not honestly reuse it as untouched evidence. Before collecting additional data, the experiment fixed:

- Development starting January 2022.
- A new reserve from May through October 2024, entirely before the old production raw dataset begins in November 2024.
- A fixed global 25-hour origin grid and exact 24-hour target horizon.
- Three 60-origin outer folds, two 45-origin inner folds and 45-origin calibration blocks.
- Two feature recipes, four model families, 18 total parameter/recipe configurations and a macro-F1/Brier selection rule.

No later candles enter development or reserved-test feature construction. The older block is a **newly collected historical reserve**, not prospective evidence of 2026 deployment performance. Its raw prices were downloaded before scoring; procedural and tested code boundaries kept derived reserve labels out of tuning and selection. Human research hypotheses were motivated by the already examined later-period model, which remains a historical-design limitation.

The final model's training/calibration/test sizes were **763 / 45 / 176**. Final training ended 13 March 2024, calibration ended 28 April, and test origins ran from 1 May to 30 October. Exact timestamp/target-maturity boundaries are in the report. Origin windows never overlap; broader market regimes can still be dependent.

## Models, features and selection

The base recipe retains the production 21 features. The expanded recipe adds 27 causal features: lagged returns; 3/48/168-hour momentum; RSI changes; EMA-gap changes; MACD-histogram changes; 24/168-hour relative volume; 6/168-hour volatility and ratios; prior rolling high/low distances; and scaled momentum. High/low and volume reference windows exclude the current candle. There is no centered window, future fill or target-derived feature.

Models are Logistic Regression, Random Forest, HistGradientBoosting and ExtraTrees. Scaling fits only training rows. HistGradientBoosting has `early_stopping=False`; no random internal validation split is used. Hyperparameters are selected on inner purged chronological folds. Outer calibration blocks are excluded from inner tuning. The final calibration block is excluded from all family, feature and parameter selection.

The family/feature procedure is selected from outer development scores. Final parameters are then tuned on earlier development only, and the trained model plus selection metadata are saved before reserve targets are constructed. Only that selected model and predeclared baselines are scored on the reserve; the other algorithms are compared through development folds. The selected regularization was Logistic Regression `C=0.1` with expanded features.

The previously saved production weights were trained on dates after this new reserve. Applying those weights to 2024 would leak future training information, so LR/RF comparisons use fresh earlier-data fits instead.

## Source gaps and protocol amendments

The isolated cache contains **43,967 real Binance hourly records** from November 2021 through November 2026. Only the earlier protocol periods enter the experiments.

The public source omitted the 24 March 2023, 13:00 UTC candle. An adjacent 12:00 candle ended early at 12:39:48.301 UTC. The initial strict collector and validator stopped safely on these issues, before model fitting or reserve-label access.

The old unopened protocol files were preserved with explicit amendment reasons. The amended protocol keeps the same features, models, hyperparameter grid, prediction anchor, objectives and reserve dates, while specifying gap-safe preprocessing. The raw partial candle remains preserved but is excluded from usable hourly observations. All indicators reset at segment boundaries; 168 hours of new warm-up and a fully known 24-hour target are required within each segment. No price, volume or candle was invented or interpolated.

Gap and exclusion details are saved in the report. Both feature recipes use the same surviving observations. Opened or evaluated test reservations cannot be superseded.

## Commands

From the project folder in PowerShell:

```powershell
# Reuse the completed immutable comparison and regenerate readable report files.
# With the same frozen method this returns the saved result, without rescoring.
.\.venv\Scripts\python.exe investigate_trend.py --offline

# Run development-only research; never construct new reserve targets.
.\.venv\Scripts\python.exe investigate_trend.py --offline --development-only

# Launch the unchanged production dashboard.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

# Run automated tests.
.\.venv\Scripts\python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v
```

For a new workspace without the isolated cache, omit `--offline`. Collection uses ordinary public Binance requests, incremental checkpoints and persisted HTTP429/418 cooldowns. It does not bypass access controls or invent missing candles.

Frozen plans and reservation status live under `data/trend_experiments/`. Completed matching runs are reused. Changing the method cannot silently reuse a consumed reserve. Interrupted runs whose reserve has already been opened fail closed rather than refitting against it. Future model redesigns need a new untouched evaluation period; use development-only work while collecting current prospective observations.

## Validation and next research needs

The full automated suite passed 182 tests. It covers production workflows, causality, feature lags, contiguous-segment handling, maturity purging, nested tuning, regime thresholds, protocol immutability, reserve reuse protection, preservation, gap-aware collection and rate-limit behavior. The production preservation manifest checked 93 existing source/data/model files with zero changes. The dashboard continues using its original experimental signal and probability gate.

Further work should emphasize new contemporaneous data across regimes, enough non-overlapping outcomes, simpler preregistered hypotheses and fresh calibration. Broader parameter searches alone can overfit a small development sample. News enhancement still needs credible point-in-time label/context availability. Neither the historical reserve nor small post-hoc monthly groups establish current-market accuracy or profitability.

Official references: [scikit-learn nested model selection](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [HistGradientBoosting validation settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

This is for education only, not financial advice. Crypto is high risk.
