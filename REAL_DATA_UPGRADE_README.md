# Real XRP headline classifier + 24-hour research evaluation

**Educational, read-only research. Not financial advice; crypto markets are high risk.**
**Existing synthetic model is preserved. The new model does not place spot or futures orders.**

## What is genuinely real?

* `collect_real_news.py` downloads *real publisher headline metadata* from GDELT DOC 2.0. It does NOT create synthetic articles. GDELT's `seendate` is *first indexed/seen time*; do not interpret this as verified original publication time.
* `label_real_news.py` asks YOU to review each headline, optionally open the actual publisher link, and choose one of three categories. SKIP any ambiguous/irrelevant entry. This makes human-labeled training data; no keyword auto-labeling.
* `train_real_model.py` trains TF-IDF + Logistic Regression on human-labeled headlines. A **chronological 70/15/15 split** with a 24-hour boundary gap reduces (but does not eliminate) duplicate-event leakage. Test data must not be used to tune the model.
* `evaluate_real_signals.py` matches **held-out** news classifications to completed hourly Binance spot price candles. Compares observed next-24h UP/DOWN/NEUTRAL rates with a technical-only baseline on the same days. It does **not** generate validated probabilities or orders.

## Start in your existing project folder, PowerShell

If your VS Code terminal already displays `(.venv)`:

```powershell
python -m pip install -r requirements.txt
python collect_real_news.py --days 60 --window-days 3
python label_real_news.py --count 50
```

Repeat the `label_real_news.py` command to label more items. Since the dataset is inherently imbalanced, it might take many articles to label at least 15 genuine examples *per category*, plus enough examples per category in each chronological partition.

Check results inside `data/xrp_real_labeled.csv`. Annotations save after every choice. `SKIP` labels are excluded. Run only once you have enough examples:

```powershell
python train_real_model.py
```

This writes:

* `artifacts/news_tfidf_real.joblib` — **separate** trained model.
* `artifacts/metrics_tfidf_real.json` — genuine headline classification accuracy, balanced accuracy and macro-F1 on chronologically held-out articles.
* `artifacts/confusion_matrix_tfidf_real.png`
* `data/real_news_test_predictions.csv` — predictions for **held-out** article rows only.

For a **descriptive** 24-hour market-event evaluation:

```powershell
python evaluate_real_signals.py
```

This writes `reports/real_24h_event_backtest.csv`, `reports/real_signal_comparison.csv`, and `reports/real_signal_summary.csv`, *if valid matches exist*. Small samples will be marked with warnings. Don't infer signal profitability from this evaluation.

## Use the real model in your EXISTING dashboard

Create a file named `.env` in the root project folder, or add the following line to your existing `.env`:

```env
MODEL_BACKEND=tfidf
NEWS_TFIDF_MODEL=real
```

Restart Streamlit:

```powershell
python -m streamlit run app.py
```

The dashboard will display a label saying the active news classifier uses **REAL human-labeled headlines**. The News Feed and Try Your Headline tabs will now use this model. Your original `NEWS_TFIDF_MODEL=synthetic` remains available. Do **not** switch the backend to DistilBERT unless you deliberately train a real-data DistilBERT model; the older DistilBERT is trained on the synthetic examples.

## Methodological limits to explain to your lecturer

1. Training on real headlines does **not** verify article truth. Label is content type, not credibility; manually check claims and publisher links independently.
2. GDELT DOC 2.0 ArticleList normally covers **about 3 recent months**, not 6 months. For an honest 6-month combined backtest, preserve your own daily article archive for six months or obtain a licensed longer archive.
3. GDELT first-seen times may be later than publication. The event study uses an **hourly price approximation** rather than minute-level event time and is vulnerable to news-release timing errors.
4. Be careful about repeated syndicated headlines, duplicate events, overlap between 24-hour outcome windows, regime changes, and outcome class imbalance.
5. Test accuracy for news classification is not price-direction accuracy. The 24-hour directional evaluation should also report confusion matrices, class baselines, confidence intervals, sample sizes, and at least several non-overlapping walk-forward test periods before any forecasting claims.
6. For price-derived features, use ONLY closed candles available at the observation timestamp. Do not leak future article text, indicators, or prices into past evaluation.
7. NO trade automation, position sizing, leverage rules, cost estimates, execution, or real-account connections are provided.

## Data sources & documentation

* GDELT DOC 2.0 documentation: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
* Binance public read-only market data: https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md
* Binance historical public downloads: https://github.com/binance/binance-public-data
