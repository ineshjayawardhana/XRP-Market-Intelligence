# PROJECT PROPOSAL

## Title
XRP Live Hype Detector and Future Trend Analyzer: An Explainable News-and-Market Analytics Dashboard

## Problem Statement
Cryptocurrency news feeds mix reported ecosystem developments, sensational price speculation, and negative coverage. Many existing trackers emphasize prices without showing how headline categories align with technical market conditions. This can make it difficult for students and researchers to distinguish information types, investigate market reactions, and assess whether a trend rule has any measurable historical support. The proposed system is an educational analytics prototype, not a trading or investment-advice service.

## Objectives
1. Build a three-class NLP model to identify Real_Adoption, Hype_Speculation and Negative_FUD headlines.
2. Display near-real-time XRP/USDT spot statistics and recent historical prices from public APIs.
3. Calculate RSI(14), EMA(20), EMA(50), MACD and volume-spike indicators.
4. Combine current news categories and indicators into transparent trend-warning signals.
5. Evaluate empirical 24-hour UP/DOWN/NEUTRAL outcome frequencies using responsibly timestamped historical data, and visualize evidence in Streamlit.

## Methodology
Phase 1: Generate a clearly marked 200-row synthetic teaching dataset, train a TF-IDF + Logistic Regression baseline, and evaluate with a held-out split, accuracy, macro-F1, classification report and confusion matrix. Optionally fine-tune DistilBERT; replace synthetic data with verified real headlines before drawing external validity conclusions.

Phase 2: Poll the read-only Binance public endpoint at 30-second dashboard intervals and retrieve hourly price/volume candles for 30-day charts and approximately 180-day technical baselines. Calculate indicators using completed candles to avoid look-ahead leakage.

Phase 3: Retrieve XRP headlines through a permitted CryptoPanic API account or a free public news RSS source every five minutes. Apply the NLP model and combine it with technical conditions. Archive the time first observed, label, model confidence and contemporaneous polled price in SQLite. Backtest event signals against the next 24-hour return (UP >= +1%, DOWN <= -1%, neutral otherwise), showing number of cases and observed frequencies. A complete six-month combined backtest will only be reported after genuine historical headlines are available.

## Expected Outcome
A four-tab Streamlit application showing a live price panel, chart with marked news events and RSI, timestamped classified news feed, headline-testing tool, and backtest summaries. Deliverables include reproducible source code, labeled teaching dataset, baseline evaluation, an optional transformer model, limitations, and a research-oriented evaluation report.

## Ethics and Limitations
Synthetic headlines do not prove real-world accuracy. Technical signals and classifier confidence do not establish the probability of future price movements. APIs can be delayed/unavailable. The dashboard will prominently display: "This is for education only, not financial advice. Crypto is high risk."
