"""Read-only, EVENT-BASED research: do 24-hour XRP outcomes follow held-out news?

Command: python evaluate_real_signals.py

Inputs: real_news_test_predictions.csv (OUT-OF-SAMPLE news predictions ONLY)
        180 days historical, complete Binance XRPUSDT spot hourly bars.
Outputs: reports/real_24h_event_backtest.csv + signal_summary.csv

Important: This is descriptive research, NOT a calibrated probability forecast,
trade signal, futures advice, or a deployable strategy. Dataset size matters.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from live_data import (combined_news_backtest, get_historical_prices, summarize_backtest)

ROOT = Path(__file__).resolve().parent
PRED = ROOT / 'data' / 'real_news_test_predictions.csv'
OUT = ROOT / 'reports'


def summarize_signal_comparison(events: pd.DataFrame) -> pd.DataFrame:
    """Empirical frequencies on the same held-out article days, no future feature values."""
    if events.empty:
        return pd.DataFrame()
    rows = []
    for rule in ['technical', 'signal']:
        for label, group in events.groupby(rule):
            n = len(group)
            rows.append({
                'system': 'technical_only' if rule == 'technical' else 'news_plus_technical',
                'label': label, 'cases': n,
                'up_pct': round(100 * group['outcome'].eq('UP').mean(), 2),
                'down_pct': round(100 * group['outcome'].eq('DOWN').mean(), 2),
                'neutral_pct': round(100 * group['outcome'].eq('NEUTRAL').mean(), 2),
                'small_sample_warning': n < 20,
            })
    return pd.DataFrame(rows)


def evaluate(candles: pd.DataFrame | None = None) -> pd.DataFrame:
    if not PRED.exists():
        raise FileNotFoundError('Train/evaluate real labels first: python train_real_model.py')
    news = pd.read_csv(PRED)
    needed = {'date', 'headline', 'predicted_label', 'held_out'}
    if not needed.issubset(news.columns):
        raise ValueError(f'Expected columns: {needed - set(news.columns)}')
    news = news[news['held_out'].astype(str).str.lower().eq('true')].copy()
    news = news.rename(columns={'date': 'observed_at', 'predicted_label': 'label'})
    news['observed_at'] = pd.to_datetime(news['observed_at'], utc=True, errors='coerce')
    news = news.dropna(subset=['observed_at']).copy()
    if candles is None:
        candles = get_historical_prices(days=180)
    event_results = combined_news_backtest(candles, news, days=180)
    OUT.mkdir(exist_ok=True)
    if event_results.empty:
        print('No valid matched held-out news events. No rates will be invented.')
        return pd.DataFrame()
    # Don't claim proof of next 24-hour trade viability; hourly prices are approximations.
    event_results['time_source'] = 'GDELT_FIRST_SEEN (publication time not verified)'
    event_results.to_csv(OUT / 'real_24h_event_backtest.csv', index=False)
    summary = summarize_backtest(event_results)
    comparisons = summarize_signal_comparison(event_results)
    comparisons.to_csv(OUT / 'real_signal_comparison.csv', index=False)
    summary.to_csv(OUT / 'real_signal_summary.csv', index=False)
    print('\nHeld-out REAL news + Binance market outcomes (descriptive):')
    print(comparisons.to_string(index=False))
    print('\nCAUTIONS: results use hourly price approximation at GDELT first-seen time;')
    print('news timestamps may lag; repeated market regimes; small n; event overlap;')
    print('no transaction costs, execution, slippage or walk-forward training.')
    print('A high news classification accuracy does NOT imply profitable trading.')
    return comparisons


if __name__ == '__main__':
    evaluate()
