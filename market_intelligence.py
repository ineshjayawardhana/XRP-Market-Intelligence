"""Testable market heuristics, SLST charts and historical outcome summaries.

No model fitting, trading, or probability claims. UTC stays authoritative.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from live_data import calculate_indicators, technical_baseline_backtest, combined_news_backtest

TZ = 'Asia/Colombo'
LABEL_NAMES = {'Real_Adoption': 'Real adoption', 'Hype_Speculation': 'Hype / speculation',
               'Negative_FUD': 'Negative reporting', 'Neutral_Market_News': 'Neutral market news',
               'REVIEW_REQUIRED': 'Needs review'}
LABEL_COLORS = {'Real_Adoption': '#10b981', 'Hype_Speculation': '#f59e0b',
                'Negative_FUD': '#f43f5e', 'Neutral_Market_News': '#94a3b8'}


def utc(value):
    return pd.to_datetime(value, utc=True, errors='coerce')


def slst(value):
    date = utc(value)
    return date.tz_convert(TZ).strftime('%d %b %Y, %I:%M:%S %p') if pd.notna(date) else 'Not available'


def plot_times(values):
    return pd.to_datetime(values, utc=True, errors='coerce').dt.tz_convert(TZ).dt.tz_localize(None)


def freshness(value, max_age_seconds, now=None):
    current = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    instant = utc(value)
    if pd.isna(instant):
        return dict(fresh=False, age_seconds=None, description='Timestamp unavailable')
    age = (current - instant).total_seconds()
    if age < -60:
        return dict(fresh=False, age_seconds=age, description='Timestamp is in the future')
    return dict(fresh=-60 <= age <= max_age_seconds, age_seconds=max(0, age),
                description=f'{max(0, age) / 60:.1f} minutes old')


def candle_quality(candles, now=None, minimum=60):
    if candles is None or candles.empty:
        return ['Hourly price candles are unavailable.']
    required = {'close_time', 'open_time', 'close', 'open', 'high', 'low', 'volume'}
    if not required.issubset(candles.columns):
        return ['Hourly data is missing required price or volume fields.']
    if len(candles) < minimum:
        return [f'At least {minimum} completed hourly candles are needed for stable indicator calculations.']
    dates = pd.to_datetime(candles['close_time'], utc=True, errors='coerce')
    issues = []
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        issues.append('Hourly candle timestamps are invalid, duplicated or unordered.')
    if dates.diff().dropna().ne(pd.Timedelta(hours=1)).any():
        issues.append('Missing hourly candles interrupt the price history.')
    current = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if dates.max() > current:
        issues.append('A forming or future candle is present.')
    numeric = candles[['close', 'open', 'high', 'low', 'volume']].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy()).all() or (numeric[['close', 'open', 'high', 'low']] <= 0).any().any() or (numeric['volume'] < 0).any():
        issues.append('Hourly data contains invalid price/volume values.')
    return issues


def recent_news(news, now=None, hours=6):
    """A recently collected old article is not fresh news. Require BOTH clocks."""
    if news is None or news.empty:
        return pd.DataFrame()
    current = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    frame = news.copy()
    published = pd.to_datetime(frame['published_at'], utc=True, errors='coerce')
    observed = pd.to_datetime(frame['observed_at'], utc=True, errors='coerce')
    cutoff = current - pd.Timedelta(hours=hours)
    eligible = (published.between(cutoff, current) & observed.between(cutoff, current)
                & frame['label'].isin(LABEL_COLORS))
    if 'outlook_eligible' in frame:
        eligible &= frame['outlook_eligible'].eq(True)
    return frame[eligible].copy()


def market_outlook(ticker, candles, news, errors=(), now=None):
    quote = freshness((ticker or {}).get('retrieved_at'), 90, now)
    exchange = (ticker or {}).get('exchange_close_at')
    quote_ok = quote['fresh'] and (not exchange or freshness(exchange, 120, now)['fresh'])
    quality = candle_quality(candles, now)
    candle_ok = not quality and freshness(candles['close_time'].iloc[-1], 90 * 60, now)['fresh']
    if not quote_ok or not candle_ok or errors:
        reasons = [
            ('The latest quote is missing, stale or could not be refreshed.' if not quote_ok or errors else 'The current quote is available, but price alone cannot establish a trend.'),
            quality[0] if quality else ('The last completed hourly candle is stale.' if not candle_ok else 'Indicator history is available, but the latest refresh has a source problem.'),
            'Wait for complete, current evidence. The separate 1-hour model remains experimental.']
        return dict(condition='Mixed / Insufficient Evidence', tone='caution', reasons=reasons, indicators=None, recent_news_count=0)
    frame = calculate_indicators(candles)
    last = frame.iloc[-1]
    values = ['rsi', 'ema20', 'ema50', 'macd', 'macd_signal', 'close']
    if not all(math.isfinite(float(last[key])) for key in values):
        return dict(condition='Mixed / Insufficient Evidence', tone='caution',
                    reasons=['Indicators have not finished warming up.', 'More complete hourly data is required.',
                             'The separate 1-hour model remains experimental.'], indicators=None, recent_news_count=0)
    close, ema20, ema50 = [float(last[key]) for key in ('close', 'ema20', 'ema50')]
    gap = (ema20 - ema50) / close
    ema_vote = 1 if gap > .001 and close > ema20 else -1 if gap < -.001 and close < ema20 else 0
    histogram = float(last['macd'] - last['macd_signal'])
    macd_vote = 1 if histogram > close * .00001 else -1 if histogram < -close * .00001 else 0
    rsi = float(last['rsi'])
    rsi_vote = 1 if 55 < rsi < 70 else -1 if 30 < rsi < 45 else 0
    extreme = rsi >= 70 or rsi <= 30
    volume_spike = bool(last['volume_spike'])
    price_step = close - float(frame.iloc[-2]['close'])
    fresh = recent_news(news, now)
    counts = fresh['label'].value_counts().to_dict() if not fresh.empty else {}
    sources = fresh['source'].nunique() if not fresh.empty else 0
    adoption, negative, hype = [counts.get(label, 0) for label in ('Real_Adoption', 'Negative_FUD', 'Hype_Speculation')]
    news_vote = (1 if adoption > negative else -1 if negative > adoption else 0) if len(fresh) >= 3 and sources >= 2 else 0
    direction = 1 if ema_vote == macd_vote == 1 and rsi_vote >= 0 else -1 if ema_vote == macd_vote == -1 and rsi_vote <= 0 else 0
    conflict = (direction and news_vote == -direction) or (direction and volume_spike and price_step * direction < 0)
    hype_caution = hype >= 3 and hype > adoption + negative
    if direction and not extreme and not conflict and not hype_caution:
        condition, tone = ('Bullish', 'positive') if direction == 1 else ('Bearish', 'negative')
    elif not any((ema_vote, macd_vote, rsi_vote)) and not extreme and not news_vote:
        condition, tone = 'Neutral', 'neutral'
    else:
        condition, tone = 'Mixed / Insufficient Evidence', 'caution'
    trend = ('above' if ema20 > ema50 else 'below' if ema20 < ema50 else 'equal to')
    momentum = 'improving' if macd_vote > 0 else 'weakening' if macd_vote < 0 else 'near balance'
    rsi_text = 'stretched upward' if rsi >= 70 else 'stretched downward' if rsi <= 30 else 'in its middle range'
    volume_text = 'Trading volume is unusually high for the last completed hour' if volume_spike else 'Trading volume has no unusual spike in the last completed hour'
    news_text = (f'{len(fresh)} recent classified headlines include {adoption} adoption, {hype} hype and {negative} negative reports; these labels are content context, not verified market forecasts.'
                 if len(fresh) else 'No sufficiently recent classified news is available to confirm the technical picture.')
    return dict(condition=condition, tone=tone,
                reasons=[f'Trend: the faster 20-hour average is {trend} the slower 50-hour average. Price is {"above" if close > ema20 else "below"} the faster average.',
                         f'Momentum: MACD is {momentum}; RSI is {rsi:.1f}, {rsi_text}. Extreme RSI does not guarantee a reversal.',
                         f'Participation and news: {volume_text}. {news_text}'],
                indicators={key: float(last[key]) for key in values} | {'volume_spike': volume_spike, 'macd_histogram': histogram},
                recent_news_count=len(fresh))


def intelligence_chart(candles, news, days=7, theme='Dark', interval='1h', now=None):
    data = calculate_indicators(candles)
    data = data[data.close_time >= data.close_time.max() - pd.Timedelta(days=days)]
    dark = theme == 'Dark'
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.55, .2, .25],
                        vertical_spacing=.06, subplot_titles=['Price and moving averages', 'RSI · momentum range', 'MACD · momentum change'])
    # A forming chart candle ends in the future; plot it at its last actual update.
    # These display-only bars never enter model inference.
    current = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    display_times = data.close_time.clip(upper=current)
    if 'updated_at' in data:
        updates = pd.to_datetime(data.updated_at, utc=True).clip(upper=current)
        display_times = display_times.where(data.close_time <= updates, updates)
    times = plot_times(display_times)
    fig.add_trace(go.Scatter(x=times, y=data.close, name='XRP/USDT', line=dict(color='#38bdf8' if dark else '#0369a1', width=2.2)), row=1, col=1)
    for column, color, label, dash in [('ema20', '#2dd4bf', 'EMA20 · faster trend', 'solid'), ('ema50', '#a78bfa', 'EMA50 · slower trend', 'dash')]:
        fig.add_trace(go.Scatter(x=times, y=data[column], name=label, line=dict(color=color, width=1.4, dash=dash)), row=1, col=1)
    if news is not None and not news.empty:
        events = news.copy()
        events['observed_at'] = pd.to_datetime(events['observed_at'], utc=True, errors='coerce')
        closed = data[data.close_time <= current]
        events = events[events.observed_at.between(data.close_time.min(), min(data.close_time.max(), current))]
        if not events.empty:
            aligned = pd.merge_asof(events.sort_values('observed_at'), closed[['close_time', 'close']].sort_values('close_time'),
                                    left_on='observed_at', right_on='close_time', direction='backward', tolerance=pd.Timedelta(hours=1))
            symbols = {'Real_Adoption': 'diamond', 'Hype_Speculation': 'triangle-up', 'Negative_FUD': 'x', 'Neutral_Market_News': 'circle'}
            for label, group in aligned.dropna(subset=['close']).groupby('label'):
                if label not in LABEL_COLORS:
                    continue
                fig.add_trace(go.Scatter(x=plot_times(group.observed_at), y=group.close, mode='markers',
                    name=LABEL_NAMES[label], text=group.headline,
                    marker=dict(size=9, color=LABEL_COLORS[label], symbol=symbols[label]),
                    hovertemplate='%{x|%d %b %I:%M %p} SLST<br>%{text}<br>Prior chart close: %{y:.4f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=data.rsi, name='RSI14', line=dict(color='#fbbf24', width=1.6)), row=2, col=1)
    fig.add_hline(y=70, line_dash='dot', line_color='#f43f5e', row=2, col=1)
    fig.add_hline(y=30, line_dash='dot', line_color='#10b981', row=2, col=1)
    hist = data.macd - data.macd_signal
    seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}[interval]
    fig.add_trace(go.Bar(x=times, y=hist, width=seconds * 700, name='MACD histogram · above / below zero', marker_color=np.where(hist >= 0, '#10b981', '#f43f5e')), row=3, col=1)
    fig.add_trace(go.Scatter(x=times, y=data.macd, name='MACD', line=dict(color='#38bdf8', width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=times, y=data.macd_signal, name='MACD signal', line=dict(color='#c4b5fd', width=1.3, dash='dash')), row=3, col=1)
    fig.update_yaxes(title_text='USDT', row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_xaxes(tickformat='%d %b %H:%M', hoverformat='%d %b %Y, %I:%M %p SLST')
    fig.update_xaxes(title_text='Sri Lanka time · SLST (UTC+05:30)', row=3, col=1)
    # Plotly's bar-width auto-range and future candle close used to pad the axis.
    if len(times):
        fig.update_xaxes(range=[times.iloc[0], times.iloc[-1] + pd.Timedelta(seconds=2)], autorange=False)
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white', height=740, hovermode='x unified',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=15, r=15, t=65, b=15), legend=dict(orientation='h', y=1.18),
                      font=dict(family='Arial', color='#dbe7f5' if dark else '#24364b'))
    return fig


def outcome_counts(results):
    if results.empty:
        return pd.DataFrame(columns=['Signal', 'Samples', 'UP', 'DOWN', 'NEUTRAL'])
    return pd.DataFrame([dict(Signal=signal, Samples=len(group), **{label: int(group.outcome.eq(label).sum()) for label in ['UP', 'DOWN', 'NEUTRAL']})
                         for signal, group in results.groupby('signal')])


def evaluate_history(candles, news, days=180):
    quality = candle_quality(candles)
    if quality:
        raise ValueError('Historical evaluation blocked: ' + ' '.join(quality))
    baseline = technical_baseline_backtest(candles, days=days)
    valid = news[news.label.isin(LABEL_COLORS)].copy() if news is not None and not news.empty else pd.DataFrame()
    combined = combined_news_backtest(candles, valid, days=days)
    return dict(baseline=baseline, combined=combined, evaluated_at=pd.Timestamp.now(tz='UTC').isoformat(),
                period_start=candles.close_time.min().isoformat(), period_end=candles.close_time.max().isoformat(), requested_days=days)
