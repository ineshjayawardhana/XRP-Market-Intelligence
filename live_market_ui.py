"""Fast live fragment and beginner observation panel; no trading actions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from market_intelligence import freshness, intelligence_chart, recent_news, slst
from streaming_market import MarketStream

STRETCH = {'width': 'stretch'} if tuple(int(p) for p in st.__version__.split('.')[:2]) >= (1, 50) else {'use_container_width': True}


@st.cache_resource
def stream_resource(root):
    # Shared per application/root, never a socket per rerun or per user.
    return MarketStream()


def stream_snapshot(root, interval='1m'):
    return stream_resource(str(root)).start().snapshot(interval)


@st.fragment(run_every=2)
def render_live_market(root, ticker, hourly, news, theme):
    st.subheader('Live price and chart')
    controls = st.columns([2, 2, 1])
    interval = controls[0].selectbox('Chart timeframe', ['1m', '5m', '15m', '1h'], key='live_timeframe')
    bars = controls[1].select_slider('Visible candles', [60, 180, 360], value=180, key='live_bars')
    follow = controls[2].toggle('Follow latest', value=True, key='live_follow')
    snapshot = stream_snapshot(root, interval)
    trade = snapshot['trade']
    if trade:
        price, price_at = trade['price'], trade['received_at']
        price_source = trade['source']
        status = ('LIVE STREAM' if snapshot['connected'] and snapshot['trade_fresh'] and price_source.startswith('WebSocket')
                  else 'REST FALLBACK' if snapshot['trade_fresh'] and price_source.startswith('REST') else 'STALE LAST TRADE')
        st.metric('Last-traded XRP/USDT', f'${price:,.5f}')
        st.caption(f'{status} · {price_source} · Last trade: {slst(trade["trade_at"])} SLST · Received: {slst(price_at)} SLST')
        if not snapshot['trade_fresh']:
            st.warning('The last trade is stale. This price is retained for reference and is not live.')
    elif ticker:
        st.metric('Last XRP/USDT snapshot', f'${ticker["price"]:,.5f}')
        st.caption('REST rolling-ticker snapshot · Last trade time unavailable · Received: ' + slst(ticker.get('retrieved_at')) + ' SLST')
        if not freshness(ticker.get('retrieved_at'), 45)['fresh']:
            st.warning('The REST snapshot is stale. It is not a live price.')
    else:
        st.info('Waiting for public XRP market data. No current price is available.')
    if not snapshot['connected']:
        st.warning('WebSocket disconnected or awaiting fresh events. Public REST fallback is checked in the background; stale values retain their original times.')
    st.caption('Socket: ' + ('receiving fresh events' if snapshot['connected'] else 'disconnected / waiting') +
               f' · Reconnection attempts: {snapshot["reconnects"]} · Display checked: {slst(pd.Timestamp.now(tz="UTC"))} SLST')
    frame = snapshot['candles']
    shown_interval = interval
    if frame.empty and hourly is not None and not hourly.empty:
        frame, shown_interval = hourly.copy(), '1h'
        st.info('The requested chart feed is not available yet. Showing completed 1-hour history until it arrives.')
    if not frame.empty:
        # Keep indicator warm-up before slicing the visible window.
        from live_data import calculate_indicators
        all_indicators = calculate_indicators(frame)
        tail = all_indicators.tail(bars)
        # intelligence_chart calculates the same causal indicators; pass enough
        # history and use a time window to retain their warm-up.
        days = max((frame.close_time.max() - tail.close_time.min()).total_seconds() / 86400, .00001)
        fig = intelligence_chart(frame, news, days=days, theme=theme, interval=shown_interval)
        fig.update_layout(uirevision=f'live-{shown_interval}-{bars}-{follow}', datarevision=str(frame.iloc[-1].get('updated_at', frame.close_time.iloc[-1])))
        if not follow:
            fig.update_xaxes(autorange=True, range=None)
        st.plotly_chart(fig, key='live_market_chart', **STRETCH, config={'displaylogo': False, 'scrollZoom': True})
        last = frame.iloc[-1]
        completed = frame[frame.close_time < pd.Timestamp.now(tz='UTC')]
        if 'closed' in completed:
            completed = completed[completed.closed.eq(True)]
        st.caption('Chart timeframe: ' + shown_interval + ' · Last completed chart candle: ' +
            slst(completed.close_time.iloc[-1] if not completed.empty else None) + ' SLST · Last candle update: ' +
            slst(last.get('updated_at', last.close_time)) + ' SLST')
        st.caption('Latest completed hourly model candle: ' +
            slst(hourly.close_time.max() if hourly is not None and not hourly.empty else None) +
            ' SLST · Last chart candle opens: ' + slst(last.open_time) + ' SLST · ' +
            ('FORMING — chart only' if last.close_time >= pd.Timestamp.now(tz='UTC') else 'Completed candle'))
        st.caption('The rightmost candle may still be forming; its indicators can change. Chart-only bars never enter the hourly ML model. News markers use the prior completed chart close and do not establish causation.')
        if not snapshot['chart_fresh'] or shown_interval != interval:
            st.warning('Chart updates are stale or unavailable for the selected timeframe. The chart is retained with its actual timestamps.')
        gaps = frame.open_time.diff().dropna().ne(pd.Timedelta(seconds={'1m': 60, '5m': 300, '15m': 900, '1h': 3600}[shown_interval])).sum()
        if gaps:
            st.warning(f'{gaps} chart interval gap(s) detected. Indicators are limited until public REST repairs the history; missing bars are not invented.')
    else:
        st.info('Chart history is loading or unavailable. Reconnection and public REST fallback run without blocking the page.')
    if snapshot['last_error'] or snapshot['rest_error']:
        with st.expander('Live connection details'):
            st.write('Last socket error: ' + str(snapshot['last_error'] or 'none'))
            st.write('Current REST error: ' + str(snapshot['rest_error'] or 'none'))
    st.caption('Trade/chart display refreshes every 2 seconds; the surrounding analysis refreshes every 30 seconds. All displayed times use Asia/Colombo (SLST).')


def observation_evidence(outlook, news):
    """Explain independent pieces of hourly evidence without invented confidence."""
    up, down, uncertain = [], [], []
    v = outlook.get('indicators')
    if v:
        (up if v['ema20'] > v['ema50'] else down if v['ema20'] < v['ema50'] else uncertain).append(
            'EMA trend: the faster hourly average is ' + ('above' if v['ema20'] > v['ema50'] else 'below' if v['ema20'] < v['ema50'] else 'equal to') + ' the slower average.')
        (up if v['macd_histogram'] > 0 else down if v['macd_histogram'] < 0 else uncertain).append(
            'MACD momentum is ' + ('above' if v['macd_histogram'] > 0 else 'below' if v['macd_histogram'] < 0 else 'at') + ' its signal line.')
        rsi = v['rsi']
        if 55 < rsi < 70:
            up.append(f'RSI {rsi:.1f}: recent upward moves are stronger.')
        elif 30 < rsi < 45:
            down.append(f'RSI {rsi:.1f}: recent downward moves are stronger.')
        else:
            uncertain.append(f'RSI {rsi:.1f}: ' + ('a stretched move; reversal is not guaranteed.' if rsi >= 70 or rsi <= 30 else 'momentum is near its middle range.'))
        uncertain.append('Volume: ' + ('unusually high in the completed hour; activity alone has no direction.' if v['volume_spike'] else 'no unusual spike in the completed hour.'))
    else:
        uncertain.append('Hourly indicator evidence is unavailable or stale.')
    recent = recent_news(news)
    if recent.empty:
        summary = 'No sufficiently recent eligible classified headlines. News does not confirm a direction.'
    else:
        counts = recent.label.value_counts()
        summary = f'{len(recent)} eligible headlines in six hours: {counts.get("Real_Adoption", 0)} adoption, {counts.get("Hype_Speculation", 0)} hype, {counts.get("Negative_FUD", 0)} negative, {counts.get("Neutral_Market_News", 0)} neutral.'
    uncertain.append('News content labels are context, not verified price sentiment or trading signals.')
    if up and down:
        uncertain.insert(0, 'Trend and momentum point in different directions. The evidence conflicts.')
    return dict(up=up, down=down, uncertain=uncertain, news=summary)


def similar_history(root, candles):
    """Read the audited existing descriptive study; never train or alter it."""
    from market_conditions_study import active_reserves, outside_reserves
    from market_intelligence import candle_quality
    from news_label_safety import sha256
    folder = Path(root) / 'reports' / 'market_conditions'
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    audit = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
    if report['study_type'] != 'descriptive_historical_outcomes' or audit['report_sha256'] != sha256(folder / 'report.json'):
        raise ValueError('Historical study audit does not match.')
    reserves = active_reserves(root)
    if reserves:
        dates = pd.read_csv(folder / 'observations.csv', usecols=['close_time', 'target_end'])
        if not outside_reserves(pd.to_datetime(dates.close_time, utc=True), pd.to_datetime(dates.target_end, utc=True), reserves).all():
            raise ValueError('Historical study overlaps an unfinished ML reserve.')
    issues = candle_quality(candles)
    if issues:
        raise ValueError(' '.join(issues))
    from live_data import calculate_indicators
    last = calculate_indicators(candles).iloc[-1]
    if not freshness(last.close_time, 3600)['fresh']:
        raise ValueError('Current hourly history is stale.')
    gap = (last.ema20 - last.ema50) / last.close
    name = 'Bullish' if gap > .001 else 'Bearish' if gap < -.001 else 'Sideways'
    return name, report['groups'][name], report['evaluation']


def render_observations(outlook, news, candles, root):
    st.subheader('Market observation · understand the evidence')
    st.caption('Technical evidence uses completed hourly candles. Selecting a faster chart does not change this analysis or the separate ML signal.')
    evidence = observation_evidence(outlook, news)
    for column, key, title in zip(st.columns(3), ['up', 'down', 'uncertain'],
            ['↑ Upward pressure', '↓ Downward pressure', '△ Uncertain / conflicting']):
        with column:
            with st.container(border=True):
                st.markdown('**' + title + '**')
                for value in evidence[key] or ['No clear supporting evidence.']:
                    st.write('• ' + value)
    st.write('**News context:** ' + evidence['news'])
    st.caption('Human annotations and AI-provisional categories remain separate in News intelligence. Neither establishes a causal price effect.')
    with st.expander('What happened historically in similar EMA conditions?'):
        try:
            name, group, period = similar_history(root, candles)
            st.write(f'Historical EMA regime: **{name}** · **{group["samples"]}** non-overlapping observations.')
            rows = []
            for label, value in group['outcomes'].items():
                interval = value['block_95']
                rows.append(dict(outcome=label, observations=value['count'], historical_frequency=f'{value["historical_frequency"]:.1%}',
                    descriptive_interval=f'{interval["lower"]:.1%}–{interval["upper"]:.1%}' if interval else 'Unavailable'))
            st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
            st.caption('Observed NEXT 24 HOURS: UP ≥ +1%, DOWN ≤ −1%, otherwise NEUTRAL. This existing study is not a 1-hour forecast and these frequencies are not future probabilities.')
            st.caption(f'Study origins: {slst(period["start"])} to {slst(period["end"])} SLST. EMA-only grouping omits many market differences. 25-hour spacing avoids overlapping outcomes; persistent regimes still cause dependence. Intervals use the existing moving-block method.')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.info('Comparable audited history unavailable: ' + str(exc))
    st.caption('Reliability: technical conditions are rules, news labels may be provisional, and the ML model is experimental. Historical similarities do not establish a reliable future outcome. Crypto prices can move sharply.')
