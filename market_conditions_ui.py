"""Read-only beginner-facing historical study, separate from forecast gates."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from news_label_safety import sha256

from market_intelligence import plot_times, slst
from trend_metrics import OUTCOMES
STRETCH = {'width': 'stretch'} if tuple(int(part) for part in st.__version__.split('.')[:2]) >= (1, 50) else {'use_container_width': True}


def render_market_conditions(root, theme='Dark'):
    st.header('Market Conditions Research')
    st.warning('Historical outcome frequencies are descriptions of this dataset, not validated future probabilities.')
    folder = Path(root) / 'reports' / 'market_conditions'
    try:
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        if report.get('study_type') != 'descriptive_historical_outcomes':
            raise ValueError('Unexpected study format.')
    except (OSError, ValueError) as exc:
        st.info('No readable study is available. Run: .\\.venv\\Scripts\\python.exe market_conditions_study.py')
        st.caption(str(exc))
        return
    try:
        from market_conditions_study import active_reserves, outside_reserves
        reserves = active_reserves(root)
        if reserves:
            dates = pd.read_csv(folder / 'observations.csv', usecols=['close_time', 'target_end'])
            starts, ends = pd.to_datetime(dates.close_time, utc=True), pd.to_datetime(dates.target_end, utc=True)
            if not outside_reserves(starts, ends, reserves).all():
                st.warning('Study withheld: an unfinished ML reserve overlaps these saved outcomes. Previously published history is already exposed and cannot become a new untouched test. Review the reserve dates and rebuild the study.')
                return
    except (OSError, ValueError, KeyError) as exc:
        st.warning('Study withheld because ML reserve boundaries could not be checked: ' + str(exc))
        return
    st.write('This study asks what happened after past market conditions. It does not estimate what will happen next or change the experimental model’s validation status.')
    try:
        audit = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
        if audit.get('report_sha256') != sha256(folder / 'report.json'):
            raise ValueError('The audit refers to an older report.')
        st.caption(f'Verified source: {audit["study_cache"]["rows"]:,} hourly records, with {audit["study_cache"]["incomplete_rows"]} partial/non-standard row excluded. The earlier model cache contained {audit["original_cache"]["rows"]:,}; the caches overlap and must not be added together.')
        st.info('This descriptive study includes previously evaluated ML test history. It is not separate forecasting validation. Unfinished ML reserves are excluded.')
        with st.expander('Audit: source dates, selected observations and ML boundaries'):
            st.write('Data consistency checks passed. This does not validate prediction probabilities.')
            st.json(audit['waterfall'])
            st.caption('Exact source timestamps below are UTC; the main charts and display timestamps use SLST.')
            st.json({key: audit['study_cache'][key] for key in ('earliest_open', 'earliest_close', 'latest_open', 'latest_close')})
            st.json(audit['interval_audit'])
            st.json(audit['ml_test_exposure'])
    except (OSError, ValueError, KeyError):
        audit = None
        st.caption('No current consistency audit is available. Run audit_market_conditions.py to reconcile source counts and ML boundaries.')
    for column, label, value in zip(st.columns(3), ['Non-overlapping observations', 'Reference-majority class', 'Sample spacing'],
            [report['evaluation']['samples'], report['reference']['majority'], '25 hours']):
        column.metric(label, value)
    st.caption(f'Study run: {slst(report["created_at"])} SLST · Market data ends: {slst(report["data"]["end"])} SLST')
    st.caption(f'Evaluation origins: {slst(report["evaluation"]["start"])} to {slst(report["evaluation"]["end"])} SLST. Older data defines the fixed majority baseline.')
    year = st.selectbox('Historical period (UTC year)', ['All evaluated years'] + sorted({str(row['year']) for row in report['annual']}), key='regime_study_year')
    groups = report['groups'] if year == 'All evaluated years' else {row['regime']: row for row in report['annual'] if str(row['year']) == year}
    st.subheader('What happened during each condition?')
    st.caption('Bullish, bearish and sideways are separate trend groups. High volatility is an overlapping flag, so its count must not be added to the other three.')
    fig, table = go.Figure(), []
    for label, color, pattern in [('UP', '#10b981', '/'), ('DOWN', '#f43f5e', 'x'), ('NEUTRAL', '#94a3b8', '.')]:
        names, frequencies, upper, lower, text = [], [], [], [], []
        for name, group in groups.items():
            value = group['outcomes'][label]
            if value['historical_frequency'] is None:
                continue
            fraction, interval = value['historical_frequency'], value['block_95']
            names.append(name)
            frequencies.append(100 * fraction)
            upper.append(100 * max(0, interval['upper'] - fraction) if interval else 0)
            lower.append(100 * max(0, fraction - interval['lower']) if interval else 0)
            text.append(f'{value["count"]} of {group["samples"]} observations')
            table.append(dict(condition=name, samples=group['samples'], outcome=label, count=value['count'],
                historical_frequency=f'{fraction:.1%}', block_interval=f'{interval["lower"]:.1%}–{interval["upper"]:.1%}' if interval else 'Unavailable',
                nominal_wilson_interval=f'{value["nominal_wilson_95"]["lower"]:.1%}–{value["nominal_wilson_95"]["upper"]:.1%}',
                small_sample=group['small_sample']))
        fig.add_bar(name=label, x=names, y=frequencies, text=text,
            marker=dict(color=color, pattern_shape=pattern), error_y=dict(type='data', symmetric=False, array=upper, arrayminus=lower),
            hovertemplate='%{x}<br>%{fullData.name}: %{y:.1f}% historically<br>%{text}<extra></extra>')
    fig.update_layout(template='plotly_dark' if theme == 'Dark' else 'plotly_white', height=400, barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(title='Historical outcome frequency (%)', range=[0, 100]),
        legend=dict(orientation='h'), title='Observed next-24-hour outcomes · descriptive only')
    st.plotly_chart(fig, **STRETCH)
    st.caption('UP: observed return ≥ +1%. DOWN: ≤ −1%. NEUTRAL: between those boundaries. Whiskers are pointwise 95% moving-block intervals, not prediction confidence.')
    st.dataframe(pd.DataFrame(table), hide_index=True, **STRETCH)
    if any(value['small_sample'] for value in groups.values()):
        st.warning('At least one group has fewer than 30 observations. Its frequency estimate is especially unstable.')
    for name, group in groups.items():
        if group['samples']:
            up = group['outcomes']['UP']
            st.write(f'**{name}:** {up["count"]} of {group["samples"]} past observations rose at least 1% over the following 24 hours. This is a historical count, not a forecast for today.')
    with st.expander('Compare simple baselines on the same observations'):
        rows = []
        for name, group in groups.items():
            for baseline, values in group['baselines'].items():
                interval = values['block_95']
                rows.append(dict(condition=name, baseline=baseline, samples=values['samples'], matches=values['matches'],
                    historical_match_rate=f'{values["historical_match_rate"]:.1%}' if values['historical_match_rate'] is not None else 'Unavailable',
                    block_interval=f'{interval["lower"]:.1%}–{interval["upper"]:.1%}' if interval else 'Unavailable'))
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
        st.caption('The majority class is fixed using reference outcomes before 2023. The technical rule uses existing EMA/RSI conditions at each origin. These are descriptive match rates, not validated model accuracy.')
    with st.expander('Historical price and regime timeline · SLST'):
        try:
            frame = pd.read_csv(folder / 'observations.csv')
            frame['close_time'] = pd.to_datetime(frame.close_time, utc=True)
            if year != 'All evaluated years':
                frame = frame[frame.close_time.dt.year.eq(int(year))]
            timeline = go.Figure(go.Scatter(x=plot_times(frame.close_time), y=frame.close, name='XRP/USDT close', line_color='#38bdf8'))
            for name, color, symbol in [('Bullish', '#10b981', 'triangle-up'), ('Bearish', '#f43f5e', 'triangle-down'), ('Sideways', '#94a3b8', 'square')]:
                subset = frame[frame.regime.eq(name)]
                timeline.add_scatter(x=plot_times(subset.close_time), y=subset.close, name=name, mode='markers', marker=dict(color=color, symbol=symbol, size=5))
            high = frame[frame.high_volatility.eq(True)]
            timeline.add_scatter(x=plot_times(high.close_time), y=high.close, name='High volatility', mode='markers', marker=dict(color='#f59e0b', symbol='circle-open', size=9))
            timeline.update_layout(template='plotly_dark' if theme == 'Dark' else 'plotly_white', height=380,
                paper_bgcolor='rgba(0,0,0,0)', xaxis_title='Sri Lanka time · SLST', yaxis_title='USDT', legend=dict(orientation='h'))
            st.plotly_chart(timeline, **STRETCH)
        except (OSError, ValueError, KeyError) as exc:
            st.info('Timeline unavailable: ' + str(exc))
    st.subheader('News events and later price changes')
    news = report['news']
    st.write(news['reason'])
    st.caption(f'Trustworthy timed category records: {news.get("trustworthy_timed_records", 0)} · Matured non-overlapping events: {news.get("matured_nonoverlapping_events", 0)} · Matched earlier controls: {news.get("matched_prior_controls", 0)}')
    st.caption('Gold and Silver labels remain distinct. Old publication dates do not prove today’s labels were available then. Price entry is at or after recorded availability, never at a convenient earlier price. Association would not establish causation.')
    with st.expander('Methods, dependence and limitations'):
        st.write('Bullish: EMA20 exceeds EMA50 by more than 0.1% of price. Bearish: it is below by that amount. Otherwise sideways. High volatility: current 24-hour volatility exceeds the 75th percentile of preceding hourly volatility observations over 90 days, with at least 30 days of earlier evidence.')
        st.write('Seven-origin moving blocks preserve some serial dependence. Blocks never bridge missing source periods. Nominal Wilson intervals are also provided for comparison, but their independence assumption may be too strong.')
        if audit:
            sensitivity = []
            for block, values in audit['interval_sensitivity'].items():
                for name, group in values.items():
                    for outcome, result in group['outcomes'].items():
                        interval = result['block_95']
                        sensitivity.append(dict(block_origins=int(block), condition=name, outcome=outcome,
                            interval=f'{interval["lower"]:.1%}–{interval["upper"]:.1%}' if interval else 'Unavailable'))
            st.caption('All-year sensitivity to 3, 7 or 14-origin blocks. None of these intervals fully removes long-range dependence or regime-change uncertainty. Sparse/zero cells can have collapsed bootstrap intervals; consult Wilson intervals and small-sample flags too.')
            st.dataframe(pd.DataFrame(sensitivity), hide_index=True, **STRETCH)
        for limitation in report['limitations']:
            st.write('- ' + limitation)
        st.json(report['data']['quality'])
