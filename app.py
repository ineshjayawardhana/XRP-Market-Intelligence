"""XRP Market Intelligence Dashboard. Run: streamlit run app.py"""
from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_data as source
from market_intelligence import (LABEL_COLORS, LABEL_NAMES, candle_quality, evaluate_history,
    freshness, intelligence_chart, market_outlook, outcome_counts, plot_times, recent_news, slst)
from model import predict_news

ROOT = Path(os.getenv('XRP_DASHBOARD_ROOT', str(Path(__file__).resolve().parent)))
NOTICE = 'This is for education only, not financial advice. Crypto is high risk.'
STRETCH = {'width': 'stretch'} if tuple(int(part) for part in st.__version__.split('.')[:2]) >= (1, 50) else {'use_container_width': True}
PALETTES = {
 'Dark': dict(bg='#0b1421', card='#142235', line='#283b52', text='#edf4ff', muted='#a0b1c8', positive='#34d399', negative='#fb7185', caution='#fbbf24', neutral='#a5b4c8'),
 'Light': dict(bg='#f3f6fb', card='#ffffff', line='#dce5f0', text='#16283e', muted='#556b83', positive='#047857', negative='#b91c1c', caution='#946200', neutral='#52647a')}


@st.cache_data(ttl=25, show_spinner=False)
def cached_ticker(root):
    return source.market_component('ticker', Path(root))


@st.cache_data(ttl=55, show_spinner=False)
def cached_candles(root):
    return source.market_component('candles', Path(root))


@st.cache_data(ttl=1, show_spinner=False)
def cached_public_news(root):
    # This is status only. Collection has its own process-lifetime scheduler.
    return source.news_poll_state(Path(root))


@st.cache_data(ttl=3600, show_spinner=False)
def historical_prices(days):
    if os.getenv('XRP_DASHBOARD_OFFLINE') == '1':
        raise RuntimeError('Offline mode: historical price requests are disabled.')
    return source.get_historical_prices(days=days)


def esc(value):
    return html.escape(str(value))


def number(value, pattern=',.4f', prefix=''):
    try:
        result = float(value)
        return prefix + format(result, pattern) if pd.notna(result) and abs(result) != float('inf') else 'Not available'
    except (ValueError, TypeError):
        return 'Not available'


def skin(theme):
    p = PALETTES[theme]
    st.markdown(f'''<style>
    .stApp,[data-testid="stAppViewContainer"]{{background:{p['bg']};color:{p['text']};}}
    [data-testid="stHeader"]{{background:{p['bg']};}}
    .block-container{{max-width:1480px;padding:2rem 2.5rem 3rem;}}
    [data-testid="stSidebar"]{{background:{p['card']};border-right:1px solid {p['line']};}}
    h1,h2,h3,h4,p,label,[data-testid="stMarkdownContainer"]{{color:inherit;}}
    [data-testid="stSidebar"] *,[data-testid="stWidgetLabel"]{{color:{p['text']};}}
    [data-baseweb="select"]>div,[data-baseweb="input"]>div,input,textarea{{background:{p['card']};color:{p['text']};}}
    [data-testid="stButton"] button{{background:{p['card']};color:{p['text']};border:1px solid {p['line']};}}
    [data-testid="stMetric"]{{background:{p['card']};border:1px solid {p['line']};border-radius:14px;padding:1rem;}}
    [data-testid="stMetricLabel"],[data-testid="stCaptionContainer"]{{color:{p['muted']};}}
    [data-testid="stMetricValue"]{{color:{p['text']};}}
    .eyebrow{{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:{p['muted']};font-weight:700;}}
    .hero{{background:{p['card']};border:1px solid {p['line']};border-radius:18px;padding:24px 28px;margin:12px 0 18px;}}
    .price{{font-size:clamp(36px,5vw,58px);font-weight:750;letter-spacing:-.04em;line-height:1.2;color:{p['text']};}}
    .fine{{color:{p['muted']};font-size:13px;line-height:1.7;}}
    .tag{{display:inline-block;border:1px solid {p['line']};padding:4px 9px;border-radius:20px;font-size:12px;margin:0 5px 8px 0;}}
    .condition{{font-size:clamp(26px,3vw,38px);font-weight:700;line-height:1.3;margin:10px 0;}}
    .news-card{{border:1px solid {p['line']};border-radius:14px;background:{p['card']};padding:18px 20px;margin-bottom:12px;min-height:175px;}}
    .news-card h4{{font-size:16px;line-height:1.5;margin:5px 0 12px;color:{p['text']};}}
    a{{color:{'#7dd3fc' if theme == 'Dark' else '#075985'};}}
    [data-testid="stAlertContainer"]{{border-radius:12px;}}
    @media(max-width:700px){{.block-container{{padding:1rem;}}.hero{{padding:18px;}}}}
    </style>''', unsafe_allow_html=True)


def news_card(row, theme):
    p = PALETTES[theme]
    label = row['label']
    color = LABEL_COLORS.get(label, p['caution'])
    score = ('Human content annotation' if row['label_status'] == 'Human verified' else
             ('Heuristic score ' if row.get('score_type', '').startswith('Heuristic') else 'Classifier score ') + number(row.get('score'), '.2f'))
    link = f'<a href="{esc(row["url"])}" target="_blank" rel="noopener noreferrer">Read article ↗</a>' if str(row.get('url', '')).startswith(('http://', 'https://')) else 'Article link unavailable'
    st.markdown(f'''<div class="news-card"><div class="eyebrow">{esc(row['source'])}</div>
    <h4>{esc(row['headline'])}</h4><span class="tag" style="color:{color}">{esc(LABEL_NAMES.get(label, label))}</span>
    <span class="tag">{esc(row['label_status'])}</span><div class="fine">{esc(score)} · Published: {esc(slst(row['published_at']))} SLST</div>
    <div class="fine">First seen: {esc(slst(row.get('first_seen_at', row.get('observed_at'))))} SLST · Collected: {esc(slst(row.get('collected_at')))} SLST</div>
    <div class="fine">{esc(row.get('classifier_backend') or 'Historical classification')} · {esc(row.get('model_provenance') or 'Provenance unrecorded')}</div>
    <div class="fine">{link}</div></div>''', unsafe_allow_html=True)


def legacy_trend_research(candles):
    # Inference only. Training is an explicit CLI research workflow.
    forecast = dict(validated=False, probabilities=None, experimental_label=None,
                    message='Predictive probabilities are not yet validated.',
                    limitation='No evaluated 24-hour direction model is available.',
                    model_version=None, feature_as_of=None, generated_at=None,
                    evaluated_at=None, sample_size=None, test_start=None, test_end=None,
                    quality_failures=[])
    try:
        from trend_pipeline import live_forecast
        forecast.update(live_forecast(candles, root=ROOT))
    except Exception as exc:
        forecast['limitation'] = 'Trend inference unavailable: ' + str(exc)
    st.subheader('Next 24 hours · research outlook')
    if not forecast['validated']:
        st.info('Predictive probabilities are not yet validated.')
        if forecast.get('experimental_label') in ('UP', 'DOWN', 'NEUTRAL'):
            st.warning('Experimental model signal: **' + forecast['experimental_label'] + '**. '
                       'The model has not passed all quality and calibration checks. This signal is not a reliable probability or a trading instruction.')
    else:
        st.caption('Chronologically evaluated research probabilities. Historical checks do not guarantee future performance.')
    for column, label, definition in zip(st.columns(3), ['UP', 'DOWN', 'NEUTRAL'], ['24h return ≥ +1%', '24h return ≤ −1%', 'Between −1% and +1%']):
        column.metric(label, f'{forecast["probabilities"][label]:.1%}' if forecast['validated'] else 'Not validated')
        column.caption(definition)
    st.caption('Model version: ' + str(forecast.get('model_version') or 'not available'))
    if forecast.get('feature_as_of'):
        input_freshness = freshness(forecast['feature_as_of'], 90 * 60)
        st.caption(f'Inputs through: {slst(forecast["feature_as_of"])} SLST · {input_freshness["description"]} · Inference run: {slst(forecast.get("generated_at"))} SLST')
    st.caption(f'Chronological trend-model test sample: {forecast["sample_size"] if forecast["sample_size"] is not None else "not available"} · Evaluation date: {slst(forecast["evaluated_at"])} SLST')
    if forecast['test_start'] and forecast['test_end']:
        st.caption(f'Holdout period: {slst(forecast["test_start"])} to {slst(forecast["test_end"])} SLST')
    st.caption(forecast['limitation'])
    if forecast.get('quality_failures'):
        with st.expander('Why probabilities are withheld'):
            for failure in forecast['quality_failures']:
                st.write('- ' + str(failure))
    st.caption('Evaluation metrics and baselines are available in Advanced research. Price targets and trading profits are not predicted.')


def trend_research(candles, ticker=None, reasons=()):
    from one_hour_ui import render_outlook
    render_outlook(candles, ROOT, ticker, reasons)


def overview(theme, live_news):
    from live_market_ui import render_live_market, render_observations, stream_snapshot
    stream_snapshot(ROOT)
    quote_result, candle_result = cached_ticker(str(ROOT)), cached_candles(str(ROOT))
    ticker, candles = quote_result['data'], candle_result['data']
    from streaming_market import model_candles, quote_context
    snapshot = stream_snapshot(ROOT)
    candles = model_candles(candles, snapshot)
    stats_ticker = ticker
    ticker = quote_context(ticker, snapshot)
    p = PALETTES[theme]
    errors = []
    if quote_result['error'] and not snapshot['trade_fresh']:
        errors.append(quote_result['error'])
    if candle_result['error'] and candle_quality(candles):
        errors.append(candle_result['error'])
    fresh = freshness((ticker or {}).get('retrieved_at'), 90)
    exchange_ok = not (ticker or {}).get('exchange_close_at') or freshness(ticker['exchange_close_at'], 120)['fresh']
    live = ticker and fresh['fresh'] and exchange_ok and not errors
    change = (ticker or {}).get('change_24h_pct')
    change_text = number(change, '+.2f') + '%' if change is not None else '24h change unavailable'
    change_color = p['positive'] if change is not None and change > 0 else p['negative'] if change is not None and change < 0 else p['neutral']
    news, warnings = source.load_news(ROOT)
    render_live_market(ROOT, stats_ticker, candles, news, theme)
    st.markdown(f'<span class="tag" style="color:{change_color}">{esc(change_text)} over 24h</span>', unsafe_allow_html=True)
    st.caption('Rolling 24h statistics snapshot: ' + slst((stats_ticker or {}).get('retrieved_at')) + ' SLST. This is separate from the trade stream.')
    if stats_ticker and not freshness(stats_ticker.get('retrieved_at'), 90)['fresh']:
        st.warning('The rolling 24-hour statistics are stale; their original timestamp is retained.')
    for column, key, label, pattern, prefix in zip(st.columns(4),
        ['high_24h', 'low_24h', 'volume_24h_xrp', 'volume_24h_usdt'],
        ['24h high', '24h low', '24h volume · XRP', '24h turnover · USDT'],
        [',.4f', ',.4f', ',.0f', ',.0f'], ['$', '$', '', '$']):
        column.metric(label, number((stats_ticker or {}).get(key), pattern, prefix))
    if errors:
        st.warning('A market refresh failed. Retained values keep their original timestamps; the current outlook is limited.')
        with st.expander('Market connection details'):
            for error in errors:
                st.write(error)
    elif not live:
        st.warning('Market data is stale or unavailable. A current price outlook cannot be established.')
    if live_news:
        with st.spinner('Checking permitted public news feeds…'):
            polling = cached_public_news(str(ROOT))
        news, warnings = source.load_news(ROOT)
        if polling['errors']:
            st.caption('News connection limited: showing saved headlines. Details are available in News intelligence.')
    outlook = market_outlook(ticker, candles, news, errors=errors)
    icon = {'positive': '↗', 'negative': '↘', 'neutral': '—', 'caution': '△'}[outlook['tone']]
    st.markdown(f'''<div class="hero"><div class="eyebrow">Current market condition · technical heuristic</div>
    <div class="condition" style="color:{p[outlook['tone']]}">{icon} {esc(outlook['condition'])}</div>
    <div class="fine">Current evidence, not a validated machine-learning prediction or a trading instruction.</div></div>''', unsafe_allow_html=True)
    st.markdown('**Why this condition?**')
    for index, reason in enumerate(outlook['reasons'], 1):
        st.write(f'{index}. {reason}')
    last_close = candles.close_time.iloc[-1] if candles is not None and not candles.empty and 'close_time' in candles else None
    st.caption(f'Indicator basis: completed 1-hour candles · Last close: {slst(last_close)} SLST · Recent-news context: {outlook["recent_news_count"]} classified headlines in six hours')
    render_observations(outlook, news, candles, ROOT)
    st.divider()
    with st.expander('Completed hourly archive chart · longer view'):
        days = st.select_slider('Chart window', options=[1, 7, 30], value=7, format_func=lambda value: f'{value} days', key='chart_window')
        quality = candle_quality(candles)
        if not quality:
            st.plotly_chart(intelligence_chart(candles, news, days, theme), **STRETCH, config={'displaylogo': False, 'scrollZoom': False}, key='hourly_archive_chart')
            st.caption('All chart times are SLST. News markers use first-available time and the prior closed hourly price as an approximation; they do not imply that news caused the move.')
        else:
            st.info('The chart will appear when complete hourly data is available. ' + ' '.join(quality))
    trend_research(candles, ticker, outlook['reasons'])
    st.divider()
    st.subheader('Latest available news')
    if not news.empty:
        for column, (_, row) in zip(st.columns(3), news.head(3).iterrows()):
            with column:
                news_card(row, theme)
        if recent_news(news).empty:
            st.caption('Archive context: no eligible article has both publication and observation times within six hours. These stories do not provide fresh directional confirmation.')
    else:
        st.info('No news is available. The market overview continues independently.')
    for warning in warnings:
        st.caption(warning)


def news_intelligence(theme, live_news):
    from live_news_ui import news_freshness, render_source_status
    st.header('AI news intelligence')
    st.caption('Content categories describe a story. They do not verify it or predict its price impact.')
    if live_news:
        with st.spinner('Checking public publisher feeds…'):
            polling = cached_public_news(str(ROOT))
        st.caption('Last completed feed check: ' + slst(polling['checked_at']) + ' SLST. A successful check does not make an old article recent.')
        for issue in polling['errors']:
            st.warning(issue)
    news, warnings = source.load_news(ROOT)
    for warning in warnings:
        st.warning(warning)
    news_freshness(news, live_news)
    render_source_status(ROOT, live_news)
    if news.empty:
        st.info('News is unavailable. Existing collection and labeling tools remain available independently.')
        return
    recent = recent_news(news, hours=24)
    st.subheader('News mix · last 24 hours')
    for column, label in zip(st.columns(4), LABEL_COLORS):
        column.metric(LABEL_NAMES[label], int(recent.label.eq(label).sum()) if not recent.empty else 0)
    st.caption('Only stories both observed and published in the last 24 hours. Provisional labels are unverified; counts do not establish price direction.')
    if not recent.empty:
        mix = recent.copy()
        mix['day'] = plot_times(mix['published_at']).dt.floor('D')
        grouped = mix.groupby(['day', 'label']).size().reset_index(name='headlines')
        fig = go.Figure()
        for label in LABEL_COLORS:
            subset = grouped[grouped.label.eq(label)]
            fig.add_bar(x=subset.day, y=subset.headlines, name=LABEL_NAMES[label], marker_color=LABEL_COLORS[label])
        fig.update_layout(template='plotly_dark' if theme == 'Dark' else 'plotly_white', barmode='group', height=280,
                          paper_bgcolor='rgba(0,0,0,0)', title='Publication-day counts · SLST', yaxis_title='Headlines')
        st.plotly_chart(fig, **STRETCH)
    filters = st.columns(3)
    labels = filters[0].multiselect('Category', list(LABEL_NAMES), format_func=lambda key: LABEL_NAMES[key])
    roles = filters[1].multiselect('Classification status', sorted(news.label_status.unique()))
    publishers = filters[2].multiselect('Source', sorted(news.source.unique()))
    selected = news.copy()
    for values, column in [(labels, 'label'), (roles, 'label_status'), (publishers, 'source')]:
        if values:
            selected = selected[selected[column].isin(values)]
    st.write(f'{len(selected)} available archive/live records match these filters.')
    table = selected[['headline', 'label', 'label_status', 'score', 'score_type', 'source', 'published_at', 'first_seen_at', 'collected_at', 'classifier_backend', 'model_provenance', 'url']].copy()
    table['label'] = table.label.map(LABEL_NAMES).fillna(table.label)
    table['published_at'] = table.published_at.map(slst)
    table['first_seen_at'] = table.first_seen_at.map(slst)
    table['collected_at'] = table.collected_at.map(slst)
    st.dataframe(table.head(200), hide_index=True, **STRETCH,
        column_config={'url': st.column_config.LinkColumn('Article'), 'score': st.column_config.NumberColumn('Uncalibrated score', format='%.2f'),
                       'published_at': 'Published · SLST', 'first_seen_at': 'First seen · SLST',
                       'collected_at': 'Collected · SLST', 'label_status': 'Annotation status'})
    st.caption('Human verified means a human chose the content category, not independently fact-checked. Legacy classifier provenance is unknown and excluded from the outlook.')
    with st.expander('Latest six stories in detail'):
        for _, row in selected.head(6).iterrows():
            news_card(row, theme)


def trend_evaluation(theme):
    st.subheader('24-hour trend model evaluation')
    report = source.json_file(ROOT / 'artifacts' / 'trend_24h_report.json', {})
    if not isinstance(report, dict) or not report:
        st.info('No historical trend-model evaluation is available. Training is a separate research command; opening this dashboard never starts it.')
        return
    metrics = report.get('holdout', {})
    data, split = report.get('data', {}), report.get('split', {})
    quality = report.get('quality', {})
    st.write('**' + str(report.get('selected_model', 'Unknown model')) + ' · ' + str(report.get('model_version', 'Version unavailable')) + '**')
    st.caption('Evaluation run: ' + slst(report.get('evaluated_at')) + ' SLST · Retrospective historical replay, not predictions recorded live in the past.')
    if quality.get('passed') is True:
        st.success('Recorded out-of-sample quality and calibration checks passed. Fresh input and matching model artifacts are also required before live probabilities can appear.')
    else:
        st.warning('This model remains experimental. It has not passed every documented out-of-sample quality and calibration check; probability estimates are withheld from the main dashboard.')
    for column, label, value in zip(st.columns(4),
        ['Hourly candles', 'Non-overlapping prediction origins', 'Held-out examples', '24h purge between partitions'],
        [data.get('hourly_candles'), data.get('nonoverlapping_samples', data.get('daily_samples')), metrics.get('sample_size', split.get('test_samples')), split.get('purge_hours')]):
        column.metric(label, value if value is not None else 'Not recorded')
    st.caption(f'Market history: {slst(data.get("start"))} to {slst(data.get("end"))} SLST · Source: {data.get("source", "not recorded")}')
    st.caption(f'Final holdout: {slst(split.get("test_start"))} to {slst(split.get("test_end"))} SLST · Selected before holdout: {report.get("selected_before_holdout") is True}')
    st.write('**Selected model on the final chronological holdout**')
    for column, key, label in zip(st.columns(4), ['accuracy', 'balanced_accuracy', 'macro_f1', 'calibration_ece'],
            ['Accuracy', 'Balanced accuracy', 'Macro-F1', 'Calibration error · ECE']):
        value = metrics.get(key)
        column.metric(label, number(value, '.3f') if value is not None else 'Not recorded')
    st.caption('Accuracy and balanced accuracy are proportions from 0 to 1. Macro-F1 weights the three outcomes equally. Lower ECE, log loss and Brier score indicate better probability calibration/scoring; they are historical diagnostics, not a guarantee.')
    rows = []
    for name, candidate in report.get('candidates', {}).items():
        if isinstance(candidate, dict) and isinstance(candidate.get('holdout'), dict):
            rows.append(dict(model=name, role='ML candidate', **candidate['holdout']))
    for name, baseline in report.get('baselines', {}).items():
        if isinstance(baseline, dict):
            rows.append(dict(model=name.replace('_', ' '), role='Simple baseline', **baseline))
    columns = ['model', 'role', 'sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss', 'brier_score', 'calibration_ece']
    if rows:
        table = pd.DataFrame(rows)
        st.dataframe(table[[key for key in columns if key in table]].round(4), hide_index=True, **STRETCH)
        st.caption('Candidate selection uses earlier walk-forward results. This shared holdout comparison must not be used to repeatedly tune and retest the final model.')
    with st.expander('Outcome counts, confusion matrix and classification report'):
        distribution = metrics.get('outcome_distribution', {})
        if distribution:
            st.dataframe(pd.DataFrame([{'Outcome': key, 'Held-out examples': value} for key, value in distribution.items()]), hide_index=True, **STRETCH)
        matrix = metrics.get('confusion_matrix')
        if isinstance(matrix, list) and len(matrix) == 3 and all(isinstance(row, list) and len(row) == 3 for row in matrix):
            confusion = pd.DataFrame(matrix, index=['UP', 'DOWN', 'NEUTRAL'], columns=['UP', 'DOWN', 'NEUTRAL'])
            confusion.index.name = 'Actual / predicted'
            st.write('Confusion matrix · rows are actual outcomes; columns are predictions')
            st.dataframe(confusion, **STRETCH)
        classification = metrics.get('classification_report', {})
        details = {key: value for key, value in classification.items() if isinstance(value, dict)}
        if details:
            st.dataframe(pd.DataFrame(details).transpose().round(4), **STRETCH)
    with st.expander('Walk-forward evaluation and chronological partitions'):
        st.caption(f'Each fold fits on earlier data and evaluates later data. Prediction origins are spaced {split.get("evaluation_stride_hours", "at least 24")} hours apart to avoid overlapping 24-hour outcomes; the additional purge keeps labels from crossing partition boundaries.')
        partitions = []
        for name in ['train', 'calibration', 'test']:
            partitions.append({'Partition': name, 'Start · SLST': slst(split.get(name + '_start')),
                               'End · SLST': slst(split.get(name + '_end')), 'Examples': split.get(name + '_samples')})
        st.dataframe(pd.DataFrame(partitions), hide_index=True, **STRETCH)
        for name, candidate in report.get('candidates', {}).items():
            walk = candidate.get('walk_forward', {}) if isinstance(candidate, dict) else {}
            if not walk:
                continue
            st.write('**' + str(name) + '**')
            summary = {key: walk[key] for key in ['sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss', 'brier_score', 'calibration_ece'] if key in walk}
            if summary:
                st.dataframe(pd.DataFrame([summary]).round(4), hide_index=True, **STRETCH)
            if walk.get('folds'):
                folds = pd.json_normalize(walk['folds'])
                preferred = ['fold', 'train_end', 'calibration_end', 'test_start', 'test_end', 'train_samples', 'calibration_samples', 'test_samples', 'accuracy', 'balanced_accuracy', 'macro_f1', 'metrics.accuracy', 'metrics.balanced_accuracy', 'metrics.macro_f1']
                visible = [key for key in preferred if key in folds]
                if visible:
                    fold_table = folds[visible].copy()
                    for key in visible:
                        if key.endswith(('_start', '_end')):
                            fold_table[key] = fold_table[key].map(slst)
                    fold_table = fold_table.rename(columns={key: key + ' · SLST' for key in visible if key.endswith(('_start', '_end'))})
                    st.dataframe(fold_table.round(4), hide_index=True, **STRETCH)
    with st.expander('Calibration checks and reliability bins'):
        from trend_metrics import QUALITY_MESSAGES
        for failure in quality.get('failures', []):
            st.write('- ' + QUALITY_MESSAGES.get(failure, str(failure)))
        if quality.get('checks'):
            st.json(quality['checks'])
        bins = metrics.get('reliability_bins')
        if bins:
            reliability = pd.DataFrame(bins)
            st.dataframe(reliability.round(4), hide_index=True, **STRETCH)
            if {'count', 'mean_probability', 'observed_accuracy'}.issubset(reliability.columns):
                observed = reliability[reliability['count'].gt(0)].dropna(subset=['mean_probability', 'observed_accuracy'])
                if not observed.empty:
                    figure = go.Figure()
                    figure.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect calibration reference', line=dict(color='#94a3b8', dash='dash'))
                    figure.add_scatter(x=observed.mean_probability, y=observed.observed_accuracy, mode='lines+markers', name='Held-out observations',
                                       text=observed['count'].map(lambda value: f'{value} examples'),
                                       hovertemplate='Mean probability: %{x:.3f}<br>Observed correct fraction: %{y:.3f}<br>%{text}<extra></extra>')
                    figure.update_layout(template='plotly_dark' if theme == 'Dark' else 'plotly_white', height=320,
                        paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(title='Mean held-out top-class probability', range=[0, 1]),
                        yaxis=dict(title='Observed correct fraction', range=[0, 1]), legend=dict(orientation='h'))
                    st.plotly_chart(figure, **STRETCH)
        st.caption('Calibration uses a separate earlier partition. The final holdout never fits the scaler, classifier or probability calibrator.')
    with st.expander('Historical news eligibility and limitations'):
        if report.get('news'):
            st.json(report['news'])
        st.caption('News features require evidence available at each prediction time. Human annotations and AI-provisional categories retain separate provenance; neither replaces observed price outcomes.')
        for limitation in report.get('limitations', []):
            st.write('- ' + str(limitation))


def research(theme):
    st.header('Advanced research')
    st.caption('Evidence, sample sizes and limitations. Historical observations are not validated forecasts.')
    from one_hour_ui import render_evaluation
    render_evaluation(ROOT)
    st.divider()
    st.caption('Archived 24-hour research · separate horizon and target definitions')
    trend_evaluation(theme)
    st.divider()
    readiness = source.json_file(ROOT / 'data' / 'multisource' / 'research' / 'readiness_report.json', {})
    if readiness:
        for column, key, label in zip(st.columns(4), ['processed_records', 'gold_records', 'silver_records', 'review_required'],
                ['Sourced records', 'Gold · human labels', 'Silver · provisional labels', 'Unresolved labels']):
            column.metric(label, readiness.get(key, 'Not available'))
        ready = readiness.get('tfidf', {}).get('can_begin_supervised_training')
        st.info('Real-news model readiness: ' + ('minimum data checks passed.' if ready else 'not ready. More human labels and chronological coverage are needed.'))
        st.caption('Saved readiness report: ' + slst(readiness.get('created_at')) + ' SLST. Run the existing research workflow after human review to update it.')
        with st.expander('Data quality and remaining requirements'):
            st.write(f'Missing publication timestamps: {readiness.get("missing_publication_timestamps", "unknown")}. Largest source share: {readiness.get("largest_source_share", 0):.1%}.')
            distribution = pd.DataFrame({'Human verified': readiness.get('gold_class_distribution', {}),
                                         'AI provisional': readiness.get('silver_class_distribution', {})}).fillna(0).astype(int)
            if not distribution.empty:
                distribution.index = [LABEL_NAMES.get(label, label) for label in distribution.index]
                st.dataframe(distribution, **STRETCH)
            for issue in readiness.get('tfidf', {}).get('issues', []):
                st.write('- ' + issue)
            st.write('Archive headlines retain actual first-available times; publication dates are not substituted into backtests.')
    else:
        st.info('Research readiness files are unavailable. The market overview does not require them.')
    st.subheader('News classifier evaluation')
    evaluations = source.model_evaluations(ROOT)
    if not evaluations:
        st.info('No model evaluation artifacts are available.')
    for evaluation in evaluations:
        metrics = evaluation['metrics']
        with st.container(border=True):
            st.write('**' + evaluation['name'] + ' · ' + evaluation['scope'] + '**')
            left, right = st.columns(2)
            accuracy = metrics.get('accuracy')
            left.metric('Recorded classification accuracy', number(accuracy * 100, '.1f') + '%' if accuracy is not None else 'Unknown')
            right.metric('Held-out examples', evaluation['sample_size'] if evaluation['sample_size'] is not None else 'Unknown')
            st.caption('Evaluation date: ' + slst(evaluation['evaluated_at']) + ' SLST. This measures headline categories, not future XRP returns.')
            st.caption('Chronological periods: ' + (str(evaluation['periods']) if evaluation['periods'] else 'not recorded; no chronological real-news claim is made.'))
            if 'Synthetic' in evaluation['scope']:
                st.warning('Synthetic teaching baseline only. Its score is not validated real-world news accuracy or 24-hour trend accuracy.')
            if metrics.get('report'):
                st.dataframe(pd.DataFrame(metrics['report']).transpose().round(3), **STRETCH)
    st.subheader('Historical 24-hour outcomes')
    st.write('UP: at least +1% · DOWN: at most −1% · NEUTRAL: between those thresholds.')
    period = st.selectbox('History window', [30, 90, 180], index=2, format_func=lambda value: f'{value} days')
    use_legacy = False
    if (ROOT / 'data' / 'historical_news.csv').exists():
        use_legacy = st.checkbox('Include my legacy historical_news.csv archive', value=False)
        st.caption('Legacy dates must represent when information was actually available. Its classifier labels remain AI provisional; timestamp authenticity is not independently verified.')
    if st.button('Evaluate historical outcomes', key='evaluate_history'):
        try:
            with st.spinner('Fetching closed hourly candles and evaluating observed outcomes…'):
                candles = historical_prices(period)
                news, _ = source.load_news(ROOT)
                if use_legacy:
                    news = pd.concat([news, source.optional_historical_news(ROOT)], ignore_index=True)
                st.session_state['historical_research'] = evaluate_history(candles, news, days=period)
        except Exception as exc:
            st.error('Historical evaluation unavailable: ' + str(exc))
    saved = st.session_state.get('historical_research')
    if saved:
        st.caption('Analysis run: ' + slst(saved['evaluated_at']) + f' SLST · Requested window: {saved["requested_days"]} days. No model was trained.')
        for name, title in [('baseline', 'Technical-only baseline'), ('combined', 'Exploratory news + technical analysis')]:
            results = saved[name]
            st.write('**' + title + '**')
            if results.empty:
                st.info('No matched, matured 24-hour samples are available. No result is invented.')
                continue
            timestamp = 'close_time' if name == 'baseline' else 'observed_at'
            st.caption(f'n={len(results)} · Signal period: {slst(results[timestamp].min())} to {slst(results[timestamp].max())} SLST')
            summary = outcome_counts(results)
            st.dataframe(summary, hide_index=True, **STRETCH)
            figure = go.Figure()
            for label, color in [('UP', '#10b981'), ('DOWN', '#f43f5e'), ('NEUTRAL', '#94a3b8')]:
                figure.add_bar(x=summary.Signal, y=summary[label], name=label, marker_color=color)
            figure.update_layout(barmode='group', template='plotly_dark' if theme == 'Dark' else 'plotly_white', height=310,
                                 yaxis_title='Observed outcomes · counts', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(figure, **STRETCH)
        if not saved['combined'].empty:
            from evaluate_real_signals import summarize_signal_comparison
            st.write('**Same-event comparison**')
            st.dataframe(summarize_signal_comparison(saved['combined']), hide_index=True, **STRETCH)
        st.warning('Combined results may use AI-provisional labels. They are not human-verified classification ground truth or an out-of-sample forecast test. Hourly entry approximation, overlapping events and changing regimes limit reliability; costs and slippage are not modeled.')
    else:
        st.info('Run the historical evaluation to see observed counts and sample periods. Nothing is calculated from invented prices.')
    with st.expander('Headline lab · existing classifier'):
        st.caption('The bundled classifier is a synthetic training demonstration unless a real model was explicitly configured. Scores are uncalibrated classification scores, never price probabilities.')
        text = st.text_input('Hypothetical headline', value='A wallet adds XRP Ledger payment tools for developers')
        if st.button('Classify this headline', key='classify_headline'):
            try:
                result = predict_news(text, backend=os.getenv('MODEL_BACKEND', 'tfidf'))
                st.write(LABEL_NAMES.get(result['label'], result['label']) + ' · Classifier score ' + number(result['confidence'], '.1f') + '% (uncalibrated)')
            except Exception as exc:
                st.error('Classifier unavailable: ' + str(exc))


def guide():
    st.header('Understand the dashboard')
    explanations = [
        ('Price and 24h change', 'XRP is quoted in USDT on Binance spot. The 24-hour change is a rolling historical change, not a forecast.'),
        ('EMA20 and EMA50 · the recent path', 'The chart lines smooth 20 and 50 candles of the selected timeframe. The market-condition summary and ML model use completed hourly candles. The faster line above the slower one can indicate upward pressure; crossing lines can indicate uncertainty.'),
        ('RSI · how stretched a move is', 'RSI summarizes recent upward versus downward moves. Above 70 or below 30 can mean a stretched move, not a guaranteed reversal.'),
        ('MACD · changes in momentum', 'MACD compares moving averages. The histogram above or below zero describes relative momentum; it is not a probability.'),
        ('Volume · participation', 'An unusually busy hour adds context. High volume alone says nothing about whether the next move will rise or fall.'),
        ('News labels · topic, not truth', 'Adoption, speculation, negative reporting and neutral news describe content. Human labels and AI-provisional labels remain separate.'),
        ('Historical outcomes versus predictions', 'Observed returns belong to the past. The 1-hour model uses fresh next-candle targets, training-only thresholds and separate chronological calibration. Older 24-hour research remains available for comparison. Only a model that passes all documented checks can display research probabilities; otherwise its class signal is experimental.'),
        ('Freshness and Sri Lankan time', 'Last trade, exchange candle update, local receipt, completed hourly close and article publication have separate timestamps. Public WebSocket trades update the price; a disconnected feed uses clearly labeled REST fallback. A forming candle is for the chart only. Display time is Asia/Colombo, SLST (UTC+05:30); stored research times remain UTC.')]
    for title, explanation in explanations:
        with st.container(border=True):
            st.subheader(title)
            st.write(explanation)


@st.fragment(run_every=30)
def page_content(page, theme, live_news):
    if page == 'Overview':
        overview(theme, live_news)
    elif page == 'News intelligence':
        news_intelligence(theme, live_news)
    elif page == 'Advanced research':
        research(theme)
    elif page == 'Market Conditions Research':
        from market_conditions_ui import render_market_conditions
        render_market_conditions(ROOT, theme)
    elif page == 'Paper Trading':
        from paper_trading_ui import render_paper_trading
        render_paper_trading(ROOT, theme)
    else:
        guide()


def main():
    st.set_page_config(page_title='XRP Market Intelligence', page_icon='◈', layout='wide', initial_sidebar_state='expanded')
    st.sidebar.markdown('### ◈ XRP Intelligence')
    st.sidebar.caption('A clearer view of the market')
    page = st.sidebar.radio('Explore', ['Overview', 'Paper Trading', 'News intelligence', 'Market Conditions Research', 'Advanced research', 'Indicator guide'], label_visibility='collapsed')
    theme = st.sidebar.selectbox('Appearance', ['Dark', 'Light'], key='appearance')
    live_news = st.sidebar.toggle('Poll public news feeds', value=True)
    from live_news_ui import run_news_polling, request_news_refresh
    run_news_polling(ROOT, live_news)
    st.sidebar.caption('Live price/chart refresh every 2 seconds. Analysis and displayed news refresh every 30 seconds. Public news is collected in the background about every five minutes.')
    if st.sidebar.button('Refresh data now'):
        cached_ticker.clear()
        cached_candles.clear()
        cached_public_news.clear()
        request_news_refresh(ROOT)
        st.session_state['news_manual_requested'] = True
        st.rerun()
    if st.session_state.pop('news_manual_requested', False):
        st.sidebar.info('News refresh requested. Source backoff and access limits still apply; results appear after collection completes.')
    st.sidebar.divider()
    st.sidebar.caption('Read-only market analysis. No trading or leverage integration.')
    skin(theme)
    st.markdown('<div class="eyebrow">Market clarity · research transparency</div>', unsafe_allow_html=True)
    st.title('XRP Market Intelligence')
    st.caption('Binance XRP/USDT spot · Sri Lanka time · SLST (UTC+05:30)')
    st.warning(NOTICE)
    page_content(page, theme, live_news)


if __name__ == '__main__':
    main()
