"""Main 1h outlook and read-only evaluation; old 24h research stays separate."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from market_intelligence import freshness, slst


def render_outlook(candles, root, ticker=None, reasons=()):
    from one_hour_trend import live_forecast
    forecast = live_forecast(candles, root=root)
    st.subheader('Next 1 Hour — Research Outlook')
    st.caption('Target: the next completed hourly close compared with the latest completed hourly close. This is not a fresh 60-minute window from each page refresh.')
    if ticker:
        st.metric('Current XRP/USDT price', f'${ticker["price"]:,.5f}')
        st.caption('Last successful quote: ' + slst(ticker.get('retrieved_at')) + ' SLST')
        if not freshness(ticker.get('retrieved_at'), 90)['fresh']:
            st.warning('The displayed quote is stale. It retains its original successful-update timestamp.')
    threshold = forecast.get('threshold')
    if not forecast['validated']:
        st.info('Predictive probabilities are not yet validated.')
        if forecast.get('experimental_label'):
            st.warning('Experimental model signal: **' + forecast['experimental_label'] + '**. Low reliability: the 1-hour model has not passed all required quality, calibration and recent-market checks. This is not a trading instruction.')
    else:
        st.caption('Chronologically evaluated research probabilities; future performance is not guaranteed.')
    definitions = ([f'1h return ≥ +{threshold:.3%}', f'1h return ≤ −{threshold:.3%}', f'Between ±{threshold:.3%}']
                   if threshold else ['Training-derived UP boundary unavailable', 'Training-derived DOWN boundary unavailable', 'Training-derived neutral band unavailable'])
    for column, label, definition in zip(st.columns(3), ['UP', 'DOWN', 'NEUTRAL'], definitions):
        column.metric(label, f'{forecast["probabilities"][label]:.1%}' if forecast['validated'] else 'Not validated')
        column.caption(definition)
    st.caption('1-hour model version: ' + str(forecast.get('model_version') or 'not trained'))
    latest = forecast.get('feature_as_of')
    if latest is None and candles is not None and not candles.empty and 'close_time' in candles:
        latest = candles.close_time.iloc[-1]
    if latest:
        st.caption('Latest completed hourly candle: ' + slst(latest) + ' SLST · ' + freshness(latest, 3600)['description'])
        st.caption('Target candle close: ' + slst(pd.Timestamp(latest) + pd.Timedelta(hours=1)) + ' SLST · Inference run: ' + slst(forecast.get('generated_at')) + ' SLST')
        if pd.Timestamp(latest) != pd.Timestamp.now(tz='UTC').floor('h') - pd.Timedelta(milliseconds=1):
            st.warning('The latest required completed hourly candle is unavailable. A current 1-hour signal is withheld.')
    st.caption('1-hour historical test period: ' + slst(forecast.get('test_start')) + ' to ' + slst(forecast.get('test_end')) + ' SLST. Evaluation run: ' + slst(forecast.get('evaluated_at')) + ' SLST')
    measured = forecast.get('metrics')
    if measured:
        for column, key, label in zip(st.columns(4), ['accuracy', 'balanced_accuracy', 'macro_f1', 'sample_size'],
                ['1h test accuracy', '1h balanced accuracy', '1h macro-F1', '1h held-out examples']):
            value = measured[key]
            column.metric(label, value if key == 'sample_size' else f'{value:.1%}' if key in ('accuracy', 'balanced_accuracy') else f'{value:.3f}')
        try:
            # Read the exact report already verified by live_forecast. No fit,
            # changed quality thresholds or new evaluation is performed here.
            from news_label_safety import sha256
            pointer = json.loads((Path(root) / 'artifacts' / 'trend_1h_current.json').read_text(encoding='utf-8'))
            entry = pointer['files']['report']
            report_path = (Path(root) / entry['path']).resolve()
            report_path.relative_to((Path(root) / 'artifacts' / 'trend_1h_runs').resolve())
            if pointer['version'] != forecast['model_version'] or sha256(report_path) != entry['sha256']:
                raise ValueError('Report/model mismatch.')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            st.caption(f'Same historical test: model accuracy {measured["accuracy"]:.2%} · majority baseline {report["baselines"]["majority"]["accuracy"]:.2%} · technical rules {report["baselines"]["technical_rule"]["accuracy"]:.2%}. Historical improvement does not establish current reliability.')
        except (OSError, ValueError, KeyError, TypeError):
            st.caption('Baseline comparison is unavailable here; see Advanced research for verified evaluation details.')
    st.caption(forecast['limitation'])
    if reasons:
        with st.expander('Three reasons for the current technical condition'):
            st.caption('These explain the technical condition above, not the internal reasoning or confidence of the ML classifier.')
            for index, reason in enumerate(reasons, 1):
                st.write(f'{index}. {reason}')
    st.caption('Detailed 1-hour evaluation and archived 24-hour model results are in Advanced research. Neither is an exact-price prediction.')


def render_evaluation(root):
    st.subheader('1-hour model evaluation')
    pointer = Path(root) / 'artifacts' / 'trend_1h_current.json'
    if not pointer.exists():
        st.info('No 1-hour research model is available. Train explicitly with: .\\.venv\\Scripts\\python.exe train_one_hour.py')
        return
    try:
        from one_hour_trend import verified_model
        _, report, quality = verified_model(root)
    except Exception as exc:
        st.warning('1-hour evaluation could not be verified: ' + str(exc))
        return
    st.write('**' + report['selected_model'] + ' · ' + report['version'] + '**')
    st.caption('Retrospective replay on a newly collected older historical reserve. It is not a prospective current-market track record.')
    st.write(f'Final training-only boundary: UP ≥ +{report["threshold_pct"]:.4f}%, DOWN ≤ −{report["threshold_pct"]:.4f}%, otherwise NEUTRAL.')
    st.caption(report['threshold_rule'])
    table = [dict(model=name, scope='Development validation only', **{key: value['validation'][key] for key in
        ['sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece']}) for name, value in report['candidates'].items()]
    table.append(dict(model=report['selected_model'], scope='Frozen final test', **{key: report['test'][key] for key in
        ['sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece']}))
    for name, value in report['baselines'].items():
        table.append(dict(model=name, scope='Same final test', **{key: value[key] for key in
            ['sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece']}))
    stretch = {'width': 'stretch'} if tuple(int(part) for part in st.__version__.split('.')[:2]) >= (1, 50) else {'use_container_width': True}
    st.dataframe(pd.DataFrame(table).round(4), hide_index=True, **stretch)
    st.caption('Accuracy, balanced accuracy and ECE in this table are fractions from 0 to 1. Macro-F1 also ranges from 0 to 1; lower Brier score and log loss are better.')
    if quality['passed']:
        st.info('Historical numerical quality and calibration checks passed on this older test. That alone does not validate probabilities for the current market regime.')
    st.caption('Only development validation selects the winning family. Final-test scores do not choose a different model. Majority and technical rules have one-hot outputs; the training-prior distribution is the probabilistic scoring baseline.')
    with st.expander('1-hour confusion matrix, class report and calibration'):
        st.dataframe(pd.DataFrame(report['test']['confusion_matrix'], index=['Actual UP', 'Actual DOWN', 'Actual NEUTRAL'], columns=['Predicted UP', 'Predicted DOWN', 'Predicted NEUTRAL']), **stretch)
        details = {key: value for key, value in report['test']['classification_report'].items() if isinstance(value, dict)}
        st.dataframe(pd.DataFrame(details).transpose().round(4), **stretch)
        st.json(report['test']['outcome_distribution'])
        st.dataframe(pd.DataFrame(report['test']['reliability_bins']), **stretch)
        st.json(quality)
    with st.expander('1-hour chronological splits, fold thresholds and limitations'):
        st.json(report['split'])
        fold_rows = []
        for name, value in report['candidates'].items():
            for fold in value['folds']:
                fold_rows.append(dict(model=name, fold=fold['fold'], threshold_pct=100 * fold['threshold'],
                    train_end=slst(fold['train']['end']), calibration_end=slst(fold['calibration']['end']),
                    validation_start=slst(fold['validation']['start']), macro_f1=fold['metrics']['macro_f1']))
        st.dataframe(pd.DataFrame(fold_rows), **stretch)
        for item in report['limitations']:
            st.write('- ' + item)
        st.write('Probability display also requires recent out-of-sample market evidence (within 180 days), matching artifacts and the exact latest completed hourly candle. An older historical pass alone does not validate current-regime percentages.')
