"""Standalone local batch annotation app. The live app.py dashboard is unchanged."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_labeling import LABELS, REVIEW, ROOT, read_csv, workflow_lock
from research_workflow import build, save_reviews

STRETCH = {'width': 'stretch'} if tuple(int(part) for part in st.__version__.split('.')[:2]) >= (1, 50) else {'use_container_width': True}


def main():
    st.set_page_config(page_title='XRP Research Batch Review', layout='wide')
    folder = Path(os.getenv('XRP_RESEARCH_DATA_DIR', str(ROOT / 'data' / 'multisource')))
    output = folder / 'research'
    st.title('XRP Research Batch Review')
    st.caption('Review content categories. Silver is provisional; Gold requires a saved human decision. Article facts are not independently verified.')
    path = output / 'all_classifications.csv'
    if not path.exists() or not (output / 'readiness_report.json').exists():
        st.warning('Prepare the research files before opening the review queue.')
        st.code('.\\.venv\\Scripts\\python.exe research_workflow.py', language='powershell')
        return
    report = json.loads((output / 'readiness_report.json').read_text(encoding='utf-8'))
    target_per_class = next(iter(report['verification_plan'].values()))['target_verified_total']
    if st.sidebar.button('Refresh datasets and readiness', key='refresh'):
        try:
            with workflow_lock(folder):
                build(folder, target=target_per_class)
            st.rerun()
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    data_fields, data_rows = read_csv(path)
    data = pd.DataFrame(data_rows, columns=data_fields)
    data['heuristic_confidence'] = pd.to_numeric(data['heuristic_confidence'])
    sample_fields, sample_rows = read_csv(output / 'verification_sample.csv')
    sample = pd.DataFrame(sample_rows, columns=sample_fields)
    metrics = st.columns(4)
    metrics[0].metric('Headlines assessed', len(data))
    metrics[1].metric('Gold human labels', report['gold_records'])
    metrics[2].metric('Silver provisional labels', report['silver_records'])
    metrics[3].metric('Uncertain pending records', report['review_required'])
    if 'saved_review_message' in st.session_state:
        st.success(st.session_state.pop('saved_review_message'))

    review_tab, readiness_tab, sample_tab = st.tabs(['Batch review', 'Research readiness', 'Verification plan'])
    view = st.sidebar.selectbox('Review set', ['Recommended sample', 'All pending', 'Uncertain only', 'Gold (read-only)', 'SKIP/rejected (read-only)'])
    if view == 'Gold (read-only)':
        filtered = data[data.label_status.eq('HUMAN_VERIFIED')].copy()
    elif view == 'SKIP/rejected (read-only)':
        filtered = data[data.label_status.isin(['HUMAN_SKIPPED', 'HUMAN_REJECTED'])].copy()
    else:
        filtered = data[data.label_status.eq('AI_PROVISIONAL')].copy()
        if view == 'Recommended sample':
            filtered = filtered[filtered.id.isin(sample.id)]
        elif view == 'Uncertain only':
            filtered = filtered[filtered.predicted_label.eq(REVIEW)]
    categories = st.sidebar.multiselect('Suggested category', list(LABELS) + [REVIEW])
    sources = st.sidebar.multiselect('Publisher', sorted(data.publisher_group.unique()))
    confidence = st.sidebar.multiselect('Heuristic confidence band', ['high', 'medium', 'low'])
    uncertainty = st.sidebar.multiselect('Uncertainty', ['high', 'medium', 'low'])
    for selected, column in [(categories, 'predicted_label'), (sources, 'publisher_group'),
                             (confidence, 'confidence_band'), (uncertainty, 'uncertainty')]:
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    group = st.sidebar.selectbox('Group counts by', ['Suggested category', 'Publisher', 'Confidence', 'Uncertainty'])
    group_column = {'Suggested category': 'predicted_label', 'Publisher': 'publisher_group',
                    'Confidence': 'confidence_band', 'Uncertainty': 'uncertainty'}[group]
    with review_tab:
        st.caption('Scores rank heuristic suggestions. They are not calibrated probabilities, truth scores or future price forecasts.')
        st.dataframe(filtered.groupby(group_column, dropna=False).size().rename('records').reset_index(), hide_index=True)
        st.write(f'{len(filtered)} records match these filters.')
        if filtered.empty:
            st.info('No records match the current filters.')
        else:
            filtered = filtered.sort_values(['sampling_time', 'source', 'id'])
            size = st.sidebar.selectbox('Rows per batch', [10, 25, 50, 100], index=1)
            identity = hashlib.sha256(('|'.join(filtered.id) + sha256_file(path)).encode()).hexdigest()[:16]
            pages = max(1, math.ceil(len(filtered) / size))
            page = st.sidebar.selectbox('Page', list(range(1, pages + 1)), key='page_' + identity)
            shown = filtered.iloc[(page - 1) * size:page * size].copy()
            protected = view in ('Gold (read-only)', 'SKIP/rejected (read-only)')
            columns = ['id', 'headline', 'predicted_label', 'publisher_group', 'heuristic_confidence',
                       'uncertainty', 'published_at', 'availability_at', 'context_kind', 'review_flags',
                       'label', 'label_status', 'prediction_fingerprint']
            shown = shown[columns]
            if protected:
                st.dataframe(shown.drop(columns=['prediction_fingerprint']), hide_index=True, **STRETCH)
                st.info('Existing human decisions are protected and cannot be overwritten here.')
            else:
                select_all = st.checkbox('Select all records shown in this batch', value=False, key=f'all_{identity}_{page}')
                shown.insert(0, 'selected', select_all)
                shown.insert(1, 'action', '')
                shown.insert(2, 'human_label', '')
                edited = st.data_editor(shown, hide_index=True, num_rows='fixed', **STRETCH,
                    disabled=[column for column in shown.columns if column not in ('selected', 'action', 'human_label')],
                    column_config={
                        'selected': st.column_config.CheckboxColumn('Select'),
                        'action': st.column_config.SelectboxColumn('Decision', options=['', 'accept', 'correct', 'reject', 'defer']),
                        'human_label': st.column_config.SelectboxColumn('Corrected category', options=[''] + list(LABELS)),
                        'prediction_fingerprint': None,
                        'headline': st.column_config.TextColumn('Headline', width='large'),
                        'heuristic_confidence': st.column_config.NumberColumn('Heuristic score', format='%.2f')},
                    key=f'editor_{identity}_{page}_{size}')
                st.caption('Edit individual rows, or apply one decision to the selected rows below. Blank decisions never create Gold labels.')
                controls = st.columns(3)
                reviewer = controls[0].text_input('Human reviewer name', key='reviewer_name')
                bulk_action = controls[1].selectbox('Default decision for selected rows', ['', 'accept', 'correct', 'reject', 'defer'])
                bulk_label = controls[2].selectbox('Default corrected category', [''] + list(LABELS))
                if st.button('Save selected decisions', type='primary', key='save_reviews'):
                    chosen = edited[edited['selected'].eq(True)].to_dict('records')
                    for row in chosen:
                        row['action'] = row['action'] or bulk_action
                        row['human_label'] = row['human_label'] or bulk_label
                    if not chosen:
                        st.warning('Select the records you have reviewed before saving.')
                    else:
                        try:
                            saved = save_reviews(folder, chosen, reviewer)
                        except (ValueError, RuntimeError, OSError) as exc:
                            st.error(str(exc))
                        else:
                            if not saved:
                                st.info('No decisions saved. Choose an action; blank or deferred rows remain unreviewed.')
                            else:
                                try:
                                    with workflow_lock(folder):
                                        build(folder, target=target_per_class)
                                except (ValueError, RuntimeError, OSError) as exc:
                                    st.warning(f'{saved} human decisions were saved, but report refresh failed: {exc}. Use Refresh datasets to recover.')
                                else:
                                    st.session_state['saved_review_message'] = f'Saved {saved} explicit human decisions. Gold and Silver files have been refreshed.'
                                    st.rerun()
            detail_id = st.selectbox('Inspect evidence for one displayed record', shown.id.tolist(),
                                     format_func=lambda key: data.loc[data.id.eq(key), 'headline'].iloc[0])
            detail = data[data.id.eq(detail_id)].iloc[0]
            with st.expander('Headline, reasoning and available context', expanded=True):
                st.write(detail['headline'])
                st.link_button('Open original article', detail['url'])
                st.write(detail['reasoning'])
                st.write(detail['context_excerpt'] or 'No substantive article excerpt is available. The suggestion is headline-only.')
                st.caption(f'Context type: {detail["context_kind"]}. Retrieved: {detail["context_retrieved_at"] or "unknown"}. Facts independently verified: no.')
    with readiness_tab:
        distributions = pd.DataFrame({'Category': LABELS,
                                      'Gold': [report['gold_class_distribution'][label] for label in LABELS],
                                      'Silver': [report['silver_class_distribution'][label] for label in LABELS]})
        st.dataframe(distributions, hide_index=True)
        st.write('TF-IDF training ready:', report['tfidf']['can_begin_supervised_training'])
        st.write('DistilBERT training ready:', report['distilbert']['can_begin_supervised_training'])
        for issue in report['tfidf']['issues']:
            st.write('- ' + issue)
        st.warning('Archive publication dates are used for sample coverage only. Training/backtesting uses first-available timestamps; most new headlines were collected on one day.')
        st.dataframe(pd.DataFrame(report['data_quality_flags'].items(), columns=['Quality flag', 'Records']), hide_index=True)
        st.write(f'Largest publisher share: {report["largest_source_share"]:.1%}. Missing publication dates: {report["missing_publication_timestamps"]}.')
        st.download_button('Download Gold dataset', (output / 'gold_dataset.csv').read_bytes(), 'xrp_gold_dataset.csv', 'text/csv')
        st.download_button('Download Silver dataset', (output / 'silver_dataset.csv').read_bytes(), 'xrp_silver_dataset.csv', 'text/csv')
    with sample_tab:
        st.caption('Target: 25 verified examples per category, counting existing Gold labels. Quartiles use source-reported publication dates when available, with source and confidence diversity where possible. Review can change the proposed category.')
        st.dataframe(pd.DataFrame(report['verification_plan']).transpose(), **STRETCH)
        st.dataframe(sample[['headline', 'predicted_label', 'source', 'sampling_time', 'sampling_time_basis', 'sampling_role']], hide_index=True, **STRETCH)
        st.download_button('Download verification sample', (output / 'verification_sample.csv').read_bytes(), 'xrp_verification_sample.csv', 'text/csv')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
