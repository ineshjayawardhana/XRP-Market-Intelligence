"""Build disjoint Silver/Gold datasets and an auditable human-review plan.

Never fits a model or alters source labels. Only explicit review submissions can
append human decisions to the existing multisource human ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from ai_labeling import (DECISION_FIELDS, LABELS, REVIEW, ROOT, atomic_csv, backup, dataset,
                         decide, load_decisions, matching, now, read_csv, workflow_lock)
from batch_news_labeling import read_json_rows
from news_label_safety import atomic_json, duplicate_groups, human_origin_allowed, sha256, utc_date
from prepare_real_training import prepare
from research_context import fetch_additional
from research_labels import VERSION, classify_research, clean_excerpt

FIELDS = ['id', 'headline', 'source', 'publisher_group', 'url', 'date', 'timestamp_type',
          'published_at', 'gdelt_first_seen_at', 'availability_at', 'sampling_time', 'sampling_time_basis',
          'predicted_label', 'heuristic_confidence', 'confidence_band', 'uncertainty', 'reasoning',
          'label', 'label_status', 'label_origin', 'human_reviewed_by', 'human_reviewed_at',
          'article_verified', 'context_kind', 'context_excerpt', 'context_retrieved_at', 'context_sha256',
          'classification_method', 'previous_ai_label', 'needs_manual_review', 'review_flags',
          'duplicate_group', 'canonical_record', 'eligible_content', 'prediction_fingerprint']
SAMPLE_FIELDS = FIELDS + ['sampling_role', 'publication_time_bin']


def counts(rows, key='label'):
    counter = Counter(row[key] for row in rows)
    return {label: counter.get(label, 0) for label in LABELS}


def protected_humans(rows, decisions):
    protected = {}
    conflicts = []
    for row in rows:
        decision = decisions.get(row['id'])
        valid_decision = decision if decision and matching(row, decision) else None
        label = row['label'].strip()
        if label == 'SKIP':
            protected[row['id']] = dict(label='SKIP', status='HUMAN_SKIPPED', origin='human_skip', reviewer='', reviewed_at='')
            continue
        if label in LABELS and human_origin_allowed(row):
            protected[row['id']] = dict(label=label, status='HUMAN_VERIFIED',
                                       origin=row.get('label_origin') or 'human_manual_legacy',
                                       reviewer=row.get('human_reviewed_by', ''), reviewed_at=row.get('human_reviewed_at', ''))
        if valid_decision:
            if row['id'] in protected and valid_decision['human_label'] != label:
                conflicts.append(row['id'])
                continue
            if valid_decision['review_status'] == 'rejected':
                protected[row['id']] = dict(label='', status='HUMAN_REJECTED', origin='human_rejection',
                                           reviewer=valid_decision['reviewed_by'], reviewed_at=valid_decision['reviewed_at'])
            else:
                if not utc_date(valid_decision['reviewed_at']):
                    raise ValueError('Human decision has invalid UTC review time')
                protected[row['id']] = dict(label=valid_decision['human_label'], status='HUMAN_VERIFIED',
                                           origin='human_reviewed_ai', reviewer=valid_decision['reviewed_by'],
                                           reviewed_at=valid_decision['reviewed_at'])
    return protected, conflicts


def load_inputs(folder):
    folder = Path(folder)
    _, rows = dataset(folder / 'xrp_real_labeled.csv')
    contexts = {row['id']: row for row in read_json_rows(folder / 'xrp_article_context.json')}
    for row in read_json_rows(folder / 'research' / 'additional_context.json'):
        source = next((item for item in rows if item['id'] == row['id']), None)
        if source and matching(source, row) and row['status'] == 'retrieved_unverified':
            contexts[row['id']] = row
    previous_path = folder / 'xrp_auto_labeled.csv'
    previous = {row['id']: row for row in read_csv(previous_path)[1]} if previous_path.exists() else {}
    proposals = {row['id']: row for row in read_json_rows(folder / 'xrp_ai_proposals.json')}
    decisions = {row['id']: row for row in load_decisions(folder / 'xrp_human_reviews.csv')}
    return rows, contexts, previous, proposals, decisions


def make_classifications(rows, contexts, previous, proposals, decisions):
    protected, human_conflicts = protected_humans(rows, decisions)
    output = []
    for row in rows:
        label, score, reason, flags, excerpt, method = classify_research(
            row, contexts.get(row['id']), previous.get(row['id']), proposals.get(row['id']))
        human = protected.get(row['id'])
        if human and human['status'] == 'HUMAN_VERIFIED' and label in LABELS and label != human['label']:
            flags.append('human_ai_label_conflict')
        if row['id'] in human_conflicts:
            flags.append('conflicting_human_annotations')
        availability = row.get('availability_at') or row['date']
        available_time = utc_date(availability)
        published = utc_date(row.get('published_at', ''))
        good_pub = bool(published and published <= utc_date(now()))
        sampling_time = published.isoformat() if good_pub else available_time.isoformat() if available_time else availability
        if not available_time or available_time > utc_date(now()):
            flags.append('invalid_or_future_availability')
        if not row.get('published_at'):
            flags.append('publication_time_missing')
        context = contexts.get(row['id'], {})
        context_valid = matching(row, context)
        context_kind = context.get('context_kind') or row.get('context_kind') or 'CACHED_ARTICLE_EXCERPT'
        if not excerpt:
            context_kind = 'HEADLINE_ONLY'
        digest = hashlib.sha256(excerpt.encode('utf-8')).hexdigest() if excerpt else ''
        fingerprint = hashlib.sha256(json.dumps([row['id'], row['headline'], row['url'], row['date'],
                                                label, score, digest, VERSION]).encode()).hexdigest()
        high_uncertainty = label == REVIEW or 'changed_from_previous_ai_label' in flags
        output.append(dict(id=row['id'], headline=row['headline'], source=row['source'],
                           publisher_group=(urlparse(row['url']).hostname or row['source']).lower().removeprefix('www.'),
                           url=row['url'], date=row['date'], timestamp_type=row.get('timestamp_type', ''),
                           published_at=row.get('published_at', ''), gdelt_first_seen_at=row.get('gdelt_first_seen_at', ''),
                           availability_at=availability, sampling_time=sampling_time,
                           sampling_time_basis='SOURCE_REPORTED_PUBLICATION_FOR_SAMPLING_ONLY' if good_pub else 'FIRST_AVAILABLE',
                           predicted_label=label, heuristic_confidence=round(score, 3),
                           confidence_band='high' if score >= .85 else 'medium' if score >= .7 else 'low',
                           uncertainty='high' if high_uncertainty else 'medium' if flags or score < .8 else 'low',
                           reasoning=reason, label=human['label'] if human else '',
                           label_status=human['status'] if human else 'AI_PROVISIONAL',
                           label_origin=human['origin'] if human else VERSION,
                           human_reviewed_by=human['reviewer'] if human else '',
                           human_reviewed_at=human['reviewed_at'] if human else '', article_verified='False',
                           context_kind=context_kind, context_excerpt=excerpt,
                           context_retrieved_at=context.get('fetched_at', '') if context_valid else '',
                           context_sha256=digest, classification_method=method,
                           previous_ai_label=previous.get(row['id'], {}).get('predicted_label', ''),
                           needs_manual_review=str(not human or 'human_ai_label_conflict' in flags),
                           review_flags=';'.join(sorted(set(flags))), duplicate_group='', canonical_record='True',
                           eligible_content=str(bool(available_time and available_time <= utc_date(now())
                                                and 'conflicting_human_annotations' not in flags)),
                           prediction_fingerprint=fingerprint))
    for group in duplicate_groups(output):
        identifier = 'group_' + min(row['id'] for row in group)
        humans = [row for row in group if row['label_status'] == 'HUMAN_VERIFIED']
        labels = {row['label'] if humans else row['predicted_label'] for row in humans or group}
        labels.discard(REVIEW)
        canonical = min(humans or group, key=lambda row: (utc_date(row['availability_at']) or utc_date('9999-12-31T00:00:00+00:00'), row['id']))
        for row in group:
            row['duplicate_group'] = identifier
            row['canonical_record'] = str(row is canonical)
            if len(group) > 1:
                row['review_flags'] += ';detected_duplicate_group'
            if len(labels) > 1:
                row['eligible_content'] = 'False'
                row['review_flags'] += ';conflicting_duplicate_labels'
                row['uncertainty'] = 'high'
                row['needs_manual_review'] = 'True'
    return output


def research_datasets(all_rows):
    gold = [dict(row) for row in all_rows if row['label_status'] == 'HUMAN_VERIFIED']
    gold_groups = {row['duplicate_group'] for row in gold}
    silver = [dict(row, label=row['predicted_label']) for row in all_rows
              if row['label_status'] == 'AI_PROVISIONAL' and row['predicted_label'] in LABELS
              and row['duplicate_group'] not in gold_groups and row['canonical_record'] == 'True'
              and row['eligible_content'] == 'True']
    if {row['id'] for row in gold} & {row['id'] for row in silver}:
        raise AssertionError('Silver and Gold must be disjoint')
    return silver, gold


def verification_sample(all_rows, gold, target=25, uncertainty_count=12):
    if not 20 <= target <= 30:
        raise ValueError('Verification target must be between 20 and 30 per category')
    gold_counts = counts([row for row in gold if row['canonical_record'] == 'True' and row['eligible_content'] == 'True'])
    selected, plan = [], {}
    gold_groups = {row['duplicate_group'] for row in gold}
    candidates = [row for row in all_rows if row['label_status'] == 'AI_PROVISIONAL'
                  and row['canonical_record'] == 'True' and row['duplicate_group'] not in gold_groups
                  and row['eligible_content'] == 'True']
    for label in LABELS:
        pool = sorted([row for row in candidates if row['predicted_label'] == label],
                      key=lambda row: (row['sampling_time'], row['id']))
        missing = max(0, target - gold_counts[label])
        source_counts, confidence_counts = Counter(), Counter()
        buckets = [pool[index * len(pool) // 4:(index + 1) * len(pool) // 4] for index in range(4)]
        chosen = []
        while len(chosen) < min(missing, len(pool)):
            for index, bucket in enumerate(buckets):
                if not bucket or len(chosen) >= missing:
                    continue
                best = min(bucket, key=lambda row: (source_counts[row['publisher_group']],
                           confidence_counts[row['confidence_band']], row['id']))
                bucket.remove(best)
                chosen.append(dict(best, sampling_role='category_verification', publication_time_bin=f'quartile_{index+1}'))
                source_counts[best['publisher_group']] += 1
                confidence_counts[best['confidence_band']] += 1
        selected.extend(chosen)
        plan[label] = dict(target_verified_total=target, already_gold=gold_counts[label],
                           provisional_candidates=len(pool), suggested_new_reviews=len(chosen),
                           remaining_if_all_confirmed=max(0, missing - len(chosen)))
    uncertain = sorted([row for row in candidates if row['predicted_label'] == REVIEW], key=lambda row: (row['sampling_time'], row['id']))
    if uncertain:
        for index in range(min(uncertainty_count, len(uncertain))):
            row = uncertain[index * len(uncertain) // min(uncertainty_count, len(uncertain))]
            selected.append(dict(row, sampling_role='uncertainty_audit', publication_time_bin='spread_over_time'))
    return selected, plan


def build(folder, target=25, fetch_limit=0, max_requests=40, retry_unavailable=False):
    folder = Path(folder)
    output = folder / 'research'
    output.mkdir(parents=True, exist_ok=True)
    protected_paths = [path for path in folder.iterdir() if path.is_file() and path.suffix in ('.csv', '.json', '.sqlite3')]
    protected_hashes = {path.name: sha256(path) for path in protected_paths}
    rows, contexts, previous, proposals, decisions = load_inputs(folder)
    retrieval = None
    if fetch_limit:
        retrieval = fetch_additional(folder, rows, contexts, fetch_limit, max_requests, retry_unavailable)
        rows, contexts, previous, proposals, decisions = load_inputs(folder)
    all_rows = make_classifications(rows, contexts, previous, proposals, decisions)
    silver, gold = research_datasets(all_rows)
    sample, plan = verification_sample(all_rows, gold, target)
    for name, values, fields in [('all_classifications.csv', all_rows, FIELDS), ('silver_dataset.csv', silver, FIELDS),
                                 ('gold_dataset.csv', gold, FIELDS), ('verification_sample.csv', sample, SAMPLE_FIELDS),
                                 ('review_queue.csv', [row for row in all_rows if row['label_status'] == 'AI_PROVISIONAL'], FIELDS)]:
        path = output / name
        if path.exists():
            backup(path)
        atomic_csv(path, fields, values)
    prepared = prepare(folder, output / 'training_prepared')
    source_counts = Counter(row['publisher_group'] for row in all_rows)
    flags = Counter(flag for row in all_rows for flag in row['review_flags'].split(';') if flag)
    verified_ready = prepared['ready_for_training'] and not any('conflicting_human_annotations' in row['review_flags'] for row in gold)
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ('torch', 'transformers', 'datasets')}
    access_records = read_json_rows(output / 'additional_context.json')
    report = dict(created_at=now(), rubric_version=VERSION, processed_records=len(rows),
                  total_usable_distinct_headlines=len({row['duplicate_group'] for row in all_rows if row['eligible_content'] == 'True' and row['label_status'] not in ('HUMAN_SKIPPED', 'HUMAN_REJECTED')}),
                  usable_definition='Distinct non-SKIP/non-rejected headlines with valid availability; includes unresolved annotation candidates, not training-ready examples.',
                  gold_records=len(gold), gold_class_distribution=counts(gold),
                  distinct_eligible_gold_records=sum(row['canonical_record'] == 'True' and row['eligible_content'] == 'True' for row in gold),
                  silver_records=len(silver), silver_class_distribution=counts(silver),
                  unreviewed_provisional_assignments=sum(row['label_status'] == 'AI_PROVISIONAL' and row['predicted_label'] in LABELS for row in all_rows),
                  review_required=sum(row['label_status'] == 'AI_PROVISIONAL' and row['predicted_label'] == REVIEW for row in all_rows),
                  human_skip_count=sum(row['label_status'] == 'HUMAN_SKIPPED' for row in all_rows),
                  human_rejection_count=sum(row['label_status'] == 'HUMAN_REJECTED' for row in all_rows),
                  reused_previous_assessments=sum('cached_assessment' in row['classification_method'] for row in all_rows),
                  source_distribution=dict(source_counts), largest_source_share=max(source_counts.values(), default=0) / max(1, len(rows)),
                  source_imbalance_warning=max(source_counts.values(), default=0) / max(1, len(rows)) > .30,
                  missing_publication_timestamps=sum(not row['published_at'] for row in all_rows),
                  usable_context_records=sum(bool(row['context_excerpt']) for row in all_rows),
                  availability_dates=dict(Counter(row['availability_at'][:10] for row in all_rows)),
                  data_quality_flags=dict(flags), verification_plan=plan, suggested_review_sample=len(sample),
                  additional_context=dict(unique_records_attempted=len(access_records),
                                          total_record_attempts=sum(row.get('record_attempt_count', 1) for row in access_records),
                                          usable_public_excerpts=sum(row['status'] == 'retrieved_unverified' for row in access_records),
                                          current_run=retrieval,
                                          failures=[{key: row.get(key, '') for key in ('id', 'url', 'detail', 'fetched_at')}
                                                    for row in access_records if row['status'] != 'retrieved_unverified']), input_sha256=protected_hashes,
                  tfidf=dict(can_begin_supervised_training=verified_ready, issues=prepared['issues']),
                  distilbert=dict(can_begin_supervised_training=verified_ready and all(dependencies.values()),
                                  data_ready=verified_ready, dependencies=dependencies,
                                  note='No weights downloaded or training started. Meeting minimum counts does not establish transformer performance.'),
                  safeguards=['Silver contains AI_PROVISIONAL labels only; Gold contains existing or explicitly reviewed human annotations only.',
                              'Gold content verification is not independent fact checking. Legacy reviewer/timestamp blanks remain blank.',
                              'Chronological preparation uses first-available/project-observed timestamps, not backdated archive publication dates.',
                              'Publication quartiles are for sampling only. Verification may change category counts; quotas are not guaranteed.',
                              'Most new records were observed on one day. Labeling an archive sample alone cannot fix temporal coverage/embargo gaps.',
                              'Silver remains an unassigned weak-label pool; no AI label enters validation/test or standard training.',
                              'Reported price targets and approval odds are quoted content claims, never model predictions or calibrated probabilities.'],
                  no_model_training_performed=True)
    for name, digest in protected_hashes.items():
        if sha256(folder / name) != digest:
            raise RuntimeError('Protected multisource input changed: ' + name)
    atomic_json(output / 'readiness_report.json', report)
    lines = ['# XRP research readiness', '', f'Processed: {len(rows)}. Gold: {len(gold)}. Silver: {len(silver)}. Uncertain: {report["review_required"]}.',
             '', '| Category | Gold human | Silver provisional | New reviews suggested | Remaining quota gap |', '| --- | ---: | ---: | ---: | ---: |']
    lines += [f'| {label} | {counts(gold)[label]} | {counts(silver)[label]} | {plan[label]["suggested_new_reviews"]} | {plan[label]["remaining_if_all_confirmed"]} |' for label in LABELS]
    lines += ['', f'TF-IDF training ready: {verified_ready}. DistilBERT training ready: {report["distilbert"]["can_begin_supervised_training"]}.',
              '', '## Remaining requirements', ''] + ['- ' + issue for issue in prepared['issues']]
    lines += ['', '## Data quality', ''] + [f'- {flag}: {count}' for flag, count in sorted(flags.items())]
    lines += [f'- Largest publisher share: {report["largest_source_share"]:.1%}.', f'- Missing publication timestamps: {report["missing_publication_timestamps"]}.', '', '## Safeguards', '']
    lines += ['- ' + value for value in report['safeguards']]
    (output / 'readiness_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('processed_records', 'total_usable_distinct_headlines', 'silver_records',
                     'silver_class_distribution', 'gold_records', 'gold_class_distribution', 'review_required', 'suggested_review_sample')}, indent=2))
    return report


def save_reviews(folder, choices, reviewer):
    if not reviewer.strip():
        raise ValueError('Enter a human reviewer name before saving')
    folder = Path(folder)
    with workflow_lock(folder):
        _, current = dataset(folder / 'xrp_real_labeled.csv')
        by_source = {row['id']: row for row in current}
        _, predictions = read_csv(folder / 'research' / 'all_classifications.csv')
        by_prediction = {row['id']: row for row in predictions}
        path = folder / 'xrp_human_reviews.csv'
        ledger = load_decisions(path)
        by_decision = {row['id']: row for row in ledger}
        protected, _ = protected_humans(current, by_decision)
        if len({row['id'] for row in choices}) != len(choices):
            raise ValueError('Duplicate review IDs in submitted batch')
        additions = []
        for choice in choices:
            action = choice.get('action', '').strip().lower()
            if action in ('', 'defer'):
                continue
            row = by_prediction.get(choice['id'])
            source = by_source.get(choice['id'])
            if (not row or not source or not matching(row, source) or row['date'] != source['date']
                    or row['prediction_fingerprint'] != choice.get('prediction_fingerprint')):
                raise ValueError('Stale article or suggestion; refresh before reviewing')
            if choice['id'] in protected or choice['id'] in by_decision or source['label'].strip():
                raise ValueError('Existing human/SKIP decision is protected')
            if action == 'correct' and choice.get('human_label') in LABELS:
                key = str(LABELS.index(choice['human_label']) + 1)
            elif action in ('accept', 'reject'):
                key = 'a' if action == 'accept' else 'r'
            else:
                raise ValueError('Choose accept, correct with a category, reject, or defer')
            additions.append(decide(dict(row, suggested_label=row['predicted_label']), key, reviewer))
        if additions:
            if path.exists():
                backup(path)
            atomic_csv(path, DECISION_FIELDS, ledger + additions)
        return len(additions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'multisource')
    parser.add_argument('--target-per-class', type=int, default=25, choices=range(20, 31))
    parser.add_argument('--fetch-context', type=int, default=0, help='Maximum new article context attempts; 0 means offline')
    parser.add_argument('--max-requests', type=int, default=40)
    parser.add_argument('--retry-unavailable-context', action='store_true',
                        help='Retry only earlier extraction failures; access blocks and robots/cooldowns remain respected')
    args = parser.parse_args()
    if args.fetch_context < 0 or args.max_requests < 1:
        parser.error('Use nonnegative fetch count and positive request budget')
    with workflow_lock(args.data_dir):
        build(args.data_dir, args.target_per_class, args.fetch_context, args.max_requests, args.retry_unavailable_context)


if __name__ == '__main__':
    main()
