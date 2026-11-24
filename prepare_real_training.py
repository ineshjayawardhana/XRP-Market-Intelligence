"""Prepare chronological human-only evaluation partitions without training a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

from ai_labeling import LABELS, ROOT, atomic_csv, backup, dataset, load_decisions, matching, now, read_csv, workflow_lock
from batch_news_labeling import read_json_rows
from news_label_safety import atomic_json, duplicate_groups, human_origin_allowed, sha256, utc_date

TRAIN_FIELDS = ['id', 'date', 'headline', 'source', 'url', 'label', 'label_status',
                'label_origin', 'human_reviewed_by', 'human_reviewed_at', 'timestamp_type', 'duplicate_group']
AUDIT_FIELDS = ['id', 'headline', 'reason']
MIN_HUMAN = 120
MIN_PER_CLASS = 20
EMBARGO_HOURS = 24


def record_signature(row):
    values = {key: row[key] for key in ('id', 'date', 'headline', 'url', 'label')}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode('utf-8')).hexdigest()


def class_counts(rows):
    counts = Counter(row['label'] for row in rows)
    return {label: counts.get(label, 0) for label in LABELS}


def prepare(folder, output=None, include_provisional=False):
    folder = Path(folder)
    output = Path(output) if output else folder / 'training_prepared'
    if output.resolve() == folder.resolve():
        raise ValueError('Prepared output must be a separate directory')
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob('*.csv'):
        backup(path)
    input_names = ['xrp_real_labeled.csv', 'xrp_human_reviews.csv',
                   'xrp_auto_labeled.csv', 'xrp_article_context.json']
    input_hashes = {name: sha256(folder / name) if (folder / name).exists() else None for name in input_names}
    _, source = dataset(folder / 'xrp_real_labeled.csv')
    decisions = {row['id']: row for row in load_decisions(folder / 'xrp_human_reviews.csv')}
    contexts = {row['id']: row for row in read_json_rows(folder / 'xrp_article_context.json')}
    audit, humans = [], []
    for row in source:
        label = row['label'].strip()
        origin = row.get('label_origin', '') or 'human_manual_legacy'
        reviewer = row.get('human_reviewed_by', '')
        reviewed_at = row.get('human_reviewed_at', '')
        decision = decisions.get(row['id'])
        if label and not human_origin_allowed(row):
            audit.append(dict(id=row['id'], headline=row['headline'], reason='nonhuman_or_invalid_provenance'))
            continue
        if (decision and matching(row, decision) and decision['review_status'] in ('accepted', 'corrected')
                and (not label or label == decision['human_label'])):
            label, origin = decision['human_label'], 'human_reviewed_ai'
            reviewer, reviewed_at = decision['reviewed_by'], decision['reviewed_at']
            if not utc_date(reviewed_at):
                raise ValueError('Human decision has an invalid review timestamp')
        if label not in LABELS:
            continue  # Blank, SKIP and rejected records are never training labels.
        context = contexts.get(row['id'])
        humans.append(dict(id=row['id'], date=row.get('date', ''), headline=row['headline'],
                           source=row.get('source', ''), url=row['url'], label=label,
                           label_status='HUMAN_VERIFIED', label_origin=origin,
                           human_reviewed_by=reviewer, human_reviewed_at=reviewed_at,
                           timestamp_type=row.get('timestamp_type', 'UNKNOWN'), duplicate_group='',
                           context_excerpt=context.get('excerpt', '') if context and matching(row, context) else ''))
    auto_path = folder / 'xrp_auto_labeled.csv'
    provisional = []
    by_source = {row['id']: row for row in source}
    if auto_path.exists():
        for row in read_csv(auto_path)[1]:
            current = by_source.get(row['id'])
            if (not current or not matching(row, current) or current['label'].strip()
                    or row.get('timestamp') != current.get('date')):
                continue
            # A human decision takes precedence over provisional copies, including rejection.
            if row['id'] in decisions:
                continue
            if (row.get('label_status') != 'AI_PROVISIONAL' or row.get('predicted_label') not in LABELS
                    or row.get('needs_manual_review') != 'False'):
                continue
            provisional.append(dict(id=row['id'], date=row['timestamp'], headline=row['headline'],
                                    source=row['source'], url=row['url'], label=row['predicted_label'],
                                    label_status='AI_PROVISIONAL', label_origin='ai_provisional',
                                    human_reviewed_by='', human_reviewed_at='',
                                    timestamp_type=row.get('timestamp_type', 'UNKNOWN'), duplicate_group='',
                                    context_excerpt=row.get('context_excerpt', '')))
    valid = []
    for row in humans + provisional:
        timestamp = utc_date(row['date'])
        reason = ''
        if not timestamp:
            reason = 'invalid_or_timezone_missing_timestamp'
        elif timestamp > utc_date(now()):
            reason = 'future_timestamp'
        elif 'synthetic' in row['source'].lower():
            reason = 'synthetic_source'
        elif len(row['headline'].strip()) < 12:
            reason = 'empty_or_too_short_headline'
        if reason:
            audit.append(dict(id=row['id'], headline=row['headline'], reason=reason))
        else:
            row['date'] = timestamp.isoformat()
            valid.append(row)
    clean_human, clean_ai = [], []
    for group in duplicate_groups(valid):
        human_group = [row for row in group if row['label_status'] == 'HUMAN_VERIFIED']
        group_id = 'group_' + min(row['id'] for row in group)
        labels = {row['label'] for row in human_group or group}
        if len(labels) > 1:
            audit.extend(dict(id=row['id'], headline=row['headline'], reason='conflicting_duplicate_labels') for row in group)
            continue
        # Prefer a human annotation; AI can never displace a human label.
        chosen = min(human_group or group, key=lambda row: (utc_date(row['date']), row['id']))
        chosen['duplicate_group'] = group_id
        (clean_human if human_group else clean_ai).append(chosen)
        audit.extend(dict(id=row['id'], headline=row['headline'], reason='duplicate_of:' + chosen['id'])
                     for row in group if row is not chosen)
    clean_human.sort(key=lambda row: (utc_date(row['date']), row['id']))
    issues = []
    reservation_path = output / 'test_reservation.json'
    reservation = json.loads(reservation_path.read_text(encoding='utf-8')) if reservation_path.exists() else None
    train, validation, test = [], [], []
    val_start = test_start = None
    if reservation:
        frozen = reservation['record_signatures']
        by_human = {row['id']: row for row in clean_human}
        if any(key not in by_human or record_signature(by_human[key]) != signature
               for key, signature in frozen.items()):
            issues.append('Frozen test records changed, disappeared or became conflicting duplicates. Resolve before training.')
        else:
            test = [by_human[key] for key in frozen]
            test.sort(key=lambda row: utc_date(row['date']))
            test_start = min(utc_date(row['date']) for row in test)
            earlier = [row for row in clean_human if utc_date(row['date']) < test_start]
            cut = int(len(earlier) * 70 / 85)
            train, validation = earlier[:cut], earlier[cut:]
            audit.extend(dict(id=row['id'], headline=row['headline'], reason='on_or_after_frozen_test_start')
                         for row in clean_human if row['id'] not in frozen and utc_date(row['date']) >= test_start)
    elif len(clean_human) >= 3:
        cut1 = max(1, int(len(clean_human) * .70))
        cut2 = min(len(clean_human) - 1, max(cut1 + 1, int(len(clean_human) * .85)))
        train, validation, test = clean_human[:cut1], clean_human[cut1:cut2], clean_human[cut2:]
        test_start = min(utc_date(row['date']) for row in test)
    else:
        test = clean_human[:]  # Reserve available humans; never invent a training split.
        issues.append('At least three chronologically distinct human records are needed even to propose partitions.')
    embargo = timedelta(hours=EMBARGO_HOURS)
    if validation and test_start:
        val_start = min(utc_date(row['date']) for row in validation)
        purged_train = [row for row in train if utc_date(row['date']) < val_start - embargo]
        purged_val = [row for row in validation if utc_date(row['date']) < test_start - embargo]
        audit.extend(dict(id=row['id'], headline=row['headline'], reason='24_hour_boundary_embargo')
                     for row in train + validation if row not in purged_train + purged_val)
        train, validation = purged_train, purged_val
    eligible_ai = [row for row in clean_ai if val_start and utc_date(row['date']) < val_start - embargo]
    weak_train = eligible_ai if include_provisional else []
    used_humans = train + validation + test
    counts = class_counts(used_humans)
    if len(used_humans) < MIN_HUMAN:
        issues.append(f'Need >= {MIN_HUMAN} human records after duplicate checks and embargo; have {len(used_humans)}.')
    for label, count in counts.items():
        if count < MIN_PER_CLASS:
            issues.append(f'{label}: need >= {MIN_PER_CLASS} human labels; have {count}.')
    for name, rows in [('train', train), ('validation', validation), ('test', test)]:
        missing = [label for label, count in class_counts(rows).items() if count == 0]
        if len(rows) < 10 or missing:
            issues.append(f'{name}: need >=10 human records and all four classes; have {len(rows)}, missing {missing}.')
    partitions = {'train_human.csv': train, 'validation_human.csv': validation,
                  'test_human.csv': test, 'train_ai_provisional.csv': weak_train}
    # Invariants apply even when not ready: test/validation must always be human-only.
    combined = list(partitions.items())
    for index, (name, rows) in enumerate(combined):
        if name != 'train_ai_provisional.csv' and any(row['label_status'] != 'HUMAN_VERIFIED' for row in rows):
            raise AssertionError('Provisional label in a human-only partition')
        groups = {row['duplicate_group'] for row in rows}
        for other_name, other in combined[index + 1:]:
            if groups & {row['duplicate_group'] for row in other}:
                raise AssertionError(f'Duplicate group crosses {name} and {other_name}')
        atomic_csv(output / name, TRAIN_FIELDS, [{key: row[key] for key in TRAIN_FIELDS} for row in rows])
    atomic_csv(output / 'excluded_records.csv', AUDIT_FIELDS, audit)
    for name, digest in input_hashes.items():
        current_digest = sha256(folder / name) if (folder / name).exists() else None
        if current_digest != digest:
            raise RuntimeError(f'Input changed during preparation: {name}')
    if not issues and not reservation:
        reservation = dict(created_at=now(), test_start=test_start.isoformat(),
                           record_signatures={row['id']: record_signature(row) for row in test})
        atomic_json(reservation_path, reservation)
    manifest = dict(created_at=now(), ready_for_training=not issues, issues=issues,
                    human_annotation_count=len(humans), unique_valid_human_count=len(clean_human),
                    human_class_counts_before_embargo=class_counts(clean_human),
                    human_class_counts_after_embargo=counts,
                    partitions={name: dict(count=len(rows), class_counts=class_counts(rows)) for name, rows in partitions.items()},
                    excluded_records=len(audit), eligible_provisional_train_count=len(eligible_ai),
                    provisional_training_enabled=include_provisional, embargo_hours=EMBARGO_HOURS,
                    validation_start=val_start.isoformat() if val_start else None,
                    test_start=test_start.isoformat() if test_start else None,
                    test_reservation_frozen=bool(reservation),
                    test_reservation_status='frozen' if reservation else 'tentative_until_sufficient_human_data',
                    input_sha256=input_hashes,
                    output_sha256={name: sha256(output / name) for name in partitions},
                    evaluation_ground_truth='HUMAN_VERIFIED content annotations only; not article factual verification',
                    features='headline only; no future prices, article excerpts, AI confidence or label metadata',
                    timestamp_caution='GDELT_FIRST_SEEN is availability metadata, not verified publication time.',
                    duplicate_caution='Detected syndicated/near-duplicate groups are excluded across partitions; paraphrase detection is heuristic.',
                    no_model_training_performed=True)
    atomic_json(output / 'manifest.json', manifest)
    print(json.dumps({key: manifest[key] for key in ('ready_for_training', 'human_annotation_count',
                     'unique_valid_human_count', 'human_class_counts_before_embargo', 'partitions', 'issues')}, indent=2))
    return manifest


def load_verified_partitions(folder, output):
    """Fail closed on insufficient data or any changed input/output after preparation."""
    import pandas as pd
    folder, output = Path(folder), Path(output)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest['ready_for_training']:
        raise ValueError('Training blocked: ' + ' '.join(manifest['issues']))
    for name, digest in manifest['input_sha256'].items():
        current_digest = sha256(folder / name) if (folder / name).exists() else None
        if current_digest != digest:
            raise ValueError('Prepared inputs changed; run preparation again.')
    frames = []
    for name in ('train_human.csv', 'validation_human.csv', 'test_human.csv'):
        if sha256(output / name) != manifest['output_sha256'][name]:
            raise ValueError('Prepared partition changed; run preparation again.')
        frame = pd.read_csv(output / name, dtype=str).fillna('')
        if not frame['label_status'].eq('HUMAN_VERIFIED').all():
            raise ValueError('Evaluation and standard training require human annotations.')
        frame['date'] = pd.to_datetime(frame['date'], utc=True, errors='raise')
        frames.append(frame)
    return tuple(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--include-provisional-training', action='store_true',
                        help='Export a separate optional weak-label training file; never evaluation ground truth')
    args = parser.parse_args()
    with workflow_lock(args.data_dir):
        prepare(args.data_dir, args.output_dir, args.include_provisional_training)


if __name__ == '__main__':
    main()
