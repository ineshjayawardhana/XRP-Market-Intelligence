"""Batch provisional classification and optional multi-record human review.

Uses saved AI assessments plus available excerpts, or conservative evidence rules
for new articles. No external LLM call is made or implied.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from ai_labeling import (DECISION_FIELDS, LABELS, REVIEW, atomic_csv, backup, dataset,
                         decide, fetch_one, load_decisions, matching, now, read_csv)
from news_label_safety import atomic_json, duplicate_groups, sha256, utc_date

AUTO_FIELDS = ['id', 'headline', 'predicted_label', 'heuristic_confidence', 'reasoning',
               'source', 'url', 'timestamp', 'timestamp_type', 'classified_at',
               'label_status', 'verification_status', 'article_verified',
               'context_status', 'context_retrieved_at', 'context_excerpt',
               'context_sha256', 'classification_method', 'needs_manual_review',
               'review_flags', 'protected_human_label', 'existing_review_status', 'duplicate_group']
REVIEW_FIELDS = ['id', 'headline', 'url', 'predicted_label', 'review_flags', 'action', 'human_label']


def read_json_rows(path):
    if not Path(path).exists():
        return []
    rows = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list) or len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Invalid JSON rows or duplicate IDs: {path}')
    return rows


def rules(headline, excerpt):
    """Return an explainable category only for narrow, explicit content cues."""
    title, text = headline.casefold(), excerpt.casefold()
    if not re.search(r'\b(xrp|xrpl|rlusd)\b', title):
        return REVIEW, .2, 'No clear XRP/XRPL/RLUSD subject in the headline.'
    if re.search(r'presale|alphapepe|remittix|ftmining|cloud mining|daily passive income', title):
        return REVIEW, .3, 'Unrelated token promotion or income advertising is mixed with XRP.'
    adoption = bool(re.search(r'programmable wallet tools|partners with.*integrat|adds.*shares.*ledger', title))
    if 'tracked transactions' in title and re.search(r'settlements|registered merchants', text):
        adoption = True
    if adoption and re.search(r'\b(but|while|threat|warning)\b|price impact|price prediction', title):
        return REVIEW, .4, 'Concrete product language is mixed with price or risk framing.'
    if adoption:
        if not re.search(r'wallet|integrat|settle|payment|developer', text) or re.search(r'not yet (launched|available|integrated)', text):
            return REVIEW, .4, 'The excerpt does not clearly support an available integration or operational product.'
        return 'Real_Adoption', .85, 'Named integration, developer tooling or operational payment activity is described.'
    if re.search(r'\b(hack|exploit|outage|lawsuit|fined)\b|bug fix moved', title):
        if (re.search(r'\b(no|not|denies|rumou?r)\b', title)
                or re.search(r'\b(no|not an?) (hack|exploit|outage|lawsuit)\b', text)
                or not re.search(r'hack|exploit|outage|lawsuit|fined|bug|validator', text)):
            return REVIEW, .4, 'Risk-event wording includes a denial or uncertainty.'
        return 'Negative_FUD', .8, 'The central subject is a reported technical, legal or safety problem.'
    if re.search(r'\bi predict\b|\bmy price target\b', text) or re.search(r'guaranteed|100x|to the moon', title):
        return 'Hype_Speculation', .78, 'An explicit personal price forecast or assured-return claim is central.'
    if re.search(r'trading|moving average|day average|margin longs|exchange.*(inflows|outflows)|payment volume|support|resistance', title):
        if not re.search(r'trad|price|volume|average|support|resistance|inflows|outflows|balance|transfer', text):
            return REVIEW, .4, 'The available excerpt does not support the headline market measurements.'
        return 'Neutral_Market_News', .8, 'The subject is observed market/ledger measurements or conditional technical analysis.'
    return REVIEW, .4, 'No single category has sufficiently explicit evidence under the conservative rubric.'


def classify(row, assessment, context, suggestion=None, min_confidence=.75):
    usable = bool(context and matching(row, context) and context.get('status') == 'retrieved_unverified'
                  and len(context.get('excerpt', '').split()) >= 20
                  and re.search(r'\b(XRP|XRPL|RLUSD)\b', context['excerpt'], re.I))
    excerpt = context.get('excerpt', '') if usable else ''
    rule_label, rule_score, rule_reason = rules(row['headline'], excerpt)
    flags = []
    if assessment and matching(row, assessment):
        label = assessment['suggested_label']
        score = float(assessment['confidence_estimate'])
        reason = assessment['reasoning']
        method = 'cached_ai_assessment_with_excerpt_checks_v1'
        if label not in (*LABELS, REVIEW) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Invalid cached assessment: {row["id"]}')
        if usable and label in LABELS and rule_label in LABELS and rule_label != label:
            flags.append('evidence_rule_conflict')
            reason += f' Batch check conflicts with this assessment: {rule_label}. {rule_reason}'
            label, score = REVIEW, min(score, .49)
    else:
        label, score, reason = rule_label, rule_score, rule_reason
        method = 'conservative_evidence_rules_v1'
    if suggestion and matching(row, suggestion) and assessment and matching(row, assessment):
        if suggestion['suggested_label'] != assessment['suggested_label']:
            flags.append('saved_assessment_conflict')
            reason += ' Saved proposal and suggestion disagree.'
            label, score = REVIEW, min(score, .49)
    if not usable:
        flags.append('article_context_unavailable')
        reason += ' No matching usable article excerpt is available. Headline-only evidence is insufficient for automatic assignment.'
        label, score = REVIEW, min(score, .49)
    if score < min_confidence and label in LABELS:
        flags.append('low_heuristic_confidence')
        reason += f' Score is below the {min_confidence:g} heuristic threshold.'
        label = REVIEW
    if label == REVIEW:
        flags.append('uncertain_category')
    reason += ' AI_PROVISIONAL: content category only; article facts have not been independently verified.'
    return label, score, reason, method, flags, excerpt


def snapshot_csvs(folder):
    return {path.name: sha256(path) for path in Path(folder).glob('*.csv') if path.is_file()}


def batch(folder, refresh_missing=False, min_confidence=.75):
    folder = Path(folder)
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError('Heuristic threshold must be between 0 and 1')
    # Snapshot every existing CSV before any generated output is replaced.
    before = snapshot_csvs(folder)
    for name in before:
        backup(folder / name)
    _, source_rows = dataset(folder / 'xrp_real_labeled.csv')
    suggestions_path = folder / 'xrp_ai_suggestions.csv'
    suggestions = read_csv(suggestions_path)[1] if suggestions_path.exists() else []
    by_suggestion = {row['id']: row for row in suggestions}
    assessments = {row['id']: row for row in read_json_rows(folder / 'xrp_ai_proposals.json')}
    contexts = {row['id']: row for row in read_json_rows(folder / 'xrp_article_context.json')}
    decisions = {row['id']: row for row in load_decisions(folder / 'xrp_human_reviews.csv')}
    by_source = {row['id']: row for row in source_rows}
    if set(by_suggestion) - set(by_source):
        raise ValueError('A suggestion has no source dataset record; resolve the stale ID first.')
    candidates = [row for row in source_rows if not row['label'].strip() or row['id'] in by_suggestion]
    if refresh_missing:
        changed = False
        from concurrent.futures import ThreadPoolExecutor
        missing = [row for row in candidates if not matching(row, contexts.get(row['id'], {}))
                   or contexts[row['id']].get('status') != 'retrieved_unverified']
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(fetch_one, missing):
                contexts[result['id']] = result
                changed = True
        if changed:
            path = folder / 'xrp_article_context.json'
            if path.exists():
                backup(path)
            atomic_json(path, list(contexts.values()))
    output = []
    stamp = now()
    for row in candidates:
        context = contexts.get(row['id'])
        label, score, reason, method, flags, excerpt = classify(
            row, assessments.get(row['id']), context, by_suggestion.get(row['id']), min_confidence)
        decision = decisions.get(row['id'])
        valid_decision = decision if decision and matching(row, decision) else None
        protected = row['label'].strip()
        if valid_decision and not protected:
            protected = valid_decision['human_label'] or 'REJECTED'
        if protected and label in LABELS and label != protected:
            flags.append('conflict_with_protected_decision')
            reason += f' Protected human decision ({protected}) is unchanged.'
            label, score = REVIEW, min(score, .49)
        if protected in ('SKIP', 'REJECTED'):
            flags.append('protected_skip_or_rejection')
            label, score = REVIEW, min(score, .49)
        if not utc_date(row.get('date', '')):
            flags.append('invalid_or_timezone_missing_timestamp')
        context_matches = context and matching(row, context)
        output.append(dict(
            id=row['id'], headline=row['headline'], predicted_label=label,
            heuristic_confidence=round(score, 3), reasoning=reason, source=row.get('source', ''),
            url=row['url'], timestamp=row.get('date', ''), timestamp_type=row.get('timestamp_type', 'UNKNOWN'),
            classified_at=stamp, label_status='AI_PROVISIONAL', verification_status='NOT_FACT_CHECKED',
            article_verified='False', context_status='excerpt_available_unverified' if excerpt else 'unavailable_or_insufficient',
            context_retrieved_at=context.get('fetched_at', '') if context_matches else '',
            context_excerpt=excerpt, context_sha256=hashlib.sha256(excerpt.encode('utf-8')).hexdigest() if excerpt else '',
            classification_method=method, needs_manual_review=str(bool(flags)),
            review_flags=';'.join(flags), protected_human_label=protected,
            existing_review_status=valid_decision['review_status'] if valid_decision else '', duplicate_group=''))
    for group in duplicate_groups(output):
        if len(group) < 2:
            continue
        group_id = 'duplicate_' + min(row['id'] for row in group)
        labels = {row['predicted_label'] for row in group if row['predicted_label'] in LABELS}
        for row in group:
            row['duplicate_group'] = group_id
            row['review_flags'] = ';'.join(filter(None, [row['review_flags'], 'possible_syndicated_duplicate']))
            row['needs_manual_review'] = 'True'
            if len(labels) > 1:
                row['predicted_label'] = REVIEW
                row['review_flags'] += ';duplicate_label_conflict'
    # Never mutate source suggestions, human ledger or original dataset.
    for name in ('xrp_real_labeled.csv', 'xrp_ai_suggestions.csv', 'xrp_human_reviews.csv'):
        if name in before and sha256(folder / name) != before[name]:
            raise RuntimeError(f'Input changed during classification: {name}')
    auto_path = folder / 'xrp_auto_labeled.csv'
    atomic_csv(auto_path, AUTO_FIELDS, output)
    flagged = [row for row in output if row['needs_manual_review'] == 'True']
    atomic_csv(folder / 'xrp_batch_review.csv', AUTO_FIELDS, flagged)
    # Decisions are a blank template, not synthetic human approvals.
    template = folder / 'xrp_batch_decisions_template.csv'
    atomic_csv(template, REVIEW_FIELDS, [dict(id=row['id'], headline=row['headline'], url=row['url'],
               predicted_label=row['predicted_label'], review_flags=row['review_flags'], action='', human_label='')
               for row in output if not row['protected_human_label']])
    counts = Counter(row['predicted_label'] for row in output)
    summary = dict(created_at=stamp, total_records=len(output),
                   class_counts={label: counts.get(label, 0) for label in (*LABELS, REVIEW)},
                   automatically_classified=sum(counts.get(label, 0) for label in LABELS),
                   uncertain=counts.get(REVIEW, 0), flagged_for_review=len(flagged),
                   records_using_excerpts=sum(bool(row['context_excerpt']) for row in output),
                   label_status='AI_PROVISIONAL', calibrated_probabilities=False,
                   input_sha256=before, output_sha256=sha256(auto_path),
                   evidence_input_sha256={name: sha256(folder / name) for name in
                                          ('xrp_article_context.json', 'xrp_ai_proposals.json') if (folder / name).exists()},
                   method='Saved AI assessments plus excerpt checks; conservative rules for new items. No live LLM call.',
                   protected_input_files_unchanged=True)
    atomic_json(folder / 'xrp_batch_summary.json', summary)
    report = ['# XRP provisional batch report', '',
              f'Processed {len(output)} records. Automatically assigned: {summary["automatically_classified"]}. '
              f'Uncertain: {summary["uncertain"]}. Flagged: {len(flagged)}.', '',
              'All automatic labels are AI_PROVISIONAL. Scores are heuristic, not calibrated probabilities.',
              'Article excerpts are context, not factual verification. Human decisions and SKIP rows are preserved.', '',
              '| Category | Count |', '| --- | ---: |']
    report += [f'| {label} | {summary["class_counts"][label]} |' for label in (*LABELS, REVIEW)]
    report += ['', '## Flagged records', '', '| ID | Headline | Flags |', '| --- | --- | --- |']
    report += [f'| {row["id"]} | {row["headline"].replace("|", "/")} | {row["review_flags"]} |' for row in flagged]
    report += ['', 'Per-record reasoning and source URLs: xrp_auto_labeled.csv. Flagged details: xrp_batch_review.csv.',
               'Use a copy of xrp_batch_decisions_template.csv to accept/correct/reject multiple rows together.',
               'No human reviews or model training were performed by this batch.', '']
    (folder / 'xrp_batch_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('total_records', 'class_counts', 'automatically_classified',
                                                   'uncertain', 'flagged_for_review', 'records_using_excerpts')}, indent=2))
    return summary


def batch_review(folder, decisions_file, reviewer):
    """Validate the entire submitted batch before a single authoritative ledger write."""
    if not reviewer.strip():
        raise ValueError('A named human reviewer is required')
    folder = Path(folder)
    _, choices = read_csv(decisions_file)
    _, auto = read_csv(folder / 'xrp_auto_labeled.csv')
    _, source = dataset(folder / 'xrp_real_labeled.csv')
    by_auto = {row['id']: row for row in auto}
    by_source = {row['id']: row for row in source}
    ledger_path = folder / 'xrp_human_reviews.csv'
    ledger = load_decisions(ledger_path)
    reviewed_ids = {row['id'] for row in ledger}
    additions = []
    for choice in choices:
        action = choice.get('action', '').strip().lower()
        if not action or action == 'defer':
            continue
        row = by_auto.get(choice['id'])
        current = by_source.get(choice['id'])
        if not row or not current or not matching(row, choice) or not matching(row, current):
            raise ValueError(f'Stale or missing batch record: {choice["id"]}')
        if choice['id'] in reviewed_ids or current['label'].strip():
            raise ValueError(f'Human decision/SKIP already protected: {choice["id"]}')
        actions = {'accept': 'a', 'reject': 'r'}
        if action == 'correct' and choice.get('human_label') in LABELS:
            key = str(LABELS.index(choice['human_label']) + 1)
        elif action in actions:
            key = actions[action]
        else:
            raise ValueError(f'Invalid action or human label: {choice["id"]}')
        if choice.get('predicted_label') != row['predicted_label']:
            raise ValueError(f'Prediction changed since template export: {choice["id"]}')
        additions.append(decide(dict(row, suggested_label=row['predicted_label']), key, reviewer))
    if additions:
        if ledger_path.exists():
            backup(ledger_path)
        backup(folder / 'xrp_real_labeled.csv')
        atomic_csv(ledger_path, DECISION_FIELDS, ledger + additions)
    print(f'Saved {len(additions)} human decisions together. Original dataset unchanged.')
    return len(additions)
