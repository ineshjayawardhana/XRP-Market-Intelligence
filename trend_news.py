"""Read-only, point-in-time news eligibility for 24-hour trend research.

Publication and GDELT first-seen dates do not prove that today's excerpt or label
was available in the past. Explicit project observation/context retrieval AND
label availability evidence are required. Gold and Silver counts stay separate.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ai_labeling import LABELS, load_decisions, matching
from news_label_safety import canonical_url, normalized_headline, utc_date
from research_workflow import protected_humans

MIN_NEWS_DAYS = 60
MIN_NEWS_SPAN_DAYS = 90
MIN_COVERED_PREDICTIONS_PER_THIRD = 10


def _read(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _time(value):
    parsed = utc_date(value)
    return pd.Timestamp(parsed) if parsed else None


def _now():
    return pd.Timestamp.now(tz='UTC')


def _load(root):
    folder = Path(root) / 'data' / 'multisource'
    research = folder / 'research'
    source = _read(folder / 'xrp_real_labeled.csv')
    indexed = {row['id']: row for row in source}
    decisions = {row['id']: row for row in load_decisions(folder / 'xrp_human_reviews.csv')}
    protected, human_conflicts = protected_humans(source, decisions)
    accepted, rejected = [], Counter()
    counts, distributions = {}, {}
    seen = set()
    for role, filename in [('gold', 'gold_dataset.csv'), ('silver', 'silver_dataset.csv')]:
        rows = _read(research / filename)
        counts[role] = len(rows)
        distributions[role] = dict(Counter(row.get('label', '') for row in rows))
        for row in rows:
            identifier = row.get('id')
            original = indexed.get(identifier)
            if not original or not matching(original, row):
                rejected['missing_or_mismatched_source_identity'] += 1
                continue
            human = protected.get(identifier)
            if identifier in human_conflicts:
                rejected['conflicting_human_annotations'] += 1
                continue
            if human and human['status'] in ('HUMAN_SKIPPED', 'HUMAN_REJECTED'):
                rejected['protected_human_exclusion'] += 1
                continue
            label = row.get('label', '')
            if (label not in LABELS or row.get('eligible_content', 'True') != 'True'
                    or row.get('canonical_record', 'True') != 'True'):
                rejected['ineligible_or_noncanonical_record'] += 1
                continue
            if role == 'gold' and (not human or human['status'] != 'HUMAN_VERIFIED' or human['label'] != label):
                rejected['unsubstantiated_human_status'] += 1
                continue
            if role == 'silver' and (human or row.get('label_status') != 'AI_PROVISIONAL'):
                rejected['provenance_conflict'] += 1
                continue
            keys = [('headline', normalized_headline(row.get('headline', '')))]
            if row.get('url'):
                keys.append(('url', canonical_url(row['url'])))
            if any(key in seen for key in keys):
                rejected['duplicate_article'] += 1
                continue
            seen.update(keys)
            # Current review exports do not carry an AI classification timestamp.
            # A run's file mtime or publication date must never replace that fact.
            label_at = (_time(human.get('reviewed_at', '')) if role == 'gold' else
                        _time(row.get('label_available_at') or row.get('classified_at') or row.get('classification_generated_at')))
            observed = _time(original.get('collected_at', ''))
            context = _time(row.get('context_retrieved_at', ''))
            known_content = [instant for instant in (observed, context) if instant is not None]
            invalid = False
            if label_at is None:
                rejected['missing_explicit_label_availability'] += 1
                invalid = True
            if not known_content:
                rejected['missing_project_content_observation'] += 1
                invalid = True
            if invalid:
                continue
            available = max(known_content + [label_at])
            if available > _now():
                rejected['future_feature_availability'] += 1
                continue
            accepted.append(dict(id=identifier, role=role, label=label,
                                 feature_available_at=available, source=row.get('source', '')))
    return accepted, rejected, counts, distributions


def _prediction_times(values):
    if values is None:
        return pd.DatetimeIndex([], tz='UTC')
    # Refuse timezone-naive inputs; model code must use authoritative UTC times.
    raw = list(values)
    parsed = [_time(value.isoformat() if hasattr(value, 'isoformat') else value) for value in raw]
    if any(value is None for value in parsed):
        raise ValueError('Prediction timestamps must have explicit timezone offsets.')
    return pd.DatetimeIndex(parsed)


def _coverage(rows, times):
    if len(times) == 0:
        return None
    events = np.sort(np.array([row['feature_available_at'].value for row in rows], dtype=np.int64))
    stamps = times.asi8
    left = np.searchsorted(events, stamps - pd.Timedelta(hours=24).value, side='right')
    right = np.searchsorted(events, stamps, side='right')
    covered = right > left
    order = np.argsort(stamps)
    parts = np.array_split(order, 3)
    return dict(prediction_rows=len(times), covered_rows=int(covered.sum()),
                covered_fraction=float(covered.mean()),
                covered_rows_by_chronological_third=[int(covered[part].sum()) for part in parts])


def audit_news(root, prediction_times=None):
    """Return JSON-safe readiness and provenance facts; never change source files.

    News-enhanced comparisons require 60 distinct eligible feature days across
    at least 90 days, >=5% prediction coverage, and >=10 covered prediction rows
    in each chronological third when prediction timestamps are supplied. These
    conservative policy minima establish coverage, not predictive quality.
    """
    try:
        rows, excluded, counts, distributions = _load(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(ready=False, news_enhanced_comparison_available=False,
                    reasons=['News provenance audit failed: ' + str(exc)], eligible_records=0)
    dates = sorted({row['feature_available_at'].floor('D') for row in rows})
    span = (dates[-1] - dates[0]).days if dates else 0
    coverage = _coverage(rows, _prediction_times(prediction_times))
    reasons = []
    if len(dates) < MIN_NEWS_DAYS:
        reasons.append(f'Need at least {MIN_NEWS_DAYS} distinct point-in-time feature days; have {len(dates)}.')
    if span < MIN_NEWS_SPAN_DAYS:
        reasons.append(f'Need at least {MIN_NEWS_SPAN_DAYS} days of eligible news span; have {span}.')
    if coverage and (coverage['covered_fraction'] < .05 or any(n < MIN_COVERED_PREDICTIONS_PER_THIRD for n in coverage['covered_rows_by_chronological_third'])):
        reasons.append('Eligible news does not sufficiently cover each chronological training/evaluation segment.')
    report = dict(ready=not reasons, news_enhanced_comparison_available=not reasons,
                  reasons=reasons, raw_counts=counts, raw_class_distribution=distributions,
                  eligible_records=len(rows), eligible_by_role=dict(Counter(row['role'] for row in rows)),
                  distinct_feature_days=len(dates), feature_span_days=span,
                  feature_start=min((row['feature_available_at'] for row in rows), default=None),
                  feature_end=max((row['feature_available_at'] for row in rows), default=None),
                  exclusion_reasons=dict(excluded), prediction_coverage=coverage,
                  policy=dict(minimum_distinct_days=MIN_NEWS_DAYS, minimum_span_days=MIN_NEWS_SPAN_DAYS,
                              minimum_prediction_coverage=.05, minimum_covered_rows_per_third=MIN_COVERED_PREDICTIONS_PER_THIRD),
                  limitations=['Counts are content categories, not verified facts or price-direction ground truth.',
                               'Gold and Silver remain separate; no future annotation or later excerpt is backdated.',
                               'Publication, first-seen, sampling and filesystem times are never substituted for feature availability.',
                               'News eligibility is not a claim of news-enhanced model performance.'])
    for key in ('feature_start', 'feature_end'):
        report[key] = report[key].isoformat() if report[key] is not None else None
    return report


def news_features_asof(root, prediction_times):
    """Causal 24-hour content-count features; refuse inadequate coverage.

    The right-closed window is (prediction_time - 24h, prediction_time].
    Only explicitly available labels and acquired content enter the window.
    """
    times = _prediction_times(prediction_times)
    report = audit_news(root, times)
    if not report['ready']:
        raise ValueError('News-enhanced research is not ready: ' + ' '.join(report['reasons']))
    rows, _, _, _ = _load(root)
    result = pd.DataFrame({'prediction_time': times})
    for role in ('gold', 'silver'):
        for label in LABELS:
            events = np.sort(np.array([row['feature_available_at'].value for row in rows if row['role'] == role and row['label'] == label], dtype=np.int64))
            upper = np.searchsorted(events, times.asi8, side='right')
            lower = np.searchsorted(events, times.asi8 - pd.Timedelta(hours=24).value, side='right')
            result[f'{role}_{label}_24h'] = upper - lower
    return result


if __name__ == '__main__':
    print(json.dumps(audit_news(Path(__file__).resolve().parent), indent=2))
