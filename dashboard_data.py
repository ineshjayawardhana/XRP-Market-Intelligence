"""Read-only research adapters and isolated dashboard caches/public feed polling."""
from __future__ import annotations

import json
import os
import sqlite3
import math
from contextlib import closing
from pathlib import Path

import pandas as pd

from ai_labeling import LABELS, ROOT, matching, now, read_csv
from live_data import get_live_price, get_historical_prices, load_archived_news
from market_intelligence import candle_quality, utc
from news_label_safety import atomic_json, canonical_url, normalized_headline, utc_date
from research_workflow import protected_humans
from ai_labeling import load_decisions

NEWS_COLUMNS = ['id', 'headline', 'source', 'url', 'label', 'label_status', 'score', 'score_type',
                'published_at', 'observed_at', 'outlook_eligible', 'origin', 'first_seen_at', 'collected_at',
                'classifier_backend', 'model_provenance', 'model_sha256', 'reasoning', 'publication_time_status']


def json_file(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, TypeError):
        return default


def cache_folder(root):
    return Path(os.getenv('XRP_DASHBOARD_CACHE_DIR', str(Path(root) / 'data' / 'dashboard_cache')))


def market_component(kind, root=ROOT):
    cache = cache_folder(root)
    path = cache / ('last_ticker.json' if kind == 'ticker' else 'last_candles.json')
    error = None
    try:
        if os.getenv('XRP_DASHBOARD_OFFLINE') == '1':
            raise RuntimeError('Offline mode: network requests are disabled.')
        data = get_live_price() if kind == 'ticker' else get_historical_prices(days=30)
        if kind == 'ticker':
            validate_ticker(data)
            serialized = data
        else:
            issues = candle_quality(data)
            if issues:
                raise ValueError(' '.join(issues))
            serialized = dict(candles=json.loads(data.to_json(orient='records', date_format='iso')), retrieved_at=now())
        try:
            atomic_json(path, serialized)
        except OSError:
            pass  # Read-only filesystems can still show a genuine fetched snapshot.
    except Exception as exc:
        error = str(exc)
        serialized = json_file(path)
        try:
            if kind == 'ticker':
                validate_ticker(serialized)
            elif not isinstance(serialized, dict) or not isinstance(serialized.get('candles'), list):
                raise ValueError('Invalid saved candles.')
        except (ValueError, TypeError, KeyError):
            serialized = None
        if serialized is None:
            data = None if kind == 'ticker' else pd.DataFrame()
        elif kind == 'ticker':
            data = serialized
        else:
            data = pd.DataFrame(serialized.get('candles', []))
    if kind == 'candles' and data is not None and not data.empty:
        for column in ('open_time', 'close_time'):
            if column in data:
                data[column] = pd.to_datetime(data[column], utc=True, errors='coerce')
    return dict(data=data, error=error, last_success=(serialized or {}).get('retrieved_at') if isinstance(serialized, dict) else None,
                fallback=bool(error and serialized))


def validate_ticker(data):
    if not isinstance(data, dict):
        raise ValueError('The quote is unavailable or malformed.')
    values = [float(data[key]) for key in ('price', 'high_24h', 'low_24h', 'change_24h_pct', 'volume_24h_xrp', 'volume_24h_usdt')]
    if (not all(math.isfinite(value) for value in values) or pd.isna(utc(data.get('retrieved_at')))
            or min(values[:3]) <= 0 or min(values[4:]) < 0 or not values[2] <= values[0] <= values[1]):
        raise ValueError('The quote contains missing or invalid fields.')


def _research_news(root):
    folder = Path(root) / 'data' / 'multisource'
    source = folder / 'xrp_real_labeled.csv'
    if not source.exists():
        source = Path(root) / 'data' / 'xrp_real_labeled.csv'
        folder = source.parent
    if not source.exists():
        return []
    _, rows = read_csv(source)
    decisions = {row['id']: row for row in load_decisions(folder / 'xrp_human_reviews.csv')}
    protected, _ = protected_humans(rows, decisions)
    research = folder / 'research' / 'all_classifications.csv'
    predictions_path = research if research.exists() else folder / 'xrp_auto_labeled.csv'
    predictions = {row['id']: row for row in read_csv(predictions_path)[1]} if predictions_path.exists() else {}
    result = []
    for row in rows:
        human = protected.get(row['id'])
        if human and human['status'] in ('HUMAN_SKIPPED', 'HUMAN_REJECTED'):
            result.append(dict(id=row['id'], headline=row['headline'], url=row['url'], suppressed=True))
            continue
        prediction = predictions.get(row['id'], {})
        valid = matching(row, prediction)
        label = human['label'] if human and human['status'] == 'HUMAN_VERIFIED' else prediction.get('predicted_label', 'REVIEW_REQUIRED') if valid else 'REVIEW_REQUIRED'
        score = None if human else _score(prediction.get('heuristic_confidence')) if valid else None
        result.append(dict(id=row['id'], headline=row['headline'], source=row['source'], url=row['url'], label=label,
                           label_status='Human verified' if human else 'AI provisional' if label in LABELS else 'Needs review',
                           score=score, score_type='Human content annotation' if human else 'Heuristic score · uncalibrated',
                           published_at=row.get('published_at', ''), observed_at=row.get('availability_at') or row.get('date', ''),
                           collected_at=row.get('collected_at', ''), classifier_backend='Historical research annotation',
                           model_provenance='Human annotation' if human else 'Preserved historical AI research label',
                           outlook_eligible=True, origin='Research corpus'))
    return result


def _score(value):
    try:
        score = float(value)
        return score if 0 <= score <= 1 else None
    except (ValueError, TypeError):
        return None


def load_news(root=ROOT, as_of=None):
    rows, warnings = [], []
    try:
        rows.extend(_research_news(root))
    except (OSError, ValueError, KeyError) as exc:
        warnings.append('Research archive could not be read: ' + str(exc))
    cached = json_file(cache_folder(root) / 'public_news.json', {})
    try:
        from live_news import load_rows
        rows.extend(load_rows(root))
    except (OSError, sqlite3.Error, ValueError) as exc:
        warnings.append('Persistent live news unavailable: ' + str(exc))
    rows.extend(cached.get('items', []))
    database = Path(root) / 'data' / 'news_history.sqlite3'
    if database.exists():
        try:
            with closing(sqlite3.connect(f'file:{database.resolve().as_posix()}?mode=ro', uri=True)) as con:
                archive = pd.read_sql_query('SELECT * FROM news ORDER BY observed_at DESC', con)
            for row in archive.to_dict('records'):
                rows.append(dict(id=row['id'], headline=row['headline'], source=row['source'], url='',
                                 label=row['label'], label_status='AI provisional', score=_score(float(row['confidence']) / 100),
                                 score_type='Legacy classifier score · model provenance unavailable',
                                 published_at=row.get('published_at', ''), observed_at=row['observed_at'],
                                 outlook_eligible=False, origin='Legacy live archive'))
        except (sqlite3.Error, pd.errors.DatabaseError, ValueError, KeyError) as exc:
            warnings.append('Legacy live archive unavailable: ' + str(exc))
    retained, seen = [], {}
    # Human/research rows win over later feed copies, preserving first-available time.
    for row in rows:
        keys = [('title', normalized_headline(row['headline']))]
        if row.get('url'):
            keys.append(('url', canonical_url(row['url'])))
        owner = next((seen[key] for key in keys if key in seen), -1)
        if owner != -1:
            for key in keys:
                seen[key] = owner
            if owner is not None:
                original = retained[owner]
                candidate_time = utc_date(str(row.get('collected_at', '')))
                retained_time = utc_date(str(original.get('collected_at', '')))
                if candidate_time and (not retained_time or candidate_time > retained_time):
                    original['collected_at'] = row['collected_at']
                if (canonical_url(original.get('url', '')) == canonical_url(row.get('url', ''))
                        and not original.get('published_at') and row.get('published_at')):
                    original['published_at'] = row['published_at']
                if not original.get('url') and row.get('url'):
                    original['url'] = row['url']
            continue
        for key in keys:
            seen[key] = None if row.get('suppressed') else len(retained)
        if not row.get('suppressed'):
            row.setdefault('first_seen_at', row.get('observed_at', ''))
            retained.append(row)
    frame = pd.DataFrame(retained, columns=NEWS_COLUMNS)
    for column in ('published_at', 'observed_at', 'first_seen_at', 'collected_at'):
        frame[column] = pd.to_datetime(frame[column].map(lambda value: utc_date(str(value))), utc=True, errors='coerce')
    current = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz='UTC')
    frame['_sort_publication'] = frame.published_at.where(frame.published_at <= current)
    frame = frame.sort_values(['_sort_publication', 'observed_at'], ascending=False, na_position='last').drop(columns='_sort_publication').reset_index(drop=True)
    for column in ('classifier_backend', 'model_provenance'):
        frame[column] = frame[column].fillna('Unrecorded historical provenance')
    return frame, warnings


def refresh_public_news(root=ROOT, force=False):
    """Explicit synchronous collection for CLI/tests; UI uses the background poller."""
    from live_news import refresh
    return refresh(root, force=force)


def news_poll_state(root=ROOT):
    from live_news import monitoring
    state, _ = monitoring(root)
    return dict(checked_at=state.get('checked_at'), errors=state.get('errors', []),
                added=state.get('added', 0), running=state.get('running', False))


def model_evaluations(root=ROOT):
    entries = []
    for filename, name, scope in [('metrics_tfidf.json', 'TF-IDF demo', 'Synthetic news classification'),
                                   ('metrics_tfidf_real.json', 'TF-IDF real news', 'Human-labeled chronological news classification')]:
        path = Path(root) / 'artifacts' / filename
        metrics = json_file(path)
        if not metrics:
            continue
        entries.append(dict(name=name, scope=scope, metrics=metrics,
                            evaluated_at=metrics.get('evaluated_at'),
                            sample_size=metrics.get('test_examples') or metrics.get('counts', {}).get('test'),
                            periods=metrics.get('periods_utc', {})))
    return entries


def optional_historical_news(root=ROOT):
    """Retain the legacy, user-supplied archive workflow, with explicit provenance."""
    frame = load_archived_news(Path(root) / 'data' / 'historical_news.csv', os.getenv('MODEL_BACKEND', 'tfidf'))
    if not frame.empty:
        frame['label_status'] = 'AI provisional'
        frame['origin'] = 'User-provided legacy archive; availability timestamps assumed'
    return frame
