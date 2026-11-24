"""Persistent, incremental public news collection and a process-lifetime poller.

Only data/live_news is written. Research CSVs, original SQLite, labels and models
are read-only. No scraping, paywall bypass, model fitting or invented timestamps.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from ai_labeling import LABELS, now
from collect_multisource_news import PublicClient, Store, parse_feed
from live_news_sources import SOURCES
from news_label_safety import canonical_url, normalized_headline, sha256, utc_date

INTERVAL = 300


def folder(root):
    return Path(os.getenv('XRP_LIVE_NEWS_DIR', str(Path(root) / 'data' / 'live_news')))


def as_time(value):
    return pd.Timestamp(utc_date(value)) if utc_date(value) else None


class NewsStore:
    def __init__(self, root):
        self.root, self.directory = Path(root), folder(root)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'news.sqlite3'
        with self.connect() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY, url_key TEXT UNIQUE, title_key TEXT UNIQUE, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS state (id TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY, article_id TEXT, source_id TEXT, collected_at TEXT, duplicate INTEGER);
            ''')
        self.import_legacy()

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=20)
        try:
            con.execute('PRAGMA busy_timeout=20000')
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def import_legacy(self):
        # A one-time append-only migration of the old cache; the original remains.
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            if con.execute("SELECT 1 FROM state WHERE id='legacy_import'").fetchone():
                return
            cache = Path(os.getenv('XRP_DASHBOARD_CACHE_DIR', str(self.root / 'data' / 'dashboard_cache')))
            try:
                rows = json.loads((cache / 'public_news.json').read_text(encoding='utf-8')).get('items', [])
            except (OSError, ValueError, TypeError):
                rows = []
            for row in rows:
                row = dict(row)
                row.setdefault('first_seen_at', row.get('observed_at', ''))
                row.setdefault('collected_at', row.get('observed_at', ''))
                row.setdefault('classifier_backend', 'legacy_feed_heuristic')
                row.setdefault('model_provenance', 'Preserved legacy feed classification; exact version unrecorded')
                row.setdefault('article_verified', False)
                self._insert(con, row, 'legacy_cache', row.get('collected_at', ''))
            con.execute('INSERT INTO state VALUES (?,?)', ('legacy_import', json.dumps(dict(items=len(rows), at=now()))))

    def _insert(self, con, row, provider, collected):
        url_key = canonical_url(row.get('url', '')) or None
        title_key = normalized_headline(row['headline']) or None
        old = con.execute('SELECT id,record,url_key FROM articles WHERE url_key=? OR title_key=?', (url_key, title_key)).fetchone()
        if old:
            original = json.loads(old[1])
            original['collected_at'] = collected or original.get('collected_at', '')
            original['last_seen_at'] = collected or original.get('last_seen_at', '')
            # Same canonical article only: another publisher's syndication date
            # cannot supply the original's missing publication timestamp.
            if old[2] == url_key and not original.get('published_at') and row.get('published_at'):
                for field in ('published_at', 'published_at_raw', 'publication_time_status'):
                    original[field] = row.get(field, '')
            con.execute('UPDATE articles SET record=? WHERE id=?', (json.dumps(original), old[0]))
            identifier, added = old[0], False
        else:
            identifier = row.get('id') or 'LIVE-' + hashlib.sha256((url_key or title_key).encode()).hexdigest()[:24]
            row = dict(row, id=identifier)
            con.execute('INSERT INTO articles VALUES (?,?,?,?)', (identifier, url_key, title_key, json.dumps(row)))
            added = True
        con.execute('INSERT INTO discoveries(article_id,source_id,collected_at,duplicate) VALUES (?,?,?,?)',
                    (identifier, provider, collected, int(not added)))
        return added

    def add(self, row, provider, collected):
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            return self._insert(con, row, provider, collected)

    def existing(self, row):
        with self.connect() as con:
            return con.execute('SELECT 1 FROM articles WHERE url_key=? OR title_key=?',
                (canonical_url(row.get('url', '')) or None, normalized_headline(row['headline']) or None)).fetchone() is not None

    def rows(self):
        with self.connect() as con:
            return [json.loads(row[0]) for row in con.execute('SELECT record FROM articles')]

    def get(self, table, key):
        assert table in ('sources', 'state')
        with self.connect() as con:
            row = con.execute(f'SELECT record FROM {table} WHERE id=?', (key,)).fetchone()
            return json.loads(row[0]) if row else {}

    def put(self, table, key, value):
        assert table in ('sources', 'state')
        with self.connect() as con:
            con.execute(f'INSERT OR REPLACE INTO {table} VALUES (?,?)', (key, json.dumps(value)))

    def claim(self, current, force):
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute("SELECT record FROM state WHERE id='poll'").fetchone()
            state = json.loads(row[0]) if row else {}
            if as_time(state.get('lease_until')) and as_time(state['lease_until']) > current:
                return False, state
            if not force and as_time(state.get('next_poll_at')) and as_time(state['next_poll_at']) > current:
                return False, state
            state.update(started_at=current.isoformat(), lease_until=(current + pd.Timedelta(minutes=15)).isoformat(), running=True)
            con.execute('INSERT OR REPLACE INTO state VALUES (?,?)', ('poll', json.dumps(state)))
            return True, state


class LiveClassifier:
    """Existing weights as auxiliary evidence if they lack the fourth class."""
    def __init__(self, root):
        self.root = Path(root)
        self.model, self.error, self.digest = None, None, None
        variant = os.getenv('NEWS_TFIDF_MODEL', 'synthetic').lower()
        path = self.root / 'artifacts' / ('news_tfidf_real.joblib' if variant == 'real' else 'news_tfidf.joblib')
        self.provenance = 'Human-label model' if variant == 'real' else 'Synthetic-training demonstration model; not validated on current real news'
        self.path = str(path.relative_to(self.root))
        try:
            import joblib
            self.digest = sha256(path)
            self.model = joblib.load(path)
        except Exception as exc:
            self.error = str(exc)

    def __call__(self, entry):
        from research_labels import classify_research
        result = dict(article_verified=False, model_sha256=self.digest, model_artifact=self.path,
            model_provenance=self.provenance, classifier_error=self.error)
        classes = list(getattr(self.model, 'classes_', []))
        if self.model is not None:
            try:
                scores = self.model.predict_proba([entry['headline']])[0]
                index = int(scores.argmax())
                result.update(auxiliary_model_label=str(classes[index]), auxiliary_model_score=float(scores[index]))
                if set(LABELS).issubset(classes):
                    result.update(label=str(classes[index]), score=float(scores[index]),
                        score_type='Classifier score · uncalibrated', classifier_backend='tfidf_saved_model',
                        reasoning='Saved four-category classifier on headline only; not independent factual verification.')
                    return result
            except Exception as exc:
                result['classifier_error'] = str(exc)
        context = dict(id=entry['id'], headline=entry['headline'], url=entry['url'],
            status='retrieved_unverified', excerpt=entry.get('feed_excerpt', ''))
        label, score, reason, flags, excerpt, method = classify_research(entry, context)
        result.update(label=label, score=score, score_type='Heuristic score · uncalibrated', reasoning=reason,
            classifier_backend=method + (' + auxiliary saved TF-IDF' if self.model is not None else ' (saved model unavailable)'),
            model_provenance=self.provenance + '; four-category result uses existing research rubric because the saved model lacks four-category support',
            classification_flags=flags)
        return result


def source_status(record, current):
    error = record.get('error')
    if error:
        return 'Access restricted' if 'ROBOTS_DISALLOWED' in error or '403' in error else 'Backoff / connection error'
    latest = as_time(record.get('latest_publication_at'))
    if not record.get('last_success_at'):
        return 'Not checked yet'
    if latest is None:
        return 'Fetched · no dated XRP articles'
    age = (current - latest).total_seconds()
    return 'Fetched · recent articles' if 0 <= age <= 86400 else 'Fetched · articles are old / future-dated'


def refresh(root, force=False, sources=SOURCES, client_factory=PublicClient, classifier=None, clock=now):
    store = NewsStore(root)
    current = pd.Timestamp(clock())
    if os.getenv('XRP_DASHBOARD_OFFLINE') == '1':
        return dict(checked_at=current.isoformat(), errors=['Offline mode: public feed polling disabled.'], added=0, skipped=True)
    claimed, state = store.claim(current, force)
    if not claimed:
        return dict(checked_at=state.get('checked_at'), errors=state.get('errors', []), added=0,
            skipped=True, running=state.get('running', False))
    access_path = store.directory / 'access.sqlite3'
    legacy = Path(root) / 'data' / 'dashboard_cache' / 'public_feed_access.sqlite3'
    if not access_path.exists() and legacy.exists():
        with closing(sqlite3.connect(legacy)) as src, closing(sqlite3.connect(access_path)) as dst:
            src.backup(dst)  # Keep prior host restrictions/cooldowns, never reset them.
    access = Store(access_path)
    client = client_factory(access, budget=60)
    classify = classifier or LiveClassifier(root)
    added_total, errors = 0, []
    try:
        for job in sources:
            provider = job['provider']
            previous = store.get('sources', provider)
            attempted = pd.Timestamp(clock())
            next_at = as_time(previous.get('next_attempt_at'))
            last = as_time(previous.get('last_attempt_at'))
            if next_at and next_at > attempted and (not force or previous.get('error') or (last is not None and (attempted-last).total_seconds() < 30)):
                continue
            record = dict(previous, name=job['publisher'], provider=provider, feed_url=job['url'],
                last_attempt_at=attempted.isoformat(), articles_returned=0, relevant_articles=0, newly_saved=0)
            try:
                response = client.fetch(job, previous)
                collected = clock()
                record.update(http_status=response.status_code)
                if response.status_code != 304:
                    entries, raw_count = parse_feed(response.content, job, collected)
                    record.update(articles_returned=raw_count, relevant_articles=len(entries),
                        latest_publication_at=max((e['published_at'] for e in entries if as_time(e['published_at']) is not None), default=None),
                        etag=response.headers.get('ETag', ''), last_modified=response.headers.get('Last-Modified', ''))
                    for entry in entries:
                        entry = dict(entry)
                        entry['id'] = 'LIVE-' + hashlib.sha256(canonical_url(entry['url']).encode()).hexdigest()[:24]
                        if not store.existing(entry):
                            try:
                                assessment = classify(entry)
                            except Exception as exc:
                                assessment = dict(label='REVIEW_REQUIRED', score=None, score_type='Unavailable',
                                    classifier_backend='unavailable', model_provenance='Classification failed', reasoning=str(exc), article_verified=False)
                            entry.update(assessment)
                        entry.update(label_status='AI provisional' if entry.get('label') in LABELS else 'Needs review',
                            verification_status='AI_PROVISIONAL', first_seen_at=collected, observed_at=collected,
                            collected_at=collected, last_seen_at=collected, outlook_eligible=True, origin='Persistent public live feed')
                        if store.add(entry, provider, collected):
                            record['newly_saved'] += 1
                            added_total += 1
                record.update(error=None, failures=0, last_success_at=collected,
                    next_attempt_at=(pd.Timestamp(collected)+pd.Timedelta(seconds=INTERVAL)).isoformat())
                record['status'] = source_status(record, pd.Timestamp(clock()))
            except Exception as exc:
                failures = previous.get('failures', 0) + 1
                error = str(exc)
                delay = min(21600, INTERVAL * (2 ** min(failures-1, 7)))
                if 'ROBOTS_DISALLOWED' in error or '404' in error:
                    delay = max(delay, 3600)
                host = access.get('hosts', urlparse(job['url']).netloc, 'host')
                cooldown = as_time(host.get('cooldown_until'))
                next_try = pd.Timestamp(clock()) + pd.Timedelta(seconds=delay)
                if cooldown is not None:
                    next_try = max(next_try, cooldown)
                http = re.search(r'HTTP\s+(\d{3})', error)
                record.update(error=error, failures=failures, next_attempt_at=next_try.isoformat(),
                    http_status=int(http[1]) if http else record.get('http_status') if 'RSS/Atom' in error else None)
                record['status'] = source_status(record, pd.Timestamp(clock()))
                errors.append(job['publisher'] + ': ' + error)
            store.put('sources', provider, record)
    finally:
        access.db.close()
        checked = clock()
        state.update(checked_at=checked, lease_until=None, running=False, added=added_total, errors=errors,
            next_poll_at=(current + pd.Timedelta(seconds=INTERVAL)).isoformat())
        store.put('state', 'poll', state)
    return dict(checked_at=checked, errors=errors, added=added_total, skipped=False)


def load_rows(root):
    path = folder(root) / 'news.sqlite3'
    if not path.exists():
        return []
    with closing(sqlite3.connect('file:' + path.resolve().as_posix() + '?mode=ro', uri=True)) as con:
        return [json.loads(row[0]) for row in con.execute('SELECT record FROM articles')]


def monitoring(root):
    path = folder(root) / 'news.sqlite3'
    if not path.exists():
        return {}, [dict(provider=s['provider'], name=s['publisher'], status='Not checked yet') for s in SOURCES]
    with closing(sqlite3.connect('file:' + path.resolve().as_posix() + '?mode=ro', uri=True)) as con:
        row = con.execute("SELECT record FROM state WHERE id='poll'").fetchone()
        state = json.loads(row[0]) if row else {}
        records = {i: json.loads(v) for i, v in con.execute('SELECT id,record FROM sources')}
    return state, [records.get(s['provider'], dict(provider=s['provider'], name=s['publisher'], status='Not checked yet')) for s in SOURCES]


class NewsPoller:
    def __init__(self, root, interval=INTERVAL, refresh_fn=refresh):
        self.root, self.interval, self.refresh_fn = Path(root), interval, refresh_fn
        self.lock = threading.Lock()
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.thread = None
        self.manual = False
        self.enabled = False
        self.error = None

    def start(self):
        self.enabled = True
        return self._ensure_started()

    def _ensure_started(self):
        if os.getenv('XRP_DASHBOARD_OFFLINE') == '1':
            return self
        with self.lock:
            if not self.thread or not self.thread.is_alive():
                self.stop_event.clear()
                self.thread = threading.Thread(target=self._run, name='xrp-public-news-poller', daemon=True)
                self.thread.start()
        return self

    def request(self):
        with self.lock:
            self.manual = True
        self.wake.set()
        self._ensure_started()

    def pause(self):
        self.enabled = False
        self.wake.set()

    def stop(self):
        self.stop_event.set()
        self.wake.set()

    def _run(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.lock:
                force, self.manual = self.manual, False
            try:
                if self.enabled or force:
                    self.refresh_fn(self.root, force=force)
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            self.wake.wait(max(.05, self.interval - (time.monotonic() - started)) if self.enabled else None)
            self.wake.clear()
