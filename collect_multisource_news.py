"""Incremental public RSS/Atom and GDELT metadata collector. Never trains a model.

Original datasets/reviews remain untouched. Outputs and labeling workspace live
under data/multisource/. No article scraping, private APIs or Google URL decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from ai_labeling import LABELS, ROOT, atomic_csv, backup, dataset, load_decisions, matching, now, read_csv, workflow_lock
from batch_news_labeling import batch, read_json_rows
from collect_real_news import API as GDELT_API, QUERY as GDELT_QUERY, parse_gdelt_response
from news_label_safety import atomic_json, canonical_url, duplicate_groups, normalized_headline, sha256, utc_date

USER_AGENT = 'XRP-Academic-News-Collector/1.0'
TOPIC = re.compile(r'\b(XRP|XRPL|RLUSD)\b|XRP Ledger', re.I)
PUBLISHER_FEEDS = [
    ('coindesk', 'CoinDesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml', False),
    ('cointelegraph', 'Cointelegraph', 'https://cointelegraph.com/rss', False),
    ('decrypt', 'Decrypt', 'https://decrypt.co/feed', False),
    ('cryptoslate', 'CryptoSlate', 'https://cryptoslate.com/feed/', False),
    ('xrpl', 'XRP Ledger', 'https://xrpl.org/blog/rss.xml', True),
    ('ripple', 'Ripple', 'https://ripple.com/insights/feed/', True),
]
TOPIC_FEEDS = [
    ('cointelegraph_xrp', 'Cointelegraph', 'https://cointelegraph.com/rss/tag/xrp', False),
    ('newsbtc_xrp', 'NewsBTC', 'https://www.newsbtc.com/tag/xrp/feed/', False),
    ('bitcoinist_xrp', 'Bitcoinist', 'https://bitcoinist.com/tag/xrp/feed/', False),
    ('cryptopotato_xrp', 'CryptoPotato', 'https://cryptopotato.com/tag/ripple/feed/', False),
    ('cryptobasic_xrp', 'The Crypto Basic', 'https://thecryptobasic.com/category/ripple/feed/', False),
    ('dailyhodl_xrp', 'The Daily Hodl', 'https://dailyhodl.com/ripple-and-xrp/feed/', False),
    ('utoday', 'U.Today', 'https://u.today/rss.php', False),
]
EXTRA_FIELDS = ['publisher', 'article_url', 'canonical_url', 'discovery_source', 'source_feed',
                'published_at', 'published_at_raw', 'publication_time_status', 'gdelt_first_seen_at',
                'collected_at', 'availability_at', 'historical_eligible', 'feed_excerpt',
                'context_kind', 'url_kind', 'legacy_preserved']
BASE_FIELDS = ['id', 'date', 'headline', 'source', 'url', 'timestamp_type', 'label']


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def plain(value):
    parser = PlainText()
    parser.feed(value or '')
    return re.sub(r'\s+', ' ', html.unescape(' '.join(parser.parts))).strip()


def timestamp(value):
    """Parse only dates with explicit timezone; leave absent/date-only values blank."""
    value = (value or '').strip()
    parsed = utc_date(value)
    if parsed:
        return parsed.isoformat()
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.astimezone(timezone.utc).isoformat() if parsed.tzinfo else ''
    except (ValueError, TypeError, OverflowError):
        return ''


def element_text(element, names):
    for child in element:
        if child.tag.rsplit('}', 1)[-1] in names and child.text:
            return child.text.strip()
    return ''


def parse_feed(content, job, observed):
    if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():
        raise ValueError('DTD/entity declarations are not supported')
    root = ET.fromstring(content)
    if root.tag.rsplit('}', 1)[-1] not in ('rss', 'feed', 'RDF'):
        raise ValueError('Endpoint did not return an RSS/Atom feed')
    entries = [node for node in root.iter() if node.tag.rsplit('}', 1)[-1] in ('item', 'entry')]
    rows = []
    for entry in entries:
        title = plain(element_text(entry, {'title'}))
        link = element_text(entry, {'link'})
        for child in entry:
            if child.tag.rsplit('}', 1)[-1] == 'link' and child.attrib.get('rel', 'alternate') == 'alternate':
                link = child.attrib.get('href') or link
        publisher = element_text(entry, {'source'}) or ('' if job['provider'] == 'google_news' else job['publisher'])
        if job['provider'] == 'google_news' and publisher and title.endswith(' - ' + publisher):
            title = title[:-(len(publisher) + 3)].strip()
        if not title or not link.startswith(('https://', 'http://')):
            continue
        if not TOPIC.search(title) and not (job.get('official') or re.search(r'\bRipple\b', title)):
            continue
        raw_date = element_text(entry, {'pubDate', 'published', 'date'})
        published = timestamp(raw_date)  # Atom updated is deliberately NOT a publication timestamp.
        status = 'SOURCE_REPORTED' if published else ('INVALID_OR_TIMEZONE_MISSING' if raw_date else 'MISSING')
        if published and utc_date(published) > utc_date(observed):
            status = 'FUTURE_REPORTED_TIMESTAMP'
        excerpt = ' '.join(plain(element_text(entry, {'description', 'summary'})).split()[:180])
        google = job['provider'] == 'google_news'
        rows.append(dict(headline=title, source=publisher or 'Unknown publisher (Google News)', publisher=publisher, url=link,
                         article_url=link, canonical_url=canonical_url(link), discovery_source=job['provider'],
                         source_feed=job['url'], published_at=published, published_at_raw=raw_date,
                         publication_time_status=status, gdelt_first_seen_at='', collected_at=observed,
                         availability_at=observed, date=observed, timestamp_type='PROJECT_OBSERVED_AT', label='',
                         historical_eligible='False', feed_excerpt=excerpt,
                         context_kind='AGGREGATOR_SNIPPET' if google else 'PUBLISHER_FEED_SUMMARY',
                         url_kind='GOOGLE_NEWS_ARTICLE_LINK' if google else 'PUBLISHER_ARTICLE_LINK',
                         legacy_preserved='False'))
    return rows, len(entries)


def gdelt_rows(content, job, observed):
    payload = json.loads(content)
    if not isinstance(payload, dict) or not isinstance(payload.get('articles', []), list):
        raise ValueError('Invalid GDELT article response')
    rows = []
    for article in parse_gdelt_response(payload):
        seen = article['date']
        # Use the time THIS project retrieved the headline for new backtests.
        article.update(date=observed, timestamp_type='PROJECT_OBSERVED_AT', publisher=article['source'],
                       article_url=article['url'], canonical_url=canonical_url(article['url']),
                       discovery_source='gdelt', source_feed=job['url'], published_at='', published_at_raw='',
                       publication_time_status='MISSING', gdelt_first_seen_at=seen, collected_at=observed,
                       availability_at=observed, historical_eligible='False', feed_excerpt='', context_kind='NONE',
                       url_kind='PUBLISHER_ARTICLE_LINK', legacy_preserved='False')
        rows.append(article)
    return rows, len(payload.get('articles', []))


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS articles(id TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations(key TEXT PRIMARY KEY, article_id TEXT, provider TEXT,
                duplicate INTEGER, collected_at TEXT, record TEXT);
            CREATE TABLE IF NOT EXISTS jobs(key TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS hosts(host TEXT PRIMARY KEY, record TEXT NOT NULL);
        ''')

    def rows(self):
        return [json.loads(row['record']) for row in self.db.execute('SELECT record FROM articles ORDER BY rowid')]

    def get(self, table, key, column='key'):
        row = self.db.execute(f'SELECT record FROM {table} WHERE {column}=?', (key,)).fetchone()
        return json.loads(row['record']) if row else {}

    def put(self, table, key, value, column='key'):
        self.db.execute(f'INSERT INTO {table}({column},record) VALUES (?,?) '
                        f'ON CONFLICT({column}) DO UPDATE SET record=excluded.record', (key, json.dumps(value)))

    def seed(self, source):
        fields, rows = dataset(source)
        for row in rows:
            existing = self.get('articles', row['id'], 'id')
            if existing:
                if not matching(row, existing):
                    raise ValueError('A protected legacy article changed; resolve before collection')
                continue
            extra = {key: '' for key in EXTRA_FIELDS}
            extra.update(publisher=row['source'], article_url=row['url'], canonical_url=canonical_url(row['url']),
                         discovery_source='legacy_gdelt', gdelt_first_seen_at=row['date'] if row['timestamp_type'] == 'GDELT_FIRST_SEEN' else '',
                         availability_at=row['date'], publication_time_status='MISSING', legacy_preserved='True',
                         historical_eligible='LEGACY_FIRST_SEEN_UNVERIFIED', context_kind='LEGACY', url_kind='PUBLISHER_ARTICLE_LINK')
            self.put('articles', row['id'], dict(extra, **row), 'id')
        self.db.commit()
        return fields, rows

    def add(self, row, job_key):
        key = hashlib.sha256((job_key + row['url'] + row['headline']).encode('utf-8')).hexdigest()
        if self.db.execute('SELECT 1 FROM observations WHERE key=?', (key,)).fetchone():
            return 'already_seen'
        existing = self.rows()
        title = normalized_headline(row['headline'])
        tokens = set(title.split())
        duplicate = None
        for old in existing:
            other = normalized_headline(old['headline'])
            other_tokens = set(other.split())
            similarity = (len(tokens & other_tokens) / len(tokens | other_tokens)) if tokens and other_tokens else 0
            same_numbers = {token for token in tokens if token.isdigit()} == {token for token in other_tokens if token.isdigit()}
            if (row['canonical_url'] == old['canonical_url'] or title == other
                    or (same_numbers and min(len(tokens), len(other_tokens)) >= 6 and similarity >= .85)):
                duplicate = old
                break
        if duplicate:
            article_id = duplicate['id']
            # Enrich missing provenance only. Never change an existing headline, date, label or URL.
            enriched = dict(duplicate)
            for field in ('published_at', 'published_at_raw', 'gdelt_first_seen_at'):
                if row['canonical_url'] == duplicate['canonical_url'] and not enriched.get(field) and row.get(field):
                    enriched[field] = row[field]
            if not duplicate.get('published_at') and enriched.get('published_at'):
                enriched['publication_time_status'] = row['publication_time_status']
            self.put('articles', article_id, enriched, 'id')
        else:
            article_id = row.get('id') or 'NEWS-' + hashlib.sha256(row['canonical_url'].encode('utf-8')).hexdigest()[:20]
            row = dict(row, id=article_id)
            self.put('articles', article_id, row, 'id')
        self.db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?)',
                        (key, article_id, row['discovery_source'], int(duplicate is not None), row['collected_at'], json.dumps(row)))
        return 'duplicate' if duplicate else 'added'


class PublicClient:
    def __init__(self, store, budget=40, session=None):
        self.store, self.budget, self.used = store, budget, 0
        self.session = session or requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})

    def request(self, url, headers=None):
        if self.used >= self.budget:
            raise RuntimeError('REQUEST_BUDGET_EXHAUSTED')
        self.used += 1
        # No cookies, authentication, browser impersonation, alternate proxies or retries.
        self.session.cookies.clear()
        response = self.session.get(url, timeout=(8, 20), headers=headers or {}, allow_redirects=False)
        return response

    def remember_block(self, host, state, response):
        if response.status_code not in (403, 429, 503):
            return
        delay = 3600 if response.status_code == 403 else 900
        retry = response.headers.get('Retry-After', '')
        try:
            delay = max(delay, float(retry))
        except ValueError:
            retry_date = timestamp(retry)
            if retry_date:
                delay = max(delay, (utc_date(retry_date) - utc_date(now())).total_seconds())
        state['cooldown_until'] = (utc_date(now()) + timedelta(seconds=delay)).isoformat()
        self.store.put('hosts', host, state, 'host')
        self.store.db.commit()

    def fetch(self, job, previous, redirects=0):
        host = urlparse(job['url']).netloc
        state = self.store.get('hosts', host, 'host')
        current = utc_date(now())
        cooldown = utc_date(state.get('cooldown_until', ''))
        if cooldown and cooldown > current:
            raise RuntimeError('HOST_COOLDOWN_UNTIL ' + cooldown.isoformat())
        if not state.get('robots_checked_at') or current - utc_date(state['robots_checked_at']) > timedelta(days=1):
            robots_url = 'https://' + host + '/robots.txt'
            response = self.request(robots_url)
            if response.status_code not in (200, 404):
                self.remember_block(host, state, response)
                raise RuntimeError(f'ROBOTS_UNAVAILABLE HTTP {response.status_code}')
            state.update(robots_checked_at=now(), robots=response.text if response.status_code == 200 else '')
            self.store.put('hosts', host, state, 'host')
            self.store.db.commit()
        parser = RobotFileParser()
        parser.parse(state.get('robots', '').splitlines())
        if not parser.can_fetch(USER_AGENT, job['url']):
            raise RuntimeError('ROBOTS_DISALLOWED')
        interval = max(15 if job['provider'] == 'gdelt' else 5, parser.crawl_delay(USER_AGENT) or 0)
        last = utc_date(state.get('last_request_at', ''))
        if last:
            wait = interval - (utc_date(now()) - last).total_seconds()
            if wait > 30:
                raise RuntimeError('HOST_CRAWL_DELAY_ACTIVE')
            if wait > 0:
                time.sleep(wait)
        state['last_request_at'] = now()
        self.store.put('hosts', host, state, 'host')
        self.store.db.commit()
        response = self.request(job['url'], {key: previous[value] for key, value in
                                           [('If-None-Match', 'etag'), ('If-Modified-Since', 'last_modified')]
                                           if previous.get(value)})
        if response.status_code in (301, 302, 303, 307, 308):
            destination = urljoin(job['url'], response.headers.get('Location', ''))
            if redirects >= 3 or not destination.startswith('https://') or destination == job['url']:
                raise RuntimeError('Unsupported or excessive feed redirects')
            # Ordinary published redirects only, with fresh robots/cooldown checks.
            return self.fetch(dict(job, url=destination), previous, redirects + 1)
        self.remember_block(host, state, response)
        if response.status_code not in (200, 304):
            raise RuntimeError(f'HTTP {response.status_code}; no redirect/access bypass attempted')
        if len(response.content) > 5_000_000:
            raise ValueError('Feed response exceeds 5 MB limit')
        return response


def build_jobs(days=84, anchor=None):
    anchor = utc_date(anchor) if anchor else utc_date(now())
    jobs = []
    # Probe GDELT once before RSS; further windows obey a host-wide cooldown.
    end = anchor - timedelta(minutes=20)
    start = end - timedelta(days=min(days, 85))
    cursor = start
    gdelt = []
    while cursor < end:
        next_week = (cursor + timedelta(days=7 - cursor.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        stop = min(next_week, end)
        url = GDELT_API + '?' + urlencode(dict(query=GDELT_QUERY, mode='ArtList', format='json',
                                             startdatetime=cursor.strftime('%Y%m%d%H%M%S'),
                                             enddatetime=stop.strftime('%Y%m%d%H%M%S'), maxrecords=250, sort='DateAsc'))
        gdelt.append(dict(provider='gdelt', publisher='GDELT', url=url, closed=True))
        cursor = stop
    jobs += gdelt[:1]
    jobs += [dict(provider=name, publisher=publisher, url=url, official=official, closed=False)
             for name, publisher, url, official in PUBLISHER_FEEDS]
    jobs += [dict(provider=name, publisher=publisher, url=url, official=official, closed=False)
             for name, publisher, url, official in TOPIC_FEEDS]
    # Public WordPress RSS pagination. Each URL is independently robots-checked;
    # 404/403/429 stops access to that endpoint/host rather than trying private APIs.
    for page in range(2, 21):
        jobs += [dict(provider=name, publisher=publisher, url=url + '?' + urlencode({'paged': page}),
                      official=official, closed=False, page=page, base_feed=url)
                 for name, publisher, url, official in TOPIC_FEEDS
                 if name in ('newsbtc_xrp', 'bitcoinist_xrp', 'cryptopotato_xrp', 'cryptobasic_xrp', 'dailyhodl_xrp')]
    # Stable calendar weeks keep completed windows identical on a later resume.
    today = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    week = today - timedelta(days=today.weekday())
    stop = today + timedelta(days=1)
    while stop > today - timedelta(days=days):
        begin = week if stop > week else stop - timedelta(days=7)
        query = f'(XRP OR XRPL OR RLUSD) after:{begin.date()} before:{stop.date()}'
        jobs.append(dict(provider='google_news', publisher='Google News', closed=stop <= today,
                         url='https://news.google.com/rss/search?' + urlencode(dict(q=query, hl='en-US', gl='US', ceid='US:en'))))
        stop = begin
    jobs += gdelt[1:]
    return jobs


def run_job(store, client, job, target):
    key = hashlib.sha256(job['url'].encode('utf-8')).hexdigest()
    previous = store.get('jobs', key)
    record = dict(previous, provider=job['provider'], url=job['url'], page=job.get('page', 1))
    if job.get('page', 1) > 1:
        prior_jobs = [json.loads(row['record']) for row in store.db.execute('SELECT record FROM jobs')]
        for prior in prior_jobs:
            if prior.get('provider') != job['provider']:
                continue
            fresh = utc_date(prior.get('next_check_at', ''))
            if not fresh or fresh <= utc_date(now()):
                continue
            if ((prior.get('url') == job.get('base_feed') and prior.get('relevant_items') == 0)
                    or prior.get('pagination_stopped')):
                return 'feed_pagination_exhausted'
    payload = previous.get('pending_rows', [])
    if not payload and previous.get('next_check_at') and utc_date(previous['next_check_at']) > utc_date(now()):
        return 'cached_or_cooldown'
    if not payload:
        try:
            response = client.fetch(job, previous)
            record.update(last_checked_at=now(), http_status=response.status_code)
            if response.status_code == 304:
                record.update(status='not_modified', next_check_at=(utc_date(now()) + timedelta(hours=1)).isoformat())
                store.put('jobs', key, record)
                store.db.commit()
                return 'not_modified'
            content = response.content
            if b'<html' in content[:500].lower():
                raise ValueError('HTML response instead of a public feed/API response')
            observed = now()
            payload, raw_count = (gdelt_rows(content, job, observed) if job['provider'] == 'gdelt'
                                  else parse_feed(content, job, observed))
            record.update(pending_rows=payload, status='fetched', raw_items=raw_count, relevant_items=len(payload),
                          etag=response.headers.get('ETag', ''), last_modified=response.headers.get('Last-Modified', ''),
                          response_sha256=hashlib.sha256(content).hexdigest(), error='',
                          potentially_truncated=raw_count >= (250 if job['provider'] == 'gdelt' else 100),
                          pagination_stopped='')
            signature = hashlib.sha256(json.dumps([(row['url'], row['headline']) for row in payload]).encode()).hexdigest()
            record['item_signature'] = signature
            if job.get('page', 1) > 1:
                for stored in store.db.execute('SELECT record FROM jobs'):
                    prior = json.loads(stored['record'])
                    if prior.get('provider') == job['provider'] and prior.get('item_signature') == signature and prior.get('url') != job['url']:
                        record['pagination_stopped'] = 'Feed repeats the same entries for another page'
                        break
                if not payload:
                    record['pagination_stopped'] = 'No more relevant feed entries'
        except (requests.RequestException, RuntimeError, ValueError, ET.ParseError) as exc:
            if isinstance(exc, requests.RequestException):
                host = urlparse(job['url']).netloc
                state = store.get('hosts', host, 'host')
                state['cooldown_until'] = (utc_date(now()) + timedelta(minutes=15)).isoformat()
                store.put('hosts', host, state, 'host')
            record.update(status='failed', error=str(exc), last_checked_at=now(),
                          next_check_at=(utc_date(now()) + timedelta(hours=1)).isoformat())
            if job.get('page', 1) > 1 and 'HTTP 404' in str(exc):
                record['pagination_stopped'] = 'Public feed page does not exist'
            store.put('jobs', key, record)
            store.db.commit()
            print(job['provider'] + ': ' + str(exc), flush=True)
            return 'failed'
    added = duplicates = 0
    while payload and len(store.rows()) < target:
        row = payload.pop(0)
        result = store.add(row, key)
        added += result == 'added'
        duplicates += result == 'duplicate'
        record['pending_rows'] = payload
        # Observation, article and queue position commit together for crash-safe resume.
        store.put('jobs', key, record)
        store.db.commit()
    record.update(pending_rows=payload, status='pending' if payload else 'complete',
                  next_check_at=(utc_date(now()) + timedelta(days=3650 if job['closed'] else 0,
                                                            hours=0 if job['closed'] else 1)).isoformat(),
                  added_total=record.get('added_total', 0) + added,
                  duplicate_total=record.get('duplicate_total', 0) + duplicates)
    if not payload and added == 0 and previous.get('status') in ('complete', 'not_modified'):
        record['pagination_stopped'] = 'Previously collected feed contains no new articles'
    store.put('jobs', key, record)
    store.db.commit()
    print(f'{job["provider"]}: added {added}, duplicates {duplicates}; total {len(store.rows())}', flush=True)
    return record['status']


def merge_review_file(source, destination, csv_file=False):
    if not source.exists():
        return
    read = (lambda path: read_csv(path)[1]) if csv_file else read_json_rows
    incoming = read(source)
    old = read(destination) if destination.exists() else []
    merged = {row['id']: row for row in old}
    for row in incoming:
        if row['id'] not in merged:
            merged[row['id']] = row
        elif csv_file and merged[row['id']] != row:
            raise ValueError(f'Conflicting protected review record: {row["id"]}')
    if destination.exists():
        backup(destination)
    if csv_file:
        atomic_csv(destination, read_csv(source)[0], list(merged.values()))
    else:
        atomic_json(destination, list(merged.values()))


def export_workspace(store, root_data, folder):
    rows = store.rows()
    _, legacy = dataset(root_data / 'xrp_real_labeled.csv')
    legacy_order = {row['id']: index for index, row in enumerate(legacy)}
    rows.sort(key=lambda row: (0, legacy_order[row['id']]) if row['id'] in legacy_order else (1, 0))
    # Refresh human annotations only from actual human datasets; never predict into label.
    existing_path = folder / 'xrp_real_labeled.csv'
    existing_fields, prior = read_csv(existing_path) if existing_path.exists() else ([], [])
    by_prior = {row['id']: row for row in prior}
    by_legacy = {row['id']: row for row in legacy}
    fields = list(dict.fromkeys(BASE_FIELDS + EXTRA_FIELDS + existing_fields + [key for row in legacy for key in row]))
    exported = []
    for row in rows:
        row = dict(row)
        for old in (by_prior.get(row['id']), by_legacy.get(row['id'])):
            if not old:
                continue
            if not matching(old, row):
                raise ValueError('Protected dataset record changed')
            if old.get('label'):
                if row['label'] and row['label'] != old['label']:
                    raise ValueError('Conflicting protected human labels')
                row.update({key: val for key, val in old.items() if key == 'label' or key.startswith('human_') or key == 'label_origin'})
        exported.append({key: row.get(key, '') for key in fields})
    for name in ('xrp_real_labeled.csv', 'xrp_news_master.csv'):
        if (folder / name).exists():
            backup(folder / name)
        atomic_csv(folder / name, fields, exported)
    for name in ('xrp_ai_suggestions.csv', 'xrp_human_reviews.csv', 'xrp_ai_proposals.json', 'xrp_article_context.json'):
        merge_review_file(root_data / name, folder / name, name.endswith('.csv'))
    contexts = read_json_rows(folder / 'xrp_article_context.json')
    context_ids = {row['id'] for row in contexts}
    for row in rows:
        if row['id'] in context_ids or row['context_kind'] != 'PUBLISHER_FEED_SUMMARY' or not row['feed_excerpt']:
            continue
        contexts.append(dict(id=row['id'], headline=row['headline'], url=row['url'],
                             status='retrieved_unverified', excerpt=row['feed_excerpt'],
                             fetched_at=row['collected_at'], article_verified=False,
                             detail='Publisher-provided feed summary, not full article or factual verification.'))
    atomic_json(folder / 'xrp_article_context.json', contexts)
    return exported


def quality_report(store, folder, initial_count, protected_hashes, run_requests):
    rows = store.rows()
    groups = duplicate_groups(rows)
    auto_path = folder / 'xrp_auto_labeled.csv'
    auto = read_csv(auto_path)[1] if auto_path.exists() else []
    jobs = [json.loads(row['record']) for row in store.db.execute('SELECT record FROM jobs')]
    legacy_groups = duplicate_groups([row for row in rows if row['legacy_preserved'] == 'True'])
    human_rows = dataset(folder / 'xrp_real_labeled.csv')[1]
    decisions = {row['id']: row for row in load_decisions(folder / 'xrp_human_reviews.csv')}
    human_counts = Counter()
    for row in human_rows:
        label = row['label']
        decision = decisions.get(row['id'])
        if not label and decision and matching(row, decision) and decision['review_status'] in ('accepted', 'corrected'):
            label = decision['human_label']
        if label in LABELS:
            human_counts[label] += 1
    report = dict(generated_at=now(), total_real_records=len(rows), legacy_records_preserved=initial_count,
                  new_records=len(rows) - initial_count,
                  discovery_source_distribution=dict(Counter(row['discovery_source'] for row in rows)),
                  publisher_distribution=dict(Counter(row['publisher'] for row in rows)),
                  duplicate_observations=store.db.execute('SELECT COUNT(*) FROM observations WHERE duplicate=1').fetchone()[0],
                  retained_legacy_duplicate_groups=sum(len(group) > 1 for group in legacy_groups),
                  possible_duplicate_groups=sum(len(group) > 1 for group in groups),
                  distinct_article_groups=len(groups), missing_publication_timestamps=sum(not row['published_at'] for row in rows),
                  missing_project_collection_timestamps=sum(not row['collected_at'] for row in rows),
                  missing_availability_timestamps=sum(not row['availability_at'] for row in rows),
                  gdelt_first_seen_present=sum(bool(row['gdelt_first_seen_at']) for row in rows),
                  future_reported_timestamps=sum(row['publication_time_status'] == 'FUTURE_REPORTED_TIMESTAMP' for row in rows),
                  google_article_links=sum(row['url_kind'] == 'GOOGLE_NEWS_ARTICLE_LINK' for row in rows),
                  ai_provisional_category_distribution=dict(Counter(row['predicted_label'] for row in auto)),
                  human_dataset_category_distribution=dict(Counter(row['label'] or 'UNLABELED' for row in dataset(folder / 'xrp_real_labeled.csv')[1])),
                  human_annotation_distribution=dict(human_counts),
                  publication_time_status_distribution=dict(Counter(row['publication_time_status'] for row in rows)),
                  earliest_reported_publication=min((row['published_at'] for row in rows if row['published_at']), default=''),
                  latest_reported_publication=max((row['published_at'] for row in rows if row['published_at']), default=''),
                  http_requests_this_run=run_requests, protected_input_sha256=protected_hashes,
                  failed_source_connections=[{key: job.get(key, '') for key in ('provider', 'url', 'error', 'last_checked_at')} for job in jobs if job['status'] == 'failed'],
                  source_jobs=[{key: value for key, value in job.items() if key != 'pending_rows'} for job in jobs],
                  successful_feed_connections=sorted({job['provider'] for job in jobs if job['status'] in ('complete', 'pending', 'not_modified')}),
                  limitations=['Feed publication dates are source-reported, not independently verified.',
                               'Google News links are public article landing links; original publisher URLs are not decoded or invented.',
                               'New availability/date timestamps are project observation times, never backdated to reported publication.',
                               'All 53 legacy rows remain, including detected legacy duplicates; new duplicates are rejected.',
                               'AI labels are provisional; absent article context can require review. No model training performed.'])
    atomic_json(folder / 'data_quality_report.json', report)
    lines = ['# Multi-source XRP dataset quality', '',
             f'Real records: {len(rows)}. Preserved legacy: {initial_count}. New: {report["new_records"]}.',
             f'Duplicate observations rejected: {report["duplicate_observations"]}. Distinct article groups: {len(groups)}.',
             f'Missing publication dates: {report["missing_publication_timestamps"]}. Google landing links: {report["google_article_links"]}.',
             '', '| Discovery source | Records |', '| --- | ---: |']
    lines += [f'| {key} | {count} |' for key, count in report['discovery_source_distribution'].items()]
    lines += ['', '## AI provisional categories', '', '| Category | Records |', '| --- | ---: |']
    lines += [f'| {label} | {report["ai_provisional_category_distribution"].get(label, 0)} |' for label in (*LABELS, 'REVIEW_REQUIRED')]
    lines += ['', '## Failed source connections', '']
    lines += [f'- {item["provider"]}: {item["error"]}' for item in report['failed_source_connections']] or ['None recorded.']
    lines += ['', '## Limits', ''] + ['- ' + value for value in report['limitations']]
    (folder / 'data_quality_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def collect(root_data, folder, target=400, days=84, max_requests=40, classify=True):
    root_data, folder = Path(root_data), Path(folder)
    if not 53 <= target <= 500 or not 1 <= days <= 365 or max_requests < 1:
        raise ValueError('Use target 53..500, days 1..365 and a positive HTTP request budget')
    if folder.resolve() == root_data.resolve():
        raise ValueError('Collection workspace must be separate from original data')
    folder.mkdir(parents=True, exist_ok=True)
    with workflow_lock(folder):
        protected = {path.name: sha256(path) for path in root_data.iterdir()
                     if path.is_file() and path.suffix in ('.csv', '.json')}
        for name in protected:
            backup(root_data / name)
        store = Store(folder / 'collection.sqlite3')
        try:
            _, legacy = store.seed(root_data / 'xrp_real_labeled.csv')
            client = PublicClient(store, max_requests)
            # Fixed UTC-day anchor makes GDELT checkpoints reusable across runs today.
            anchor = utc_date(now()).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            for job in build_jobs(days, anchor):
                if len(store.rows()) >= target or client.used >= max_requests:
                    break
                run_job(store, client, job, target)
            export_workspace(store, root_data, folder)
            if classify:
                batch(folder)  # No article scraping; cached excerpts and feed summaries only.
            for name, digest in protected.items():
                if sha256(root_data / name) != digest:
                    raise RuntimeError('Protected input changed: ' + name)
            report = quality_report(store, folder, len(legacy), protected, client.used)
            print(json.dumps({key: report[key] for key in ('total_real_records', 'new_records',
                             'discovery_source_distribution', 'duplicate_observations',
                             'missing_publication_timestamps', 'ai_provisional_category_distribution')}, indent=2))
            if report['total_real_records'] < target:
                print('Target not yet reached. Resume the same command later; no records are fabricated.')
            return report
        finally:
            store.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, default=400)
    parser.add_argument('--days', type=int, default=84)
    parser.add_argument('--max-requests', type=int, default=40)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-classify', action='store_true')
    args = parser.parse_args()
    collect(args.data_dir, args.output_dir or args.data_dir / 'multisource', args.target, args.days, args.max_requests, not args.no_classify)


if __name__ == '__main__':
    main()
