"""Bounded robots-aware retrieval of public article excerpts; no verification claim."""
from __future__ import annotations

import json
import re
import sqlite3
from html.parser import HTMLParser
from pathlib import Path

import requests

from ai_labeling import backup, matching, now
from batch_news_labeling import read_json_rows
from collect_multisource_news import PublicClient, Store
from datetime import timedelta
from urllib.parse import urlparse
from news_label_safety import atomic_json, normalized_headline, utc_date
from research_labels import clean_excerpt


class ArticleParagraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scope = 0
        self.in_p = False
        self.parts = []
        self.paragraphs = []
        self.in_title = False
        self.title = []
        self.skip = 0
        self.frames = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        container = tag == 'div' and bool(re.search(r'entry-content|post-content|article-body|post-body|single-post-content|content-body', attributes.get('class', '') + ' ' + attributes.get('id', '')))
        scoped = tag in ('article', 'main') or container
        ignored = tag in ('script', 'style', 'nav', 'footer')
        if tag not in ('area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'):
            self.frames.append((tag, scoped, ignored))
        if scoped:
            self.scope += 1
        if ignored:
            self.skip += 1
        if tag == 'h1':
            self.in_title = True
        if tag == 'p' and self.scope:
            self.in_p = True
            self.parts = []

    def handle_endtag(self, tag):
        if tag == 'p' and self.in_p:
            text = re.sub(r'\s+', ' ', ' '.join(self.parts)).strip()
            if len(text) >= 65 and not re.search(r'related reading|subscribe|newsletter|read more', text, re.I):
                self.paragraphs.append(text)
            self.in_p = False
        if any(frame[0] == tag for frame in self.frames):
            while self.frames:
                name, scoped, ignored = self.frames.pop()
                self.scope = max(0, self.scope - int(scoped))
                self.skip = max(0, self.skip - int(ignored))
                if name == tag:
                    break
        if tag == 'h1':
            self.in_title = False

    def handle_data(self, text):
        if self.skip:
            return
        if self.in_title:
            self.title.append(text)
        if self.in_p:
            self.parts.append(text)


def parse_public_article(row, response):
    if 'text/html' not in response.headers.get('Content-Type', '').lower():
        raise ValueError('Not an HTML article')
    if re.search(rb'"isAccessibleForFree"\s*:\s*false|subscribe to (?:continue|unlock) reading', response.content, re.I):
        raise ValueError('Page declares restricted reading access; no article text extracted')
    parser = ArticleParagraphs()
    encoding = response.encoding if getattr(response, 'encoding', None) not in (None, 'ISO-8859-1') else 'utf-8'
    parser.feed(response.content.decode(encoding, errors='replace'))
    title_tokens = set(normalized_headline(row['headline']).split())
    page_tokens = set(normalized_headline(' '.join(parser.title)).split())
    if not page_tokens or len(title_tokens & page_tokens) / max(1, len(title_tokens)) < .5:
        raise ValueError('Article heading does not match; challenge, redirect or changed page possible')
    relevant = [text for text in parser.paragraphs if re.search(r'\b(XRP|XRPL|RLUSD|Ripple)\b', text, re.I)]
    excerpt = clean_excerpt(row['headline'], ' '.join(relevant))
    if len(excerpt.split()) < 30:
        raise ValueError('No substantive matching public article text; no restricted content extracted')
    return excerpt


def fetch_additional(folder, rows, contexts, limit=20, max_requests=40, retry_unavailable=False):
    folder = Path(folder)
    output = folder / 'research'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'additional_context.json'
    old = read_json_rows(path)
    results = {row['id']: row for row in old}
    store = Store(output / 'context_access.sqlite3')
    # Share known source cooldowns/robots policies from the collector without
    # changing its database, checkpoints, corpus or review data.
    collector = folder / 'collection.sqlite3'
    if collector.exists() and not store.db.execute('SELECT 1 FROM hosts LIMIT 1').fetchone():
        with sqlite3.connect(collector) as connection:
            for host, record in connection.execute('SELECT host,record FROM hosts'):
                store.put('hosts', host, json.loads(record), 'host')
        store.db.commit()
    client = PublicClient(store, max_requests)
    attempted = 0
    try:
        candidates = [row for row in rows if not row['label'].strip()
                      and (row['id'] not in results or (retry_unavailable and
                           results[row['id']]['status'] == 'unavailable' and
                           results[row['id']].get('detail', '').startswith(('No substantive', 'Article heading'))))
                      and (not matching(row, contexts.get(row['id'], {})) or
                           not clean_excerpt(row['headline'], contexts.get(row['id'], {}).get('excerpt', '')))]
        # Missing context on product/technical status is the most useful retrieval.
        candidates.sort(key=lambda row: (not bool(re.search(r'launch|integrat|custody|partner|release|security|approval|lending', row['headline'], re.I)), row['id']))
        selected, source_counts = [], {}
        while candidates and len(selected) < limit:
            best = min(candidates, key=lambda row: (source_counts.get(row['source'], 0),
                       not bool(re.search(r'launch|integrat|custody|partner|release|lending', row['headline'], re.I))))
            candidates.remove(best)
            selected.append(best)
            source_counts[best['source']] = source_counts.get(best['source'], 0) + 1
        if path.exists():
            backup(path)
        for row in selected:
            if client.used >= max_requests:
                break
            attempted += 1
            record = dict(id=row['id'], headline=row['headline'], url=row['url'], fetched_at=now(),
                          status='unavailable', excerpt='', article_verified=False,
                          context_kind='PUBLIC_ARTICLE_EXCERPT',
                          record_attempt_count=results.get(row['id'], {}).get('record_attempt_count', 1 if row['id'] in results else 0) + 1)
            try:
                response = client.fetch(dict(url=row['url'], provider='article_context'), {})
                record.update(excerpt=parse_public_article(row, response), status='retrieved_unverified',
                              detail='Matched public page and excerpt only; not independent factual verification.')
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                record['detail'] = str(exc)[:400]
                if isinstance(exc, requests.RequestException):
                    host = urlparse(row['url']).netloc
                    state = store.get('hosts', host, 'host')
                    state['cooldown_until'] = (utc_date(now()) + timedelta(minutes=15)).isoformat()
                    store.put('hosts', host, state, 'host')
                    store.db.commit()
            results[row['id']] = record
            atomic_json(path, list(results.values()))
            print(f'Context {row["source"]}: {record["status"]}', flush=True)
    finally:
        store.db.close()
    return dict(attempted_this_run=attempted,
                http_requests=client.used,
                total_attempted=len(results), retrieved=sum(row['status'] == 'retrieved_unverified' for row in results.values()))
