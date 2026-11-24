"""Shared provenance, time and duplicate checks; no network or model calls."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc_date(value):
    """Refuse ambiguous local timestamps rather than silently assuming UTC."""
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return date.astimezone(timezone.utc) if date.tzinfo else None
    except (ValueError, TypeError, AttributeError):
        return None


def normalized_headline(value):
    value = value.split('|')[0]
    value = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(re.findall(r'[a-z0-9]+', value))


def canonical_url(value):
    parts = urlsplit(value.strip())
    if not parts.hostname:
        return ''
    query = [(key, val) for key, val in parse_qsl(parts.query)
             if not key.lower().startswith('utm_') and key.lower() not in
             {'source', 'ref', 'fbclid', 'gclid'}]
    return parts.hostname.lower().removeprefix('www.') + parts.path.rstrip('/') + '?' + urlencode(sorted(query))


def human_origin_allowed(row):
    """Existing manual labels are legacy human annotations, never factual verification."""
    for key in ('label_status', 'label_verification_status', 'verification_status'):
        if row.get(key, '').upper() == 'AI_PROVISIONAL':
            return False
    origin = row.get('label_origin', '').lower()
    if origin not in ('', 'human_manual', 'human_verified', 'human_reviewed_ai', 'human_manual_legacy'):
        return False
    return origin != 'human_reviewed_ai' or bool(
        row.get('human_reviewed_by', '').strip() and utc_date(row.get('human_reviewed_at', '')))


def duplicate_groups(rows):
    """Conservative connected components, shared across all chronological partitions.

    Match IDs, canonical URLs, normalized/publisher-stripped titles, near titles,
    and substantial overlapping excerpt shingles. This cannot identify every
    paraphrase; detected groups never cross train/validation/test boundaries.
    """
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[find(right)] = find(left)

    titles = [normalized_headline(row['headline']) for row in rows]
    tokens = [set(title.split()) for title in titles]
    urls = [canonical_url(row.get('url', '')) for row in rows]
    excerpts = [re.findall(r'[a-z0-9]+', row.get('context_excerpt', '').lower()) for row in rows]
    shingles = [{tuple(words[i:i + 6]) for i in range(len(words) - 5)} if len(words) >= 50 else set()
                for words in excerpts]
    for right in range(len(rows)):
        for left in range(right):
            same = (rows[left]['id'] == rows[right]['id'] or
                    bool(urls[left] and urls[left] == urls[right]) or
                    bool(titles[left] and titles[left] == titles[right]))
            if not same and min(len(tokens[left]), len(tokens[right])) >= 6:
                same = len(tokens[left] & tokens[right]) / len(tokens[left] | tokens[right]) >= .85
            if not same and shingles[left] and shingles[right]:
                same = len(shingles[left] & shingles[right]) / len(shingles[left] | shingles[right]) >= .65
            if same:
                union(left, right)
    components = {}
    for index, row in enumerate(rows):
        components.setdefault(find(index), []).append(row)
    return list(components.values())
