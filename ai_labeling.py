"""Auditable, human-in-the-loop XRP labeling. Run with --help.

No model training, paid APIs, or automatic promotion of AI suggestions.
The human decision ledger is authoritative; suggestion review_status is a view.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ('Real_Adoption', 'Hype_Speculation', 'Negative_FUD', 'Neutral_Market_News')
REVIEW = 'REVIEW_REQUIRED'
SUGGESTION_FIELDS = ['id', 'headline', 'suggested_label', 'confidence_estimate',
                     'reasoning', 'needs_manual_review', 'review_status', 'url',
                     'context_status', 'article_verified', 'suggestion_method']
DECISION_FIELDS = ['id', 'headline', 'url', 'review_status', 'human_label',
                   'reviewed_by', 'reviewed_at', 'suggested_label']


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if len(set(fields)) != len(fields):
        raise ValueError(f'Duplicate CSV headers: {path}')
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Malformed CSV: {path}')
    if 'id' not in fields or any(not row['id'].strip() for row in rows):
        raise ValueError(f'Missing/empty id: {path}')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate ids: {path}')
    return fields, rows


def atomic_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def backup(path):
    path = Path(path)
    folder = path.parent / 'backups'
    folder.mkdir(exist_ok=True)
    target = folder / (path.stem + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + path.suffix)
    shutil.copy2(path, target)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
        raise OSError('Backup hash mismatch')
    return target


@contextmanager
def workflow_lock(folder):
    """Prevent two workflow instances from losing each other's decisions."""
    path = Path(folder) / '.ai_labeling.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Another review is open. If it crashed, remove {path} and retry.') from None
    try:
        os.close(fd)
        yield
    finally:
        path.unlink()


def dataset(path):
    fields, rows = read_csv(path)
    required = {'id', 'headline', 'url', 'label'}
    if not required.issubset(fields):
        raise ValueError(f'Missing dataset columns: {required - set(fields)}')
    return fields, rows


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.parts = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.depth += 1
            if self.depth == 1:
                self.parts = []

    def handle_endtag(self, tag):
        if tag == 'p' and self.depth:
            self.depth -= 1
            if not self.depth:
                text = re.sub(r'\s+', ' ', ' '.join(self.parts)).strip()
                if len(text) > 70:
                    self.paragraphs.append(text)

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def fetch_one(row):
    import requests  # Already in requirements.txt; imported only for network command.
    result = {key: row[key] for key in ('id', 'headline', 'url')}
    result.update(fetched_at=now(), status='unavailable', excerpt='', article_verified=False)
    if not row['url'].startswith(('https://', 'http://')):
        result['detail'] = 'Missing HTTP(S) URL'
        return result
    try:
        with requests.get(row['url'], timeout=(5, 10), stream=True,
                          headers={'User-Agent': 'XRP-Academic-Label-Review/1.0'}) as response:
            response.raise_for_status()
            if 'text/html' not in response.headers.get('Content-Type', ''):
                raise ValueError('Not an HTML article')
            chunks, size = [], 0
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > 2_000_000:
                    break
                chunks.append(chunk)
            parser = Paragraphs()
            parser.feed(b''.join(chunks).decode(response.encoding or 'utf-8', errors='replace'))
            relevant = [p for p in parser.paragraphs if re.search(r'\b(XRP|XRPL|RLUSD)\b', p, re.I)]
            if not relevant:
                raise ValueError('No relevant article paragraphs; blocked or unsuitable page')
            result.update(status='retrieved_unverified',
                          excerpt=' '.join(' '.join(relevant).split()[:180]),
                          final_url=response.url,
                          detail='Candidate context only. Review page relevance; no factual verification.')
    except (requests.RequestException, ValueError) as exc:
        result['detail'] = str(exc)[:250]
    return result


def fetch_context(source, output):
    _, rows = dataset(source)
    pending = [row for row in rows if not row['label'].strip()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        context = list(pool.map(fetch_one, pending))
    Path(output).write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Article attempts:', dict(Counter(row['status'] for row in context)))


def matching(left, right):
    return all(left.get(key) == right.get(key) for key in ('id', 'headline', 'url'))


def load_decisions(path):
    if not Path(path).exists():
        return []
    fields, rows = read_csv(path)
    if not set(DECISION_FIELDS).issubset(fields):
        raise ValueError('Invalid human decision ledger')
    for row in rows:
        status = row['review_status']
        label = row['human_label']
        if (status not in ('accepted', 'corrected', 'rejected') or
                (status == 'rejected' and label) or
                (status != 'rejected' and label not in LABELS) or
                not row['reviewed_by'].strip() or not row['reviewed_at'].strip()):
            raise ValueError('Invalid human decision')
    return rows


def sync_status(suggestions, decisions):
    by_id = {row['id']: row for row in decisions}
    for row in suggestions:
        decision = by_id.get(row['id'])
        row['review_status'] = decision['review_status'] if decision and matching(row, decision) else 'pending'
        row['needs_manual_review'] = str(row['review_status'] == 'pending')


def generate(source, output, proposals, decisions):
    _, rows = dataset(source)
    supplied = json.loads(Path(proposals).read_text(encoding='utf-8')) if Path(proposals).exists() else []
    if len({row['id'] for row in supplied}) != len(supplied):
        raise ValueError('Duplicate proposal ids')
    proposed = {row['id']: row for row in supplied}
    previous = read_csv(output)[1] if Path(output).exists() else []
    human_decisions = load_decisions(decisions)
    reviewed_ids = {row['id'] for row in human_decisions}
    # Retain historical suggestions, including reviewed/exported ones, on reruns.
    existing = {row['id']: row for row in previous}
    for row in rows:
        if row['label'].strip():
            continue
        old = existing.get(row['id'])
        if old and row['id'] in reviewed_ids:
            # Preserve the reviewed assessment even if source content later changes.
            # A changed article must receive a new id before another decision.
            continue
        if old and matching(row, old):
            continue
        proposal = proposed.get(row['id'], {})
        if not matching(row, proposal):
            proposal = dict(suggested_label=REVIEW, confidence_estimate=0,
                            reasoning='No matching AI proposal. Article not verified; headline only.',
                            context_status='not_checked', suggestion_method='no_proposal')
        label = proposal['suggested_label']
        score = float(proposal['confidence_estimate'])
        if label not in (*LABELS, REVIEW) or not 0 <= score <= 1:
            raise ValueError('Invalid suggestion label or heuristic score')
        existing[row['id']] = dict(id=row['id'], headline=row['headline'],
                                  suggested_label=label, confidence_estimate=score,
                                  reasoning=proposal['reasoning'], needs_manual_review='True',
                                  review_status='pending', url=row['url'],
                                  context_status=proposal.get('context_status', 'not_checked'),
                                  article_verified='False',
                                  suggestion_method=proposal.get('suggestion_method', 'AI_assisted'))
    suggestions = list(existing.values())
    sync_status(suggestions, human_decisions)
    if Path(output).exists():
        backup(output)
    atomic_csv(output, SUGGESTION_FIELDS, suggestions)
    print('Suggestions:', len(suggestions), dict(Counter(r['suggested_label'] for r in suggestions)))
    print('All confidence estimates are heuristic scores, NOT calibrated probabilities.')


def decide(suggestion, action, reviewer):
    if not reviewer.strip():
        raise ValueError('Reviewer name is required')
    if action == 'a':
        if suggestion['suggested_label'] not in LABELS:
            raise ValueError('REVIEW_REQUIRED cannot be accepted. Choose 1-4 or reject.')
        label, status = suggestion['suggested_label'], 'accepted'
    elif action in ('1', '2', '3', '4'):
        label, status = LABELS[int(action) - 1], 'corrected'
    elif action == 'r':
        label, status = '', 'rejected'
    else:
        raise ValueError('Use a, 1-4, r, d, or q')
    return dict(id=suggestion['id'], headline=suggestion['headline'], url=suggestion['url'],
                review_status=status, human_label=label, reviewed_by=reviewer.strip(),
                reviewed_at=now(), suggested_label=suggestion['suggested_label'])


def review(source, suggestions_path, decisions_path, reviewer, count=50):
    _, suggestions = read_csv(suggestions_path)
    decisions = load_decisions(decisions_path)
    reviewed_ids = {row['id'] for row in decisions}
    sync_status(suggestions, decisions)
    _, rows = dataset(source)
    current = {row['id']: row for row in rows}
    pending = [row for row in suggestions if row['review_status'] == 'pending'
               and row['id'] not in reviewed_ids
               and row['id'] in current and matching(row, current[row['id']])
               and not current[row['id']]['label'].strip()]
    print('Labels describe content, not truth or future price direction.')
    print('1=Real_Adoption 2=Hype_Speculation 3=Negative_FUD 4=Neutral_Market_News')
    print('a=accept 1-4=correct r=reject d=defer q=quit. Scores are heuristic, not probabilities.')
    print('Ordinary price declines are market reporting, not automatically Negative_FUD.')
    if Path(decisions_path).exists():
        backup(decisions_path)
    for number, row in enumerate(pending[:count], 1):
        print(f'\n[{number}/{min(count, len(pending))}] {row["id"]}\n{row["headline"]}\n{row["url"]}')
        print(f'Suggestion: {row["suggested_label"]} | heuristic score: {row["confidence_estimate"]}')
        print(row['reasoning'])
        while True:
            try:
                action = input('a / 1-4 / r / d / q > ').strip().lower()
            except EOFError:
                return
            if action == 'q':
                return
            if action == 'd':
                break
            try:
                decision = decide(row, action, reviewer)
            except ValueError as exc:
                print(exc)
                continue
            decisions.append(decision)
            atomic_csv(decisions_path, DECISION_FIELDS, decisions)
            sync_status(suggestions, decisions)
            atomic_csv(suggestions_path, SUGGESTION_FIELDS, suggestions)
            break
    print('Review saved. Dataset unchanged. Use export-reviewed when ready to promote human decisions.')


def export_reviewed(source, decisions_path):
    fields, rows = dataset(source)
    decisions = {row['id']: row for row in load_decisions(decisions_path)}
    changes = []
    for row in rows:
        decision = decisions.get(row['id'])
        if (not row['label'].strip() and decision and matching(row, decision)
                and decision['review_status'] in ('accepted', 'corrected')):
            changes.append((row, decision))
    if not changes:
        print('No new human decisions to export. Existing labels preserved.')
        return 0
    for column in ('label_origin', 'human_reviewed_by', 'human_reviewed_at'):
        if column not in fields:
            fields.append(column)
        for row in rows:
            row.setdefault(column, '')
    for row, decision in changes:
        row.update(label=decision['human_label'], label_origin='human_reviewed_ai',
                   human_reviewed_by=decision['reviewed_by'], human_reviewed_at=decision['reviewed_at'])
    print('Dataset backup:', backup(source))
    atomic_csv(source, fields, rows)
    print('Exported human decisions:', len(changes))
    return len(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch-context', 'generate', 'review', 'export-reviewed', 'batch', 'batch-review'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--proposals', type=Path, help='JSON proposals, bound to exact id/headline/url')
    parser.add_argument('--reviewer', default='', help='Required for review')
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--refresh-missing-context', action='store_true', help='Batch: retry missing/new article context')
    parser.add_argument('--min-confidence', type=float, default=.75, help='Batch heuristic threshold, not calibrated probability')
    parser.add_argument('--decisions-file', type=Path, help='Batch human decisions CSV with accept/correct/reject/defer actions')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if not 0 <= args.min_confidence <= 1:
        parser.error('--min-confidence must be between 0 and 1')
    if args.command in ('review', 'batch-review') and not args.reviewer.strip():
        parser.error('--reviewer is required for human decisions')
    if args.command == 'batch-review' and not args.decisions_file:
        parser.error('--decisions-file is required for batch review')
    folder = args.data_dir
    source = folder / 'xrp_real_labeled.csv'
    suggestions = folder / 'xrp_ai_suggestions.csv'
    decisions = folder / 'xrp_human_reviews.csv'
    with workflow_lock(folder):
        # Always preserve the dataset before starting a workflow operation.
        print('Dataset backup:', backup(source))
        if args.command == 'fetch-context':
            fetch_context(source, folder / 'xrp_article_context.json')
        elif args.command == 'generate':
            generate(source, suggestions, args.proposals or folder / 'xrp_ai_proposals.json', decisions)
        elif args.command == 'review':
            review(source, suggestions, decisions, args.reviewer, args.count)
        elif args.command == 'export-reviewed':
            export_reviewed(source, decisions)
        elif args.command == 'batch':
            from batch_news_labeling import batch
            batch(folder, args.refresh_missing_context, args.min_confidence)
        else:
            from batch_news_labeling import batch_review
            batch_review(folder, args.decisions_file, args.reviewer)


if __name__ == '__main__':
    main()
