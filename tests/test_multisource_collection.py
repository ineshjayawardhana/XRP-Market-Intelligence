"""Collection tests use local XML/JSON fixtures and mocked HTTP, never live news."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from ai_labeling import DECISION_FIELDS, LABELS, atomic_csv, decide, read_csv
from collect_multisource_news import (BASE_FIELDS, PublicClient, Store, build_jobs, export_workspace,
                                      gdelt_rows, parse_feed, run_job, timestamp)
from news_label_safety import sha256

OBSERVED = '2026-11-15T12:00:00+00:00'
JOB = dict(provider='fixture', publisher='Fixture Publisher', url='https://fixture.example/feed/', closed=True)


def rss(items=1, pubdate='Fri, 13 Nov 2026 10:30:00 GMT'):
    entries = ''.join(f'''<item><title>XRP ledger transaction development announcement {i}</title>
        <link>https://fixture.example/story/{i}</link><pubDate>{pubdate}</pubDate>
        <description>A publisher summary describes XRP Ledger payment integration and software tools for
        application developers. It reports operational details of the announced wallet service without
        promising investment gains or asserting that token prices must rise.</description></item>''' for i in range(items))
    return ('<rss version="2.0"><channel>' + entries + '</channel></rss>').encode()


class Response:
    def __init__(self, content=b'', status=200, headers=None):
        self.content, self.status_code, self.headers = content, status, headers or {}
        self.text = content.decode('utf-8')


class ParserTests(unittest.TestCase):
    def test_rss_keeps_publication_and_observation_separate(self):
        rows, raw = parse_feed(rss(), JOB, OBSERVED)
        self.assertEqual(raw, 1)
        self.assertEqual(rows[0]['published_at'], '2026-11-13T10:30:00+00:00')
        self.assertEqual(rows[0]['date'], OBSERVED)
        self.assertEqual(rows[0]['availability_at'], OBSERVED)
        self.assertEqual(rows[0]['timestamp_type'], 'PROJECT_OBSERVED_AT')
        self.assertEqual(rows[0]['label'], '')
        self.assertEqual(rows[0]['historical_eligible'], 'False')

    def test_missing_naive_and_date_only_publication_not_invented(self):
        for value in ('', '2026-11-13', '2026-11-13T10:00:00'):
            self.assertEqual(timestamp(value), '')
            row = parse_feed(rss(pubdate=value), JOB, OBSERVED)[0][0]
            self.assertEqual(row['published_at'], '')
            self.assertEqual(row['date'], OBSERVED)

    def test_atom_updated_is_not_publication_time(self):
        content = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <title>XRP Ledger update</title><link href="https://fixture.example/atom"/>
        <updated>2026-11-14T10:00:00Z</updated><summary>Example summary</summary></entry></feed>'''
        row = parse_feed(content, JOB, OBSERVED)[0][0]
        self.assertEqual(row['published_at'], '')
        self.assertEqual(row['article_url'], 'https://fixture.example/atom')

    def test_google_preserves_publisher_and_opaque_article_link(self):
        content = b'''<rss><channel><item><title>XRP Ledger news - Publisher Name</title>
        <link>https://news.google.com/rss/articles/opaque</link><source>Publisher Name</source>
        <description>Headline and source only</description></item></channel></rss>'''
        row = parse_feed(content, dict(JOB, provider='google_news'), OBSERVED)[0][0]
        self.assertEqual(row['headline'], 'XRP Ledger news')
        self.assertEqual(row['publisher'], 'Publisher Name')
        self.assertEqual(row['url'], 'https://news.google.com/rss/articles/opaque')
        self.assertEqual(row['context_kind'], 'AGGREGATOR_SNIPPET')

    def test_future_reported_publication_is_flagged(self):
        row = parse_feed(rss(pubdate='Mon, 16 Nov 2026 12:00:00 GMT'), JOB, OBSERVED)[0][0]
        self.assertEqual(row['publication_time_status'], 'FUTURE_REPORTED_TIMESTAMP')
        self.assertEqual(row['date'], OBSERVED)

    def test_dtd_and_html_are_refused(self):
        for content in (b'<!DOCTYPE rss><rss/>', b'<html><body>Challenge</body></html>'):
            with self.assertRaises(ValueError):
                parse_feed(content, JOB, OBSERVED)

    def test_gdelt_first_seen_is_never_misrepresented_as_publication(self):
        content = json.dumps(dict(articles=[dict(title='XRP Ledger test news',
                             url='https://fixture.example/gdelt', domain='fixture.example',
                             seendate='20261113T103000Z')])).encode()
        row = gdelt_rows(content, JOB, OBSERVED)[0][0]
        self.assertEqual(row['gdelt_first_seen_at'], '2026-11-13T10:30:00+00:00')
        self.assertEqual(row['published_at'], '')
        self.assertEqual(row['date'], OBSERVED)

    def test_jobs_use_stable_dates_and_public_endpoints(self):
        first = build_jobs(84, '2026-11-15T00:00:00+00:00')
        second = build_jobs(84, '2026-11-15T00:00:00+00:00')
        self.assertEqual(first, second)
        self.assertTrue(any(job['provider'] == 'google_news' for job in first))
        self.assertTrue(any(job['provider'] == 'gdelt' for job in first))
        self.assertTrue(all(job['url'].startswith('https://') for job in first))

    def test_later_day_reuses_closed_historical_windows(self):
        first = {job['url'] for job in build_jobs(84, '2026-11-15T00:00:00+00:00') if job['provider'] == 'gdelt'}
        second = {job['url'] for job in build_jobs(84, '2026-11-16T00:00:00+00:00') if job['provider'] == 'gdelt'}
        self.assertGreaterEqual(len(first & second), len(first) - 2)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.output = self.root / 'original', self.root / 'collected'
        self.data.mkdir()
        self.output.mkdir()
        self.rows = [dict(id=f'legacy-{i}', date='2026-09-21T12:00:00+00:00',
                          headline=f'XRP distinct topic{i} event{i} wallet{i}', source='legacy.example',
                          url=f'https://legacy.example/{i}', timestamp_type='GDELT_FIRST_SEEN',
                          label='Neutral_Market_News' if i == 0 else 'SKIP' if i == 1 else '')
                     for i in range(53)]
        atomic_csv(self.data / 'xrp_real_labeled.csv', BASE_FIELDS, self.rows)
        self.digest = sha256(self.data / 'xrp_real_labeled.csv')
        self.store = Store(self.output / 'collection.sqlite3')
        self.addCleanup(self.store.db.close)
        self.store.seed(self.data / 'xrp_real_labeled.csv')
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_seed_is_idempotent_and_preserves_all_53(self):
        self.store.seed(self.data / 'xrp_real_labeled.csv')
        self.assertEqual(len(self.store.rows()), 53)
        self.assertEqual(sha256(self.data / 'xrp_real_labeled.csv'), self.digest)

    def test_canonical_url_and_similar_headline_deduplicate(self):
        row = parse_feed(rss(), JOB, OBSERVED)[0][0]
        self.assertEqual(self.store.add(row, 'feed1'), 'added')
        duplicate = dict(row, url=row['url'] + '?utm_source=second')
        self.assertEqual(self.store.add(duplicate, 'feed2'), 'duplicate')
        duplicate = dict(row, headline=row['headline'] + ' | Publisher',
                         url='https://another.example/syndicated', canonical_url='another.example/syndicated?')
        self.assertEqual(self.store.add(duplicate, 'feed3'), 'duplicate')
        self.assertEqual(len(self.store.rows()), 54)

    def test_duplicate_does_not_overwrite_legacy_labels_or_dates(self):
        old = self.store.rows()[0]
        row = dict(parse_feed(rss(), JOB, OBSERVED)[0][0], headline=old['headline'],
                   url=old['url'], canonical_url=old['canonical_url'])
        self.store.add(row, 'feed')
        after = self.store.rows()[0]
        for key in BASE_FIELDS:
            self.assertEqual(after[key], old[key])
        self.assertEqual(after['published_at'], '2026-11-13T10:30:00+00:00')

    def test_syndicated_copy_does_not_supply_original_publication_time(self):
        old = self.store.rows()[0]
        row = dict(parse_feed(rss(), JOB, OBSERVED)[0][0], headline=old['headline'])
        self.store.add(row, 'syndicated-feed')
        after = self.store.get('articles', old['id'], 'id')
        self.assertEqual(after['published_at'], '')

    def test_robots_429_sets_host_cooldown(self):
        session = Mock()
        session.get.return_value = Response(status=429, headers={'Retry-After': '1800'})
        client = PublicClient(self.store, session=session)
        run_job(self.store, client, JOB, 100)
        run_job(self.store, client, dict(JOB, url=JOB['url'] + '?other=1'), 100)
        self.assertEqual(session.get.call_count, 1)

    def test_checkpoint_pending_rows_resume_without_network(self):
        session = Mock()
        session.get.side_effect = [Response(b'User-agent: *\nAllow: /'), Response(rss(2), headers={'ETag': 'v1'})]
        client = PublicClient(self.store, session=session)
        run_job(self.store, client, JOB, 54)
        self.assertEqual(len(self.store.rows()), 54)
        calls = session.get.call_count
        run_job(self.store, client, JOB, 55)
        self.assertEqual(len(self.store.rows()), 55)
        self.assertEqual(session.get.call_count, calls)
        run_job(self.store, client, JOB, 100)
        self.assertEqual(session.get.call_count, calls)

    def test_429_respects_retry_after_and_prevents_other_window_requests(self):
        session = Mock()
        session.get.side_effect = [Response(b'User-agent: *\nAllow: /'), Response(status=429, headers={'Retry-After': '7200'})]
        client = PublicClient(self.store, session=session)
        job = dict(JOB, provider='gdelt')
        self.assertEqual(run_job(self.store, client, job, 100), 'failed')
        calls = session.get.call_count
        self.assertEqual(run_job(self.store, client, dict(job, url=job['url'] + '?window=2'), 100), 'failed')
        self.assertEqual(session.get.call_count, calls)
        self.assertIn('cooldown_until', self.store.get('hosts', 'fixture.example', 'host'))
        self.assertEqual(len(self.store.rows()), 53)

    def test_robots_restriction_never_fetches_feed(self):
        session = Mock()
        session.get.return_value = Response(b'User-agent: *\nDisallow: /')
        client = PublicClient(self.store, session=session)
        self.assertEqual(run_job(self.store, client, JOB, 100), 'failed')
        self.assertEqual(session.get.call_count, 1)

    def test_failed_source_and_timeout_preserve_existing_data(self):
        session = Mock()
        session.get.side_effect = requests.Timeout('offline fixture')
        client = PublicClient(self.store, session=session)
        run_job(self.store, client, JOB, 100)
        self.assertEqual(len(self.store.rows()), 53)
        self.assertEqual(sha256(self.data / 'xrp_real_labeled.csv'), self.digest)
        run_job(self.store, client, dict(JOB, url=JOB['url'] + '?next=1'), 100)
        self.assertEqual(session.get.call_count, 1)

    def test_request_budget_is_enforced(self):
        client = PublicClient(self.store, budget=1, session=Mock())
        client.used = 1
        with self.assertRaisesRegex(RuntimeError, 'BUDGET'):
            client.request(JOB['url'])

    def test_not_modified_sends_conditional_headers(self):
        import hashlib
        key = hashlib.sha256(JOB['url'].encode()).hexdigest()
        self.store.put('jobs', key, dict(provider='fixture', url=JOB['url'], etag='v1', status='complete'))
        self.store.db.commit()
        session = Mock()
        session.get.side_effect = [Response(b'User-agent: *\nAllow: /'), Response(status=304)]
        client = PublicClient(self.store, session=session)
        self.assertEqual(run_job(self.store, client, JOB, 100), 'not_modified')
        self.assertEqual(session.get.call_args.kwargs['headers']['If-None-Match'], 'v1')
        self.assertEqual(len(self.store.rows()), 53)

    def test_normal_redirect_checks_robots_at_destination(self):
        session = Mock()
        session.get.side_effect = [Response(b'User-agent: *\nAllow: /'),
                                   Response(status=301, headers={'Location': 'https://canonical.example/rss'}),
                                   Response(b'User-agent: *\nDisallow: /')]
        client = PublicClient(self.store, session=session)
        self.assertEqual(run_job(self.store, client, JOB, 100), 'failed')
        self.assertEqual(session.get.call_count, 3)
        self.assertEqual(session.get.call_args.args[0], 'https://canonical.example/robots.txt')

    def test_public_feed_404_stops_additional_pagination(self):
        session = Mock()
        session.get.side_effect = [Response(b'User-agent: *\nAllow: /'), Response(status=404)]
        client = PublicClient(self.store, session=session)
        first = dict(JOB, page=2, base_feed=JOB['url'], url=JOB['url'] + '?paged=2')
        run_job(self.store, client, first, 100)
        calls = session.get.call_count
        self.assertEqual(run_job(self.store, client, dict(first, page=3, url=JOB['url'] + '?paged=3'), 100),
                         'feed_pagination_exhausted')
        self.assertEqual(session.get.call_count, calls)

    def test_workspace_keeps_human_and_skip_and_adds_publisher_context(self):
        row = parse_feed(rss(), JOB, OBSERVED)[0][0]
        self.store.add(row, 'feed')
        exported = export_workspace(self.store, self.data, self.output)
        self.assertEqual(len(exported), 54)
        for original, saved in zip(self.rows, exported):
            for key in BASE_FIELDS:
                self.assertEqual(original[key], saved[key])
        self.assertEqual(sha256(self.data / 'xrp_real_labeled.csv'), self.digest)
        contexts = json.loads((self.output / 'xrp_article_context.json').read_text())
        self.assertEqual(len(contexts), 1)
        self.assertFalse(contexts[0]['article_verified'])
        self.assertEqual(exported[-1]['label'], '')
        self.assertEqual(exported[-1]['date'], OBSERVED)

    def test_google_snippet_is_not_article_context(self):
        row = parse_feed(rss(), dict(JOB, provider='google_news'), OBSERVED)[0][0]
        self.store.add(row, 'feed')
        export_workspace(self.store, self.data, self.output)
        self.assertEqual(json.loads((self.output / 'xrp_article_context.json').read_text()), [])

    def test_existing_workspace_human_review_survives_resume(self):
        export_workspace(self.store, self.data, self.output)
        decision = decide(dict(self.rows[2], suggested_label='Real_Adoption'), 'a', 'Test Reviewer')
        path = self.output / 'xrp_human_reviews.csv'
        atomic_csv(path, DECISION_FIELDS, [decision])
        before = path.read_bytes()
        export_workspace(self.store, self.data, self.output)
        self.assertEqual(path.read_bytes(), before)
        prior_fields, prior = read_csv(self.output / 'xrp_real_labeled.csv')
        prior[2]['label'] = 'Real_Adoption'
        atomic_csv(self.output / 'xrp_real_labeled.csv', prior_fields, prior)
        export_workspace(self.store, self.data, self.output)
        self.assertEqual(read_csv(self.output / 'xrp_real_labeled.csv')[1][2]['label'], 'Real_Adoption')


if __name__ == '__main__':
    unittest.main()
