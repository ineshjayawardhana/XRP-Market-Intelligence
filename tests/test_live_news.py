"""Feed, persistence, scheduler and UI fixtures; not real news evidence."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import dashboard_data
from live_news import NewsPoller, NewsStore, load_rows, monitoring, refresh
from news_label_safety import sha256

NOW = '2026-11-17T12:00:00+00:00'
JOB = dict(provider='fixture', publisher='Fixture publisher', url='https://fixture.example/feed')


def rss(title='XRP fixture network update', url='https://fixture.example/article', date='Tue, 17 Nov 2026 11:00:00 GMT'):
    return f'<rss><channel><item><title>{title}</title><link>{url}</link><pubDate>{date}</pubDate><description>XRPL validators published a routine protocol governance update.</description></item></channel></rss>'.encode()


def assessment(entry):
    return dict(label='Neutral_Market_News', score=.75, score_type='Heuristic score · uncalibrated',
        classifier_backend='fixture rubric', model_provenance='Fixture only, not a trained model', reasoning='Fixture', article_verified=False)


class Client:
    response = None
    calls = []
    def __init__(self, store, budget):
        self.store = store
    def fetch(self, job, previous):
        type(self).calls.append((job, previous))
        return type(self).response


class LiveNewsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root/'cache'
        self.cache.mkdir()
        self.env = patch.dict(os.environ, {'XRP_DASHBOARD_CACHE_DIR':str(self.cache),
            'XRP_LIVE_NEWS_DIR':str(self.root/'live'), 'XRP_DASHBOARD_OFFLINE':'0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        Client.response = Mock(status_code=200, content=rss(), headers={'ETag':'v1'})
        Client.calls = []
        self.clock = NOW

    def poll(self, force=False, classify=assessment, jobs=None):
        return refresh(self.root, force=force, sources=jobs or [JOB], client_factory=Client,
                       classifier=classify, clock=lambda:self.clock)

    def test_download_save_display_publication_sort_and_provenance(self):
        result = self.poll()
        self.assertEqual(result['added'],1)
        frame, errors = dashboard_data.load_news(self.root)
        self.assertFalse(errors)
        self.assertEqual(frame.iloc[0].label_status,'AI provisional')
        self.assertEqual(frame.iloc[0].published_at,pd.Timestamp('2026-11-17T11:00:00Z'))
        self.assertEqual(frame.iloc[0].first_seen_at,pd.Timestamp(NOW))
        self.assertEqual(frame.iloc[0].collected_at,pd.Timestamp(NOW))
        self.assertEqual(frame.iloc[0].classifier_backend,'fixture rubric')
        row = load_rows(self.root)[0]
        self.assertEqual(row['verification_status'],'AI_PROVISIONAL')
        self.assertFalse(row['article_verified'])

    def test_scheduled_poll_does_not_fetch_before_five_minutes(self):
        self.poll()
        self.clock='2026-11-17T12:04:59+00:00'
        self.assertTrue(self.poll()['skipped'])
        self.assertEqual(len(Client.calls),1)
        self.clock='2026-11-17T12:05:00+00:00'
        self.poll()
        self.assertEqual(len(Client.calls),2)

    def test_manual_refresh_bypasses_normal_schedule_but_not_repeat_throttle(self):
        self.poll()
        self.clock='2026-11-17T12:00:10+00:00'
        self.poll(True)
        self.assertEqual(len(Client.calls),1)
        self.clock='2026-11-17T12:01:00+00:00'
        self.poll(True)
        self.assertEqual(len(Client.calls),2)

    def test_repeated_feeds_do_not_reclassify_duplicate_or_change_first_seen(self):
        classifier=Mock(side_effect=assessment)
        self.poll(classify=classifier)
        original=load_rows(self.root)[0]
        self.clock='2026-11-17T12:06:00+00:00'
        self.poll(classify=classifier)
        row=load_rows(self.root)[0]
        self.assertEqual(classifier.call_count,1)
        self.assertEqual(row['first_seen_at'],original['first_seen_at'])
        self.assertEqual(row['observed_at'],original['observed_at'])
        self.assertEqual(row['collected_at'],self.clock)
        self.assertEqual(len(load_rows(self.root)),1)
        # The older compatibility cache must not roll the displayed clock back.
        (self.cache/'public_news.json').write_text(json.dumps(dict(items=[original])))
        frame,_=dashboard_data.load_news(self.root)
        self.assertEqual(frame.iloc[0].collected_at,pd.Timestamp(self.clock))

    def test_304_keeps_previous_publication_and_conditional_headers(self):
        self.poll()
        self.clock='2026-11-17T12:06:00+00:00'
        Client.response=Mock(status_code=304, headers={})
        self.poll()
        state=NewsStore(self.root).get('sources','fixture')
        self.assertEqual(state['latest_publication_at'],'2026-11-17T11:00:00+00:00')
        self.assertEqual(state['articles_returned'],0)
        self.assertEqual(Client.calls[-1][1]['etag'],'v1')

    def test_failed_source_does_not_stop_others_and_backoff_blocks_manual(self):
        original=Client.fetch
        def failing(client, job, previous):
            if job['provider']=='bad':
                Client.calls.append((job,previous))
                raise RuntimeError('HTTP 429 fixture rate limit')
            return original(client,job,previous)
        bad=dict(JOB,provider='bad')
        with patch.object(Client,'fetch',failing):
            result=self.poll(jobs=[bad,JOB])
            self.assertEqual(result['added'],1)
            self.assertTrue(result['errors'])
            self.clock='2026-11-17T12:01:00+00:00'
            self.poll(force=True,jobs=[bad,JOB])
        self.assertEqual(sum(j['provider']=='bad' for j,_ in Client.calls),1)

    def test_invalid_xml_never_advances_last_success(self):
        self.poll()
        self.clock='2026-11-17T12:06:00+00:00'
        Client.response=Mock(status_code=200,content=b'<html>Access page</html>',headers={})
        self.poll()
        state=NewsStore(self.root).get('sources','fixture')
        self.assertEqual(state['last_success_at'],NOW)
        self.assertIn('RSS/Atom',state['error'])

    def test_missing_publication_is_never_substituted_with_collection_time(self):
        Client.response.content=rss(date='')
        self.poll()
        frame,_=dashboard_data.load_news(self.root)
        self.assertTrue(pd.isna(frame.iloc[0].published_at))
        self.assertEqual(frame.iloc[0].first_seen_at,pd.Timestamp(NOW))

    def test_mixed_fractional_and_timezone_formats_sort_correctly(self):
        store=NewsStore(self.root)
        for i,date in enumerate(('2026-11-16T11:00:00+00:00','2026-11-17T11:00:00.123456+00:00','2026-11-17T16:00:00+05:30')):
            store.add(dict(id=str(i), headline=f'XRP unique fixture topic {i}',url=f'https://fixture.example/{i}',published_at=date,
                observed_at=NOW,first_seen_at=NOW,collected_at=NOW,**assessment({})), 'fixture', NOW)
        frame,_=dashboard_data.load_news(self.root,as_of=NOW)
        self.assertEqual(list(frame.id),['1','2','0'])

    def test_legacy_cache_import_preserves_all_rows_and_original_file(self):
        rows=[dict(id=str(i),headline=f'XRP legacy fixture event {i}',url=f'https://fixture.example/{i}',
            published_at='2026-11-15T00:00:00+00:00',observed_at='2026-11-15T01:00:00+00:00', label='Hype_Speculation') for i in range(510)]
        path=self.cache/'public_news.json'
        path.write_text(json.dumps(dict(items=rows)))
        digest=sha256(path)
        self.assertEqual(len(NewsStore(self.root).rows()),510)
        self.assertEqual(len(NewsStore(self.root).rows()),510)
        self.assertEqual(sha256(path),digest)

    def test_classifier_failure_saves_needs_review_without_losing_article(self):
        self.poll(classify=Mock(side_effect=RuntimeError('fixture model unavailable')))
        row=load_rows(self.root)[0]
        self.assertEqual(row['label'],'REVIEW_REQUIRED')
        self.assertEqual(row['label_status'],'Needs review')
        self.assertEqual(row['verification_status'],'AI_PROVISIONAL')

    def test_cross_source_duplicates_keep_original_publication_and_label(self):
        self.poll()
        Client.response.content=rss(url='https://other.example/syndication',date='Tue, 17 Nov 2026 11:30:00 GMT')
        self.clock='2026-11-17T12:06:00+00:00'
        self.poll(jobs=[dict(JOB,provider='other')])
        rows=load_rows(self.root)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['published_at'],'2026-11-17T11:00:00+00:00')


class SchedulerTests(unittest.TestCase):
    def test_background_polls_without_page_reruns_and_manual_works_while_paused(self):
        calls=[]
        called=threading.Event()
        def run(root,force=False):
            calls.append(force)
            if len(calls)>=2:
                called.set()
        poller=NewsPoller('.',interval=.06,refresh_fn=run)
        with patch.dict(os.environ,{'XRP_DASHBOARD_OFFLINE':'0'}):
            poller.start()
            self.assertTrue(called.wait(1))
            poller.pause()
            time.sleep(.08)
            paused=len(calls)
            time.sleep(.08)
            self.assertEqual(len(calls),paused)
            poller.request()
            time.sleep(.1)
            self.assertEqual(len(calls),paused+1)
            self.assertTrue(calls[-1])
            poller.stop()
            poller.thread.join(1)
        self.assertFalse(poller.thread.is_alive())


class NewsUITests(unittest.TestCase):
    def test_stale_warning_source_status_manual_refresh_and_new_sql_rows_display(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.dict(os.environ,{'XRP_DASHBOARD_ROOT':temp,'XRP_DASHBOARD_CACHE_DIR':str(root/'cache'),
                    'XRP_LIVE_NEWS_DIR':str(root/'live'),'XRP_DASHBOARD_OFFLINE':'1'}):
                store=NewsStore(root)
                clock=pd.Timestamp.now(tz='UTC')
                def add(title, date, identifier):
                    store.add(dict(id=identifier,headline=title,url='https://fixture.example/'+identifier,
                        published_at=date,observed_at=clock.isoformat(),first_seen_at=clock.isoformat(),
                        collected_at=clock.isoformat(),label_status='AI provisional',origin='Fixture',**assessment({})), 'fixture',clock.isoformat())
                add('XRP old fixture headline',(clock-pd.Timedelta(days=3)).isoformat(),'old')
                st.cache_data.clear()
                app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
                app.radio[0].set_value('News intelligence').run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(any('News is stale' in w.value for w in app.warning))
                self.assertTrue(any('News Source Status' in h.value for h in app.subheader))
                def manual(root):
                    add('XRP newly collected fixture headline',(clock-pd.Timedelta(minutes=30)).isoformat(),'new')
                with patch('live_news_ui.request_news_refresh',side_effect=manual) as request:
                    next(b for b in app.button if b.label=='Refresh data now').click().run(timeout=30)
                self.assertTrue(request.called)
                self.assertFalse(app.exception)
                self.assertTrue(any('newly collected fixture headline' in str(d.value) for d in app.dataframe))
                self.assertFalse(any('News is stale' in w.value for w in app.warning))
                # An independently saved row appears without clearing Streamlit's status cache.
                add('XRP independently saved fixture article',(clock-pd.Timedelta(minutes=5)).isoformat(),'next')
                app.run(timeout=30)
                self.assertTrue(any('independently saved fixture article' in str(d.value) for d in app.dataframe))
                st.cache_data.clear()


if __name__=='__main__':
    unittest.main()
