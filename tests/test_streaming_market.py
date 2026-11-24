"""Deterministic public-stream fixtures, never recorded research evidence."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from market_intelligence import intelligence_chart, market_outlook, slst
from streaming_market import INTERVALS, MarketStream, model_candles, quote_context

NOW = pd.Timestamp('2026-11-16T08:00:30Z')


def ms(value):
    return int(pd.Timestamp(value).timestamp() * 1000)


def trade(when=NOW, price='1.2'):
    return dict(e='trade', s='XRPUSDT', E=ms(when), T=ms(when), p=price)


def kline(interval='1m', when=NOW, closed=False, start=None):
    start = pd.Timestamp(start) if start else when.floor({'1m': 'min', '5m': '5min', '15m': '15min', '1h': 'h'}[interval])
    end = start + pd.Timedelta(seconds=INTERVALS[interval]) - pd.Timedelta(milliseconds=1)
    return dict(e='kline', s='XRPUSDT', E=ms(when), k=dict(i=interval, t=ms(start), T=ms(end),
        o='1.1', h='1.3', l='1.0', c='1.2', v='100', q='120', x=closed))


def rest_row(interval, now=NOW):
    k = kline(interval, now)['k']
    return [k['t'], '1.1', '1.3', '1.0', '1.2', '100', k['T'], '120', 10, '1', '1', '0']


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.now = NOW
        self.feed = MarketStream(clock=lambda: self.now)

    def test_trade_price_event_clock_receipt_clock_and_slst(self):
        self.feed.connected = True
        self.feed.ingest(json.dumps({'data': trade(NOW - pd.Timedelta(seconds=3))}))
        snap = self.feed.snapshot()
        self.assertEqual(snap['trade']['price'], 1.2)
        self.assertEqual(snap['trade']['trade_at'], NOW - pd.Timedelta(seconds=3))
        self.assertEqual(snap['trade']['received_at'], NOW)
        self.assertTrue(snap['connected'] and snap['trade_fresh'])
        self.assertIn('01:30:27 PM', slst(snap['trade']['trade_at']))

    def test_old_trade_replay_does_not_replace_current_price(self):
        self.feed.ingest(trade())
        self.feed.ingest(trade(NOW - pd.Timedelta(minutes=1), '9'))
        self.assertEqual(self.feed.snapshot()['trade']['price'], 1.2)

    def test_stale_exchange_event_cannot_become_live_after_receipt(self):
        self.feed.connected = True
        self.feed.ingest(trade(NOW - pd.Timedelta(minutes=2)))
        snap = self.feed.snapshot()
        self.assertTrue(snap['connected'])
        self.assertFalse(snap['trade_fresh'])
        self.now += pd.Timedelta(seconds=46)
        self.assertFalse(self.feed.snapshot()['connected'])

    def test_invalid_prices_future_times_and_unrelated_symbols(self):
        for price in ('nan', 'inf', '-1', '0'):
            with self.assertRaises(ValueError):
                self.feed.ingest(trade(price=price))
        with self.assertRaises(ValueError):
            self.feed.ingest(trade(NOW + pd.Timedelta(minutes=1)))
        self.feed.ingest(dict(trade(), s='BTCUSDT'))
        self.assertIsNone(self.feed.snapshot()['trade'])

    def test_candle_upsert_late_event_and_closed_flag(self):
        msg = kline()
        self.feed.ingest(msg)
        self.feed.ingest(msg)
        snap = self.feed.snapshot()
        self.assertEqual(len(snap['candles']), 1)
        self.assertTrue(snap['chart_fresh'])
        closed = kline(when=NOW, closed=True, start='2026-11-16T07:59:00Z')
        self.feed.ingest(closed)
        reopened = kline(when=NOW, closed=False, start='2026-11-16T07:59:00Z')
        self.feed.ingest(reopened)
        bars = self.feed.snapshot()['candles']
        self.assertTrue(bars.iloc[0].closed)
        invalid = kline(closed=True)
        with self.assertRaises(ValueError):
            self.feed.ingest(invalid)

    def test_model_boundary_rejects_minutes_forming_hours_and_partial_hours(self):
        self.feed.ingest(kline('1m'))
        self.feed.ingest(kline('1h'))
        self.feed.ingest(kline('1h', closed=True, start='2026-11-16T07:00:00Z'))
        snap = self.feed.snapshot()
        result = model_candles(snap['candles'], snap, NOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.close_time.iloc[0], NOW.floor('h') - pd.Timedelta(milliseconds=1))
        self.assertNotIn('closed', result)
        partial = result.copy()
        partial['close_time'] -= pd.Timedelta(seconds=1)
        self.assertTrue(model_candles(partial, {}, NOW).empty)

    def test_model_boundary_deduplicates_and_keeps_hourly_feature_inputs(self):
        self.feed.ingest(kline('1h', closed=True, start='2026-11-16T07:00:00Z'))
        snap = self.feed.snapshot()
        base = model_candles(None, snap, NOW)
        before = base.copy(deep=True)
        merged = model_candles(base, snap, NOW)
        pd.testing.assert_frame_equal(merged, before)
        pd.testing.assert_frame_equal(base, before)

    def test_rest_fallback_has_true_trade_timestamp_and_separate_clocks(self):
        def get(url, params, timeout):
            data = ([dict(price='1.25', time=ms(NOW - pd.Timedelta(seconds=5)))]
                    if url.endswith('/trades') else [rest_row(params['interval'])])
            return Mock(status_code=200, json=lambda: data)
        self.feed.http.get = Mock(side_effect=get)
        self.feed.repair()
        snap = self.feed.snapshot()
        self.assertTrue(snap['trade_fresh'] and snap['chart_fresh'])
        self.assertFalse(snap['connected'])
        self.assertEqual(snap['trade']['source'], 'REST last trade')
        self.assertEqual(snap['trade']['trade_at'], NOW - pd.Timedelta(seconds=5))
        self.assertIsNone(snap['rest_error'])
        self.assertEqual(self.feed.http.get.call_count, 5)
        # The latest forming hourly REST bar never enters the model.
        self.assertTrue(model_candles(None, snap, NOW).empty)

    def test_failed_fallback_does_not_refresh_old_timestamps(self):
        self.feed.ingest(trade())
        self.feed.http.get = Mock(side_effect=requests.Timeout('fixture outage'))
        self.now += pd.Timedelta(minutes=2)
        self.feed.repair()
        snap = self.feed.snapshot()
        self.assertEqual(snap['trade']['received_at'], NOW)
        self.assertFalse(snap['trade_fresh'])
        self.assertIn('fixture outage', snap['rest_error'])

    def test_rate_limit_cooldown_prevents_repeated_requests(self):
        response = Mock(status_code=429, headers={'Retry-After': '120'})
        response.raise_for_status.side_effect = requests.HTTPError('fixture 429')
        self.feed.http.get = Mock(return_value=response)
        self.feed.repair()
        self.assertEqual(self.feed.http.get.call_count, 1)
        self.assertGreater(self.feed.cooldown_until, time.monotonic() + 100)

    def test_stream_reconnect_after_handshake_failure_recovers(self):
        socket = Mock()
        def receive(timeout):
            self.feed.stop_event.set()
            return json.dumps(trade())
        socket.recv.side_effect = receive
        context = Mock()
        context.__enter__ = Mock(return_value=socket)
        context.__exit__ = Mock(return_value=False)
        connector = Mock(side_effect=[ConnectionError('fixture disconnect'), context])
        self.feed.connector = connector
        self.feed.repair = Mock()
        with patch('streaming_market.random.uniform', return_value=0):
            self.feed._run()
        self.assertEqual(connector.call_count, 2)
        self.assertEqual(self.feed.reconnects, 1)
        self.assertEqual(self.feed.snapshot()['trade']['price'], 1.2)
        self.assertFalse(self.feed.connected)

    def test_server_shutdown_triggers_reconnect(self):
        with self.assertRaises(ConnectionError):
            self.feed.ingest(dict(e='serverShutdown', E=ms(NOW)))

    def test_offline_mode_starts_no_socket_or_rest_worker(self):
        with patch.dict(os.environ, {'XRP_DASHBOARD_OFFLINE': '1'}):
            self.feed.start()
        self.assertIsNone(self.feed.thread)

    def test_snapshot_does_not_expose_mutable_internal_data(self):
        self.feed.ingest(trade())
        snap = self.feed.snapshot()
        snap['trade']['price'] = 99
        self.assertEqual(self.feed.snapshot()['trade']['price'], 1.2)

    def test_rolling_stats_clock_is_not_overwritten_as_a_trade_time(self):
        self.feed.ingest(trade())
        ticker = dict(price=1, high_24h=2, retrieved_at='2020-01-01T00:00:00Z')
        quote = quote_context(ticker, self.feed.snapshot())
        self.assertEqual(quote['price'], 1.2)
        self.assertEqual(ticker['retrieved_at'], '2020-01-01T00:00:00Z')
        self.now += pd.Timedelta(minutes=2)
        self.assertEqual(quote_context(ticker, self.feed.snapshot()), ticker)

    def test_chart_no_future_padding_and_rsi_fixed_scale_all_timeframes(self):
        for interval in INTERVALS:
            rows = []
            for i in range(100):
                at = NOW - pd.Timedelta(seconds=(99-i)*INTERVALS[interval])
                self.feed.ingest(kline(interval, when=at))
            frame = self.feed.snapshot(interval)['candles']
            before = frame.copy(deep=True)
            fig = intelligence_chart(frame, pd.DataFrame(), days=30, interval=interval, now=NOW)
            right = pd.Timestamp(fig.layout.xaxis.range[1])
            local_now = NOW.tz_convert('Asia/Colombo').tz_localize(None)
            self.assertLessEqual(right, local_now + pd.Timedelta(seconds=2))
            self.assertEqual(tuple(fig.layout.yaxis2.range), (0, 100))
            self.assertTrue({'y', 'y2', 'y3'}.issubset({trace.yaxis for trace in fig.data}))
            pd.testing.assert_frame_equal(frame, before)


class LiveUITests(unittest.TestCase):
    def test_fast_fragment_timeframes_disconnection_and_model_separation(self):
        from streamlit.testing.v1 import AppTest
        from live_market_ui import stream_resource
        import streamlit as st
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        feed = MarketStream(clock=lambda: pd.Timestamp.now(tz='UTC'))
        for interval in INTERVALS:
            now = pd.Timestamp.now(tz='UTC')
            feed.ingest(kline(interval, now))
        feed.ingest(trade(pd.Timestamp.now(tz='UTC')))
        with patch.dict(os.environ, {'XRP_DASHBOARD_ROOT': tmp.name, 'XRP_DASHBOARD_CACHE_DIR': tmp.name,
                'XRP_DASHBOARD_OFFLINE': '1'}), patch('live_market_ui.stream_resource', return_value=feed):
            st.cache_data.clear()
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any('WebSocket disconnected' in item.value for item in app.warning))
            for interval in INTERVALS:
                app.selectbox(key='live_timeframe').select(interval).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(app.get('plotly_chart'))
                values = {item.label: item.value for item in app.metric}
                self.assertEqual(values['UP'], 'Not validated')
                self.assertEqual(values['DOWN'], 'Not validated')
                self.assertEqual(values['NEUTRAL'], 'Not validated')
            app.toggle(key='live_follow').set_value(False).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any('MODEL' in item.value.upper() and 'HOURLY' in item.value.upper() for item in app.caption))
            st.cache_data.clear()

    def test_conflicting_evidence_and_old_news_are_not_positive_forecasts(self):
        from live_market_ui import observation_evidence
        values = dict(ema20=1.2, ema50=1.1, macd_histogram=-.01, rsi=80, volume_spike=True)
        evidence = observation_evidence(dict(indicators=values), pd.DataFrame())
        self.assertTrue(evidence['up'] and evidence['down'])
        self.assertIn('conflicts', evidence['uncertain'][0])
        self.assertIn('No sufficiently recent', evidence['news'])
        self.assertNotIn('probabilities', evidence)


if __name__ == '__main__':
    unittest.main()
