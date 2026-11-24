"""Virtual-only fixture tests. No real positions or forecasting claims."""
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from paper_trading import (PaperAccount, balance_curve, capture_signal, decimal,
                           fresh_quote, instant, outcome, statistics)

NOW = pd.Timestamp('2026-11-16T09:30:00Z')
ORIGIN = NOW.floor('h') - pd.Timedelta(milliseconds=1)


def quote(now=NOW, price='2', source='WebSocket trade'):
    return dict(price=price, trade_at=now, received_at=now, source=source)


def bar(close_at, price='2'):
    close_at = pd.Timestamp(close_at)
    return pd.DataFrame([dict(open_time=close_at - pd.Timedelta(hours=1) + pd.Timedelta(milliseconds=1),
        close_time=close_at, close=float(price), open=2., high=3., low=1., volume=100, quote_volume=200, closed=True)])


def signal():
    return dict(signal='UP', version='fixture-hourly', threshold='.002', origin_at=ORIGIN.isoformat(),
        target_at=(ORIGIN + pd.Timedelta(hours=1)).isoformat(), origin_price='2',
        generated_at=NOW.isoformat(), reason='Fixture, not a production prediction.')


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'virtual.sqlite3'
        self.ledger = PaperAccount(self.path)
        self.account = self.ledger.account()['id']

    def open(self, token='entry', budget='10', fee='.001', slip='.0005', observe=True, s=None):
        return self.ledger.open(self.account, token, budget, fee, slip, quote(), signal() if s is None else s,
                                observe_hour=observe, now=NOW)

    def test_initial_100_and_default_fee_inclusive_quantity(self):
        self.assertEqual(decimal(self.ledger.account()['cash']), Decimal('100'))
        self.open()
        row = self.ledger.trades()[0]
        self.assertEqual(decimal(row['entry_price']), Decimal('2.0010'))
        self.assertEqual(decimal(row['quantity']), Decimal('10') / Decimal('1.001') / Decimal('2.001'))
        self.assertEqual(decimal(row['entry_fee']), Decimal('10') - Decimal('10') / Decimal('1.001'))
        self.assertEqual(decimal(self.ledger.account()['cash']), Decimal('90'))

    def test_flat_market_loses_fees_and_slippage_and_exit_accounting_reconciles(self):
        tid = self.open()
        at = NOW + pd.Timedelta(minutes=2)
        self.ledger.close(self.account, 'exit', tid, quote(at), at)
        row = self.ledger.trades()[0]
        gross = decimal(row['quantity']) * Decimal('1.999')
        self.assertEqual(decimal(row['exit_fee']), gross * Decimal('.001'))
        self.assertLess(decimal(row['net_pnl']), 0)
        self.assertEqual(decimal(self.ledger.account()['cash']), Decimal('100') + decimal(row['net_pnl']))
        self.assertEqual(decimal(row['proceeds']), gross - decimal(row['exit_fee']))

    def test_zero_cost_gain_and_loss_statistics(self):
        for i, price in enumerate(('2.2', '1.8')):
            tid = self.open(token='open'+str(i), fee=0, slip=0)
            at = NOW + pd.Timedelta(seconds=10+i)
            self.ledger.close(self.account, 'close'+str(i), tid, quote(at, price), at)
        stats = statistics(self.ledger.account(), self.ledger.trades(), now=NOW)
        self.assertEqual(stats['realized_pnl'], 0)
        self.assertEqual(stats['average_gain'], 1)
        self.assertEqual(stats['average_loss'], -1)
        self.assertEqual(stats['win_rate'], .5)
        self.assertEqual(stats['total_fees'], 0)
        self.assertIsNone(stats['model_accuracy'])

    def test_all_cash_spend_prevents_borrowing_and_excess_budget_rolls_back(self):
        self.open(budget=100)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            self.open('too-much', budget=1)
        self.assertEqual(decimal(self.ledger.account()['cash']), 0)
        self.assertEqual(len(self.ledger.trades()), 1)

    def test_stale_unavailable_future_ticker_and_invalid_price_block_fills(self):
        invalid = [None, dict(quote(), trade_at=NOW-pd.Timedelta(seconds=16)),
            dict(quote(), received_at=NOW-pd.Timedelta(seconds=46)),
            dict(quote(), trade_at=NOW+pd.Timedelta(seconds=1)),
            dict(quote(), source='REST ticker'), dict(quote(), price='nan'), dict(quote(), price='-1')]
        for i, q in enumerate(invalid):
            with self.assertRaises(ValueError):
                self.ledger.open(self.account, str(i), 10, 0, 0, q, {}, now=NOW)
        self.assertEqual(decimal(self.ledger.account()['cash']), 100)
        self.assertFalse(self.ledger.trades())

    def test_rest_trade_fallback_is_fresh_but_not_retroactively_renewed(self):
        fresh_quote(quote(NOW-pd.Timedelta(seconds=40), source='REST last trade'), NOW)
        with self.assertRaises(ValueError):
            fresh_quote(quote(NOW-pd.Timedelta(seconds=46), source='REST last trade'), NOW)

    def test_stale_close_and_pre_entry_exit_are_blocked(self):
        tid = self.open()
        for at, q in [(NOW+pd.Timedelta(minutes=2), quote()), (NOW-pd.Timedelta(seconds=1), quote(NOW-pd.Timedelta(seconds=1)))]:
            with self.assertRaises(ValueError):
                self.ledger.close(self.account, str(at), tid, q, at)
        self.assertEqual(self.ledger.trades()[0]['status'], 'OPEN')

    def test_action_retry_is_idempotent_even_with_new_quote_or_later_time(self):
        tid = self.open()
        self.assertEqual(self.open(), tid)
        at = NOW+pd.Timedelta(minutes=2)
        self.ledger.close(self.account, 'exit', tid, quote(at), at)
        self.ledger.close(self.account, 'exit', tid, None, at+pd.Timedelta(hours=2))
        with self.assertRaises(ValueError):
            self.open(budget=20)
        self.assertEqual(len(self.ledger.trades()), 1)
        with self.assertRaises(ValueError):
            self.ledger.close(self.account, 'different-exit', tid, quote(at), at)

    def test_concurrent_duplicate_and_competing_spends_are_atomic(self):
        def same(_):
            return PaperAccount(self.path).open(self.account, 'same', 60, 0, 0, quote(), {}, now=NOW)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(same, range(4)))
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(decimal(self.ledger.account()['cash']), 40)
        def spend(i):
            try:
                PaperAccount(self.path).open(self.account, str(i), 30, 0, 0, quote(), {}, now=NOW)
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(spend, range(2))), 1)
        self.assertEqual(decimal(self.ledger.account()['cash']), 10)

    def test_persistence_after_reopening_database(self):
        tid = self.open()
        restarted = PaperAccount(self.path)
        self.assertEqual(restarted.trades()[0]['id'], tid)
        self.assertEqual(decimal(restarted.account()['cash']), 90)
        self.assertEqual(json.loads(restarted.trades()[0]['signal_json'])['threshold'], '.002')

    def test_confirmed_reset_archives_without_fake_exits_and_blocks_old_tabs(self):
        self.open()
        with self.assertRaises(ValueError):
            self.ledger.reset(self.account, 'reset')
        new_id = self.ledger.reset(self.account, 'reset', True, NOW)
        self.assertEqual(self.ledger.reset(self.account, 'reset', True, NOW), new_id)
        self.assertEqual(decimal(self.ledger.account()['cash']), 100)
        self.assertFalse(self.ledger.trades())
        archived = self.ledger.trades(all_accounts=True)
        self.assertEqual(archived[0]['status'], 'OPEN')
        self.assertIsNone(archived[0]['exit_price'])
        with self.assertRaisesRegex(ValueError, 'reset'):
            self.open('old-tab')

    def test_exact_hour_model_outcome_differs_from_mid_hour_position_return(self):
        self.open(fee=0, slip=0)
        target = ORIGIN+pd.Timedelta(hours=1)
        self.ledger.evaluate(bar(target, '2.02'), quote(NOW+pd.Timedelta(minutes=59), '1.9'), NOW+pd.Timedelta(minutes=59))
        self.assertIsNone(self.ledger.trades()[0]['model_outcome'])
        observed = NOW+pd.Timedelta(hours=1, minutes=3)
        self.ledger.evaluate(bar(target, '2.02'), quote(observed, '1.9'), observed)
        row = self.ledger.trades()[0]
        self.assertEqual(row['model_outcome'], 'UP')
        self.assertEqual(decimal(row['observed_return']), Decimal('-.05'))
        self.assertEqual(pd.Timestamp(row['observed_at'])-pd.Timestamp(row['due_at']), pd.Timedelta(minutes=3))
        self.assertEqual(row['status'], 'OPEN')
        self.assertIsNone(row['net_pnl'])

    def test_missing_partial_duplicated_or_forming_target_never_scores(self):
        self.open()
        target = ORIGIN+pd.Timedelta(hours=1)
        frame = bar(target)
        partial = frame.copy()
        partial['open_time'] += pd.Timedelta(seconds=1)
        forming = frame.copy()
        forming['closed'] = False
        for invalid in (pd.DataFrame(), bar(target+pd.Timedelta(hours=1)), partial, forming, pd.concat([frame, frame])):
            self.ledger.evaluate(invalid, now=NOW+pd.Timedelta(hours=2))
            self.assertIsNone(self.ledger.trades()[0]['model_outcome'])

    def test_evaluation_is_immutable_and_deduplicated_by_model_hour(self):
        self.open()
        self.open('second')
        at = NOW+pd.Timedelta(hours=1)
        target = ORIGIN+pd.Timedelta(hours=1)
        self.ledger.evaluate(bar(target, '2.02'), quote(at), at)
        before = self.ledger.trades()
        self.ledger.evaluate(bar(target, '1.5'), quote(at+pd.Timedelta(minutes=5), '3'), at+pd.Timedelta(minutes=5))
        self.assertEqual(self.ledger.trades(), before)
        stats = statistics(self.ledger.account(), before, quote(at), at)
        self.assertEqual(stats['model_samples'], 1)
        self.assertEqual(stats['model_accuracy'], 1)
        self.assertEqual(stats['closed'], 0)
        self.assertIsNone(stats['win_rate'])

    def test_optional_paper_observation_does_not_disable_hourly_model_scoring(self):
        self.open(observe=False)
        at = NOW+pd.Timedelta(hours=1)
        self.ledger.evaluate(bar(ORIGIN+pd.Timedelta(hours=1), '2.02'), quote(at), at)
        row = self.ledger.trades()[0]
        self.assertIsNone(row['observed_at'])
        self.assertEqual(row['model_outcome'], 'UP')

    def test_threshold_boundaries_and_invalid_finite_inputs(self):
        for value, expected in [('.002','UP'), ('-.002','DOWN'), ('.001999','NEUTRAL')]:
            self.assertEqual(outcome(value,'.002'),expected)
        for value in ('nan','inf',None):
            with self.assertRaises(ValueError):
                decimal(value)
        with self.assertRaises(ValueError):
            outcome(0, 0)
        with self.assertRaises(ValueError):
            instant('2026-11-16T09:30:00')

    def test_signal_freezes_threshold_and_never_stores_probabilities(self):
        forecast = dict(experimental_label='DOWN', model_version='fixture', threshold=.002,
            feature_as_of=ORIGIN.isoformat(), generated_at=NOW.isoformat(), probabilities={'UP':.99}, limitation='Experimental')
        recorded = capture_signal(forecast, bar(ORIGIN), NOW)
        self.assertEqual(recorded['signal'], 'DOWN')
        self.assertNotIn('probabilities', recorded)
        self.open(s=dict(recorded, probabilities={'UP':.99}))
        self.assertNotIn('probabilities', json.loads(self.ledger.trades()[0]['signal_json']))
        stale = capture_signal(dict(forecast, feature_as_of=(ORIGIN-pd.Timedelta(hours=1)).isoformat()), bar(ORIGIN), NOW)
        self.assertIsNone(stale['signal'])

    def test_equity_unavailable_on_stale_quote_and_cash_curve_conserves_money(self):
        self.open()
        stats = statistics(self.ledger.account(), self.ledger.trades(), quote(), NOW+pd.Timedelta(minutes=1))
        self.assertIsNone(stats['equity'])
        self.assertEqual(stats['cash'], 90)
        valid = statistics(self.ledger.account(), self.ledger.trades(), quote(), NOW)
        self.assertLess(valid['equity'], 100)
        frame = balance_curve(self.ledger.account(), self.ledger.trades())
        self.assertEqual(frame.balance.iloc[-1], float(decimal(self.ledger.account()['cash'])))


class EvidenceTests(unittest.TestCase):
    def test_public_backfill_requires_exact_target_and_honors_rate_limit(self):
        from paper_market_data import PublicHourlyEvidence
        client = PublicHourlyEvidence()
        response = Mock(status_code=429, headers={'Retry-After':'120'})
        response.raise_for_status.side_effect = requests.HTTPError('fixture 429')
        with patch.dict(os.environ, {'XRP_DASHBOARD_OFFLINE':'0'}), patch('paper_market_data.requests.get', return_value=response) as get:
            with self.assertRaises(requests.HTTPError):
                client.fetch('2020-01-01T00:59:59.999Z')
            with self.assertRaisesRegex(ValueError,'cooldown'):
                client.fetch('2020-01-01T00:59:59.999Z')
            self.assertEqual(get.call_count,1)
        with patch.dict(os.environ, {'XRP_DASHBOARD_OFFLINE':'1'}):
            with self.assertRaises(ValueError):
                client.fetch('2020-01-01T00:59:59.999Z')


class PaperUITests(unittest.TestCase):
    def setUp(self):
        from streaming_market import MarketStream
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root/'virtual.sqlite3'
        self.feed = MarketStream()
        self.env = patch.dict(os.environ, {'XRP_DASHBOARD_ROOT':str(self.root),
            'XRP_PAPER_DB':str(self.db), 'XRP_DASHBOARD_OFFLINE':'1', 'XRP_DASHBOARD_CACHE_DIR':str(self.root/'cache')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.transport = patch('live_market_ui.stream_resource', return_value=self.feed)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        import streamlit as st
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        app.radio[0].set_value('Paper Trading').run(timeout=30)
        self.assertFalse(app.exception)
        return app

    def fresh(self):
        now = pd.Timestamp.now(tz='UTC')
        with self.feed.lock:
            self.feed.trade = quote(now, price=2.)

    def button(self, app, prefix):
        return next(b for b in app.button if b.key and b.key.startswith(prefix))

    def test_unavailable_price_disables_open_and_initial_balance_is_100(self):
        app = self.app()
        self.assertTrue(self.button(app, 'paper_open_').disabled)
        self.assertEqual({m.label:m.value for m in app.metric}['Available virtual USDT'],'100.000000')
        self.assertTrue(any('virtual' in i.value.lower() for i in app.info))

    def test_open_close_restart_and_confirmed_reset_ui(self):
        self.fresh()
        app = self.app()
        original_token = self.button(app,'paper_open_').key
        self.button(app,'paper_open_').click().run(timeout=30)
        self.assertFalse(app.exception)
        ledger = PaperAccount(self.db)
        self.assertEqual(len(ledger.trades()),1)
        self.assertNotEqual(self.button(app,'paper_open_').key,original_token)
        self.assertEqual(decimal(ledger.account()['cash']),90)
        self.assertIsNone(json.loads(ledger.trades()[0]['signal_json'])['signal'])
        app = self.app()
        self.assertEqual(len(PaperAccount(self.db).trades()),1)
        self.fresh()
        self.button(app,'paper_close_').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(PaperAccount(self.db).trades()[0]['status'],'CLOSED')
        self.assertTrue(self.button(app,'paper_reset_').disabled)
        next(c for c in app.checkbox if c.key.startswith('paper_reset_confirm_')).check().run(timeout=30)
        self.button(app,'paper_reset_').click().run(timeout=30)
        self.assertFalse(app.exception)
        reset = PaperAccount(self.db)
        self.assertEqual(decimal(reset.account()['cash']),100)
        self.assertFalse(reset.trades())
        self.assertEqual(len(reset.trades(all_accounts=True)),1)


if __name__ == '__main__':
    unittest.main()
