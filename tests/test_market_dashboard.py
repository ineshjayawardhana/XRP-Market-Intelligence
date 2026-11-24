"""Market/UI/probability safeguards. Prices and forecasts here are test fixtures."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import dashboard_data
from ai_labeling import atomic_csv
from forecast_validation import validated_forecast
from live_data import _forward_outcomes
from market_intelligence import (candle_quality, evaluate_history, freshness, intelligence_chart,
                                 market_outlook, recent_news, slst)
from news_label_safety import sha256


def candles(n=240, current=None):
    current = pd.Timestamp.now(tz='UTC') if current is None else pd.Timestamp(current)
    end = current.floor('h') - pd.Timedelta(milliseconds=1)
    times = pd.date_range(end=end, periods=n, freq='h')
    prices = 1.0 + np.arange(n) * .0002 + .01 * np.sin(np.arange(n) / 3)
    return pd.DataFrame(dict(close_time=times, open_time=times - pd.Timedelta(milliseconds=3599999),
        close=prices, open=prices, high=prices + .01, low=prices - .01,
        volume=np.ones(n) * 100, quote_volume=prices * 100))


def ticker(current=None):
    current = pd.Timestamp.now(tz='UTC') if current is None else pd.Timestamp(current)
    return dict(price=1.1, high_24h=1.2, low_24h=1., change_24h_pct=2., volume_24h_xrp=10000.,
                volume_24h_usdt=11000., retrieved_at=current.isoformat(), exchange_close_at=current.isoformat())


def headlines(current=None):
    current = pd.Timestamp.now(tz='UTC') if current is None else pd.Timestamp(current)
    return pd.DataFrame([dict(id=str(i), headline=f'XRP fixture report {i}', source=f'publisher{i%2}',
        url=f'https://fixture.example/{i}', label='Negative_FUD', label_status='AI provisional',
        published_at=current - pd.Timedelta(hours=1), observed_at=current - pd.Timedelta(minutes=30),
        outlook_eligible=True) for i in range(3)])


class MarketTests(unittest.TestCase):
    def test_slst_and_freshness_separate_from_system_timezone(self):
        self.assertIn('05:30:00 AM', slst('2026-11-15T00:00:00Z'))
        self.assertFalse(freshness(None, 90)['fresh'])
        self.assertFalse(freshness('2026-11-16T00:00:00Z', 90, '2026-11-15T00:00:00Z')['fresh'])
        self.assertFalse(freshness('2026-11-14T00:00:00Z', 90, '2026-11-15T00:00:00Z')['fresh'])

    def indicator_frame(self, frame, direction):
        result = frame.copy()
        result['close'] = 1.05 if direction == 1 else .95
        result['ema20'] = 1.03 if direction == 1 else 1.0
        result['ema50'] = 1.0 if direction == 1 else 1.03
        result['rsi'] = 60 if direction == 1 else 40
        result['macd'], result['macd_signal'] = direction * .01, direction * .005
        result['volume_spike'] = False
        return result

    def test_bullish_and_bearish_are_explained_without_probabilities(self):
        frame = candles()
        for direction, expected in [(1, 'Bullish'), (-1, 'Bearish')]:
            with patch('market_intelligence.calculate_indicators', return_value=self.indicator_frame(frame, direction)):
                result = market_outlook(ticker(), frame, pd.DataFrame())
            self.assertEqual(result['condition'], expected)
            self.assertEqual(len(result['reasons']), 3)
            self.assertNotIn('probabilities', result)

    def test_conflicting_momentum_extreme_rsi_and_news_are_mixed(self):
        frame = candles()
        indicators = self.indicator_frame(frame, 1)
        for variant in ('rsi', 'macd', 'news'):
            changed = indicators.copy()
            if variant == 'rsi':
                changed['rsi'] = 90
            if variant == 'macd':
                changed['macd'] = -.01
            with patch('market_intelligence.calculate_indicators', return_value=changed):
                result = market_outlook(ticker(), frame, headlines() if variant == 'news' else pd.DataFrame())
            self.assertEqual(result['condition'], 'Mixed / Insufficient Evidence')

    def test_stale_quote_bad_candles_or_failed_refresh_never_look_confident(self):
        frame = candles()
        old = ticker(pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1))
        for quote, data, errors in [(old, frame, []), (ticker(), frame.iloc[:10], []), (ticker(), frame, ['failed'])]:
            self.assertEqual(market_outlook(quote, data, pd.DataFrame(), errors)['condition'], 'Mixed / Insufficient Evidence')

    def test_recently_observed_old_article_is_not_current_news(self):
        news = headlines()
        news['published_at'] = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)
        self.assertTrue(recent_news(news).empty)
        news['published_at'] = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1)
        news['outlook_eligible'] = False
        self.assertTrue(recent_news(news).empty)

    def test_gaps_are_detected_and_forward_horizon_is_exactly_24_hours(self):
        frame = candles().drop(index=120).reset_index(drop=True)
        self.assertTrue(any('Missing hourly' in item for item in candle_quality(frame)))
        outcomes = _forward_outcomes(frame)
        prices = frame.set_index('close_time')['close']
        for _, row in outcomes.iterrows():
            target = row.close_time + pd.Timedelta(hours=24)
            self.assertIn(target, prices.index)
            self.assertAlmostEqual(row.future_price_24h, prices.loc[target])
        with self.assertRaisesRegex(ValueError, 'Historical evaluation blocked'):
            evaluate_history(frame, pd.DataFrame())

    def test_realized_outcomes_counts_do_not_create_forecasts(self):
        results = evaluate_history(candles(300), pd.DataFrame())
        self.assertGreater(len(results['baseline']), 0)
        self.assertTrue(results['combined'].empty)
        self.assertTrue(results['baseline'].outcome.isin(['UP', 'DOWN', 'NEUTRAL']).all())
        self.assertNotIn('probabilities', results)

    def test_selected_history_window_excludes_indicator_warmup_days(self):
        results = evaluate_history(candles(1000), pd.DataFrame(), days=7)
        self.assertLessEqual(len(results['baseline']), 8)
        self.assertEqual(results['requested_days'], 7)

    def test_three_chart_panels_and_slst_display_preserve_utc_inputs(self):
        frame = candles()
        before = frame.copy(deep=True)
        fig = intelligence_chart(frame, headlines(), days=7)
        self.assertTrue({'y', 'y2', 'y3'}.issubset({trace.yaxis for trace in fig.data}))
        visible = frame[frame.close_time >= frame.close_time.max() - pd.Timedelta(days=7)]
        self.assertEqual(pd.Timestamp(fig.data[0].x[0]), visible.close_time.iloc[0].tz_convert('Asia/Colombo').tz_localize(None))
        pd.testing.assert_frame_equal(frame, before)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_api_failure_reuses_original_timestamp_or_shows_unavailable(self):
        cache = self.root / 'data' / 'dashboard_cache'
        cache.mkdir(parents=True)
        old = ticker('2020-01-01T00:00:00Z')
        (cache / 'last_ticker.json').write_text(json.dumps(old))
        with patch('dashboard_data.get_live_price', side_effect=RuntimeError('network failure')):
            result = dashboard_data.market_component('ticker', self.root)
        self.assertEqual(result['data']['retrieved_at'], old['retrieved_at'])
        self.assertTrue(result['fallback'])
        (cache / 'last_ticker.json').unlink()
        with patch('dashboard_data.get_live_price', side_effect=RuntimeError('network failure')):
            self.assertIsNone(dashboard_data.market_component('ticker', self.root)['data'])

    def test_corrupted_or_nonfinite_quote_is_not_saved(self):
        with patch('dashboard_data.get_live_price', return_value=dict(ticker(), price=float('inf'))):
            result = dashboard_data.market_component('ticker', self.root)
        self.assertIsNone(result['data'])
        self.assertIsNotNone(result['error'])

    def test_malformed_cache_and_invalid_candles_are_not_used(self):
        cache = self.root / 'data' / 'dashboard_cache'
        cache.mkdir(parents=True)
        (cache / 'last_ticker.json').write_text('[1, 2]')
        with patch('dashboard_data.get_live_price', side_effect=RuntimeError('network failure')):
            self.assertIsNone(dashboard_data.market_component('ticker', self.root)['data'])
        with patch('dashboard_data.get_historical_prices', return_value=candles().drop(index=120)):
            result = dashboard_data.market_component('candles', self.root)
        self.assertTrue(result['data'].empty)
        self.assertIn('Missing hourly', result['error'])
        self.assertFalse((cache / 'last_candles.json').exists())

    def test_research_file_cannot_promote_ai_to_human(self):
        folder = self.root / 'data' / 'multisource'
        research = folder / 'research'
        research.mkdir(parents=True)
        rows = [dict(id=str(i), headline=f'XRP fixture headline number {i}', source='fixture.example',
                     url=f'https://fixture.example/{i}', date='2026-11-15T00:00:00Z',
                     published_at='2026-11-15T00:00:00Z', label='Neutral_Market_News' if i == 0 else '') for i in range(2)]
        atomic_csv(folder / 'xrp_real_labeled.csv', list(rows[0]), rows)
        predictions = [dict(id=row['id'], headline=row['headline'], url=row['url'], predicted_label='Real_Adoption',
                            label_status='HUMAN_VERIFIED', heuristic_confidence='.9') for row in rows]
        atomic_csv(research / 'all_classifications.csv', list(predictions[0]), predictions)
        before = sha256(folder / 'xrp_real_labeled.csv')
        frame, _ = dashboard_data.load_news(self.root)
        self.assertEqual(frame[frame.id.eq('0')].iloc[0].label_status, 'Human verified')
        self.assertEqual(frame[frame.id.eq('0')].iloc[0].label, 'Neutral_Market_News')
        self.assertEqual(frame[frame.id.eq('1')].iloc[0].label_status, 'AI provisional')
        self.assertEqual(sha256(folder / 'xrp_real_labeled.csv'), before)

    def test_news_metric_provenance_has_no_invented_evaluation_date(self):
        folder = self.root / 'artifacts'
        folder.mkdir()
        (folder / 'metrics_tfidf.json').write_text(json.dumps(dict(accuracy=.88, test_examples=50)))
        result = dashboard_data.model_evaluations(self.root)[0]
        self.assertIn('Synthetic', result['scope'])
        self.assertIsNone(result['evaluated_at'])
        self.assertEqual(result['sample_size'], 50)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.current = pd.Timestamp('2026-11-15T12:00:00Z')
        self.market_time = self.current - pd.Timedelta(milliseconds=1)

    def evidence(self):
        model = self.folder / 'model.fixture'
        model.write_bytes(b'Unit-test artifact only; never a production trained model')
        times = pd.date_range('2024-01-02T00:00:00Z', periods=120, freq='D')
        rows = []
        labels = ['UP', 'DOWN', 'NEUTRAL']
        for i, date in enumerate(times):
            predicted = i % 3
            actual = predicted if i % 5 < 3 else (predicted + (1 if i % 5 == 3 else 2)) % 3
            values = [.6 if index == predicted else .2 for index in range(3)]
            rows.append(dict(available_at=date.isoformat(), prediction_generated_at=date.isoformat(),
                             true_outcome=labels[actual], **dict(zip(['p_UP', 'p_DOWN', 'p_NEUTRAL'], values))))
        pd.DataFrame(rows).to_csv(self.folder / 'oos.csv', index=False)
        self.contract = dict(task='XRP_24H_DIRECTION', horizon_hours=24, labels=labels,
            chronological_out_of_sample_passed=True, calibration_passed=True,
            model_file=model.name, model_sha256=sha256(model), oos_predictions_file='oos.csv',
            oos_predictions_sha256=sha256(self.folder / 'oos.csv'), train_end='2023-11-01T00:00:00Z',
            calibration_fit_end='2023-12-30T00:00:00Z', test_start=times[0].isoformat(), test_end=times[-1].isoformat(),
            evaluated_at='2024-05-03T00:00:00Z', test_samples=120)
        self.forecast = dict(model_sha256=sha256(model), horizon_hours=24, generated_at=self.current.isoformat(),
                             feature_as_of=self.market_time.isoformat(), probabilities=dict(UP=.4, DOWN=.35, NEUTRAL=.25))
        self.save()

    def save(self):
        (self.folder / 'trend_24h_validation.json').write_text(json.dumps(self.contract))
        (self.folder / 'trend_24h_forecast.json').write_text(json.dumps(self.forecast))

    def result(self):
        return validated_forecast(self.folder, self.market_time, self.current)

    def test_missing_artifacts_never_infer_odds_from_news_accuracy(self):
        (self.folder / 'metrics_tfidf.json').write_text(json.dumps(dict(accuracy=.99)))
        self.assertFalse(self.result()['validated'])
        self.assertIsNone(self.result()['probabilities'])

    def test_all_evidence_is_required_and_recomputed(self):
        self.evidence()
        self.assertTrue(self.result()['validated'], self.result()['limitation'])
        self.assertAlmostEqual(sum(self.result()['probabilities'].values()), 1)

    def test_flags_alone_or_wrong_task_never_enable_forecasts(self):
        (self.folder / 'trend_24h_validation.json').write_text(json.dumps(dict(calibration_passed=True)))
        self.assertFalse(self.result()['validated'])
        self.evidence()
        self.contract['task'] = 'NEWS_CLASSIFICATION'
        self.save()
        self.assertFalse(self.result()['validated'])

    def test_bad_calibration_rejected_even_when_manifest_claims_pass(self):
        self.evidence()
        frame = pd.read_csv(self.folder / 'oos.csv')
        for i in range(len(frame)):
            frame.loc[i, ['p_UP', 'p_DOWN', 'p_NEUTRAL']] = [.99 if k == i % 3 else .005 for k in range(3)]
        frame.to_csv(self.folder / 'oos.csv', index=False)
        self.contract['oos_predictions_sha256'] = sha256(self.folder / 'oos.csv')
        self.save()
        self.assertFalse(self.result()['validated'])

    def test_bad_periods_hashes_and_stale_forecasts_block(self):
        self.evidence()
        self.contract['train_end'] = '2025-01-01T00:00:00Z'
        self.save()
        self.assertFalse(self.result()['validated'])
        self.evidence()
        (self.folder / 'model.fixture').write_text('modified')
        self.assertFalse(self.result()['validated'])
        self.evidence()
        self.forecast['generated_at'] = '2026-11-13T12:00:00Z'
        self.save()
        self.assertFalse(self.result()['validated'])
        self.evidence()
        self.market_time -= pd.Timedelta(days=1)
        self.forecast['feature_as_of'] = self.market_time.isoformat()
        self.save()
        self.assertFalse(self.result()['validated'])

    def test_future_prediction_generation_is_temporal_leakage(self):
        self.evidence()
        frame = pd.read_csv(self.folder / 'oos.csv')
        frame.loc[0, 'prediction_generated_at'] = '2026-11-01T00:00:00Z'
        frame.to_csv(self.folder / 'oos.csv', index=False)
        self.contract['oos_predictions_sha256'] = sha256(self.folder / 'oos.csv')
        self.save()
        self.assertFalse(self.result()['validated'])


class DashboardUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'XRP_DASHBOARD_ROOT': str(self.root), 'XRP_DASHBOARD_OFFLINE': '1',
                                           'XRP_DASHBOARD_CACHE_DIR': str(self.root / 'cache')})
        self.env.start()
        self.addCleanup(self.env.stop)
        # Streaming has its own transport fixtures. UI tests must never start
        # a real background socket when mocking only the legacy REST adapter.
        stream_start = patch('streaming_market.MarketStream.start', lambda stream: stream)
        stream_start.start()
        self.addCleanup(stream_start.stop)
        news_start = patch('live_news.NewsPoller.start', lambda poller: poller)
        news_start.start()
        self.addCleanup(news_start.stop)
        import streamlit as st
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)

    def test_missing_data_and_both_themes_all_pages_load(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any('Predictive probabilities are not yet validated.' in info.value for info in app.info))
        for theme in ('Light', 'Dark'):
            app.selectbox(key='appearance').select(theme).run(timeout=30)
            self.assertFalse(app.exception)
        for page in ('News intelligence', 'Advanced research', 'Indicator guide', 'Overview'):
            app.radio[0].set_value(page).run(timeout=30)
            self.assertFalse(app.exception)

    def test_current_mock_market_chart_and_historical_action(self):
        frame = candles(300)
        def component(kind, root):
            return dict(data=ticker() if kind == 'ticker' else frame, error=None, fallback=False, last_success=pd.Timestamp.now(tz='UTC').isoformat())
        with patch('dashboard_data.market_component', side_effect=component), patch('dashboard_data.get_historical_prices', return_value=frame), patch.dict(os.environ, {'XRP_DASHBOARD_OFFLINE': '0'}), patch('dashboard_data.refresh_public_news', return_value=dict(errors=[], checked_at=pd.Timestamp.now(tz='UTC').isoformat(), added=0)):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(app.get('plotly_chart'))
            app.radio[0].set_value('Advanced research').run(timeout=30)
            app.button(key='evaluate_history').click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn('historical_research', app.session_state)


if __name__ == '__main__':
    unittest.main()
