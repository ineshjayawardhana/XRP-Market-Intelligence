"""Fresh 1h labels, chronology and percentage gates. Fixtures are not research."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from one_hour_trend import (ROOT, feature_signature, hourly_features, labels, learn_threshold,
                           live_forecast, split, train, verified_model)
from trend_market_data import read_candles, save_frame


def candles(n=2000, current=None):
    current = pd.Timestamp.now(tz='UTC') if current is None else pd.Timestamp(current)
    ends = pd.date_range(end=current.floor('h') - pd.Timedelta(milliseconds=1), periods=n, freq='h')
    values = 2 + .1 * np.sin(np.arange(n) / 17) + .03 * np.cos(np.arange(n) / 4)
    return pd.DataFrame(dict(open_time=ends - pd.Timedelta(hours=1) + pd.Timedelta(milliseconds=1),
        close_time=ends, open=values, high=values + .01, low=values - .01, close=values,
        volume=100 + 10 * np.sin(np.arange(n) / 3), quote_volume=values * 100))


class HourlyTests(unittest.TestCase):
    def test_targets_are_new_exact_next_close_and_ignore_old_targets(self):
        raw = candles()
        raw['target'] = 'OLD_24H_LABEL'
        raw['future_return_24h'] = 999.
        frame = hourly_features(raw)
        self.assertTrue((frame.target_end - frame.close_time).eq(pd.Timedelta(hours=1)).all())
        prices = raw.set_index('close_time').close
        np.testing.assert_allclose(frame.actual_return_1h, frame.target_end.map(prices) / frame.close - 1)
        self.assertEqual(frame.close_time.max(), raw.close_time.iloc[-2])
        self.assertTrue(np.isfinite(frame.actual_return_1h).all())

    def test_prefix_features_and_returns_do_not_change_after_future_append(self):
        raw = candles(2500)
        prefix = hourly_features(raw.iloc[:2000].copy())
        full = hourly_features(raw)
        columns = ['rsi', 'ema_gap', 'macd_relative', 'return_1h', 'actual_return_1h']
        pd.testing.assert_frame_equal(prefix[columns].reset_index(drop=True), full[full.close_time.isin(prefix.close_time)][columns].reset_index(drop=True))

    def test_missing_and_partial_candles_do_not_create_cross_gap_targets(self):
        raw = candles()
        gap_time = raw.close_time.iloc[1000]
        raw.loc[1000, 'close_time'] -= pd.Timedelta(minutes=20)
        raw = raw.drop(index=1001).reset_index(drop=True)
        frame = hourly_features(raw)
        self.assertFalse(((frame.close_time < gap_time) & (frame.target_end >= gap_time)).any())
        after = frame[frame.close_time > gap_time]
        self.assertGreater(after.close_time.min(), gap_time + pd.Timedelta(hours=168))

    def test_training_threshold_uses_only_training_returns(self):
        training = pd.DataFrame(dict(actual_return_1h=[-.005, -.003, -.001, 0., .002, .004, .006]))
        expected = training.actual_return_1h.abs().quantile(1 / 3)
        self.assertAlmostEqual(learn_threshold(training), expected)
        future = training.copy()
        future.actual_return_1h *= 100
        self.assertAlmostEqual(learn_threshold(training), expected)
        self.assertNotAlmostEqual(learn_threshold(future), expected)
        self.assertEqual(labels(np.array([expected, -expected, 0.]), expected).tolist(), ['UP', 'DOWN', 'NEUTRAL'])

    def test_strict_maturity_purging_at_both_boundaries(self):
        frame = hourly_features(candles(5000))
        boundary = frame.close_time.iloc[-100]
        training, calibration = split(frame, boundary)
        self.assertLess(training.target_end.max(), calibration.close_time.min())
        self.assertLess(calibration.target_end.max(), boundary)
        self.assertTrue((frame.close_time.iloc[1:].reset_index(drop=True) >= frame.target_end.iloc[:-1].reset_index(drop=True)).all())

    def test_wrong_horizon_cannot_reuse_24h_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifacts'
            path.mkdir()
            (path / 'trend_1h_current.json').write_text(json.dumps(dict(task='XRP_24H_DIRECTION', horizon_hours=24)))
            with self.assertRaises(ValueError):
                verified_model(folder)

    def test_old_test_evidence_keeps_percentages_hidden_even_when_numeric_checks_pass(self):
        bundle = dict(version='fixture-1h', threshold=.001)
        report = dict(evaluated_at='2021-11-02T00:00Z', selected_model='fixture',
            test=dict(sample_size=100), split=dict(test=dict(start='2021-01-01T00:00Z', end='2021-10-31T00:00Z', last_target_end='2021-10-31T01:00Z')))
        with patch('one_hour_trend.verified_model', return_value=(bundle, report, dict(passed=True, failures=[]))), patch('one_hour_trend.predict_bundle', return_value=np.array([[.4, .3, .3]])):
            result = live_forecast(candles(500))
        self.assertEqual(result['experimental_label'], 'UP')
        self.assertIsNone(result['probabilities'])
        self.assertFalse(result['validated'])

    def test_stale_completed_candle_withholds_current_signal(self):
        bundle = dict(version='fixture-1h', threshold=.001)
        report = dict(evaluated_at='2021-11-02T00:00Z', selected_model='fixture', test=dict(sample_size=100),
            split=dict(test=dict(start='2021-01-01T00:00Z', end='2021-10-31T00:00Z', last_target_end='2021-10-31T01:00Z')))
        frame = candles(500)
        frame.close_time -= pd.Timedelta(hours=1)
        frame.open_time -= pd.Timedelta(hours=1)
        with patch('one_hour_trend.verified_model', return_value=(bundle, report, dict(passed=True, failures=[]))):
            self.assertIsNone(live_forecast(frame)['experimental_label'])

    def test_reader_handles_mixed_fractional_second_iso_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mixed.csv'
            pd.DataFrame(dict(open_time=['2021-08-13T00:00:00Z', '2021-08-13T01:00:00Z'],
                close_time=['2021-08-13T00:59:59.999Z', '2021-08-13T01:59:59Z'])).to_csv(path, index=False)
            frame = read_candles(path)
        self.assertEqual(frame.close_time.iloc[0].microsecond, 999000)
        self.assertEqual(frame.close_time.iloc[1].microsecond, 0)

    def test_nonfinite_or_flat_threshold_refuses(self):
        with self.assertRaises(ValueError):
            learn_threshold(pd.DataFrame(dict(actual_return_1h=[0.] * 10)))
        with self.assertRaises(ValueError):
            labels(np.array([0.]), 0.)


@unittest.skipUnless((ROOT / 'artifacts/trend_1h_current.json').exists(), 'Actual local research artifact is optional')
class SavedArtifactTests(unittest.TestCase):
    def test_exact_model_replays_saved_scores(self):
        bundle, report, quality = verified_model(ROOT)
        self.assertEqual(bundle['horizon_hours'], 1)
        self.assertEqual(report['test']['sample_size'], sum(report['test']['outcome_distribution'].values()))
        self.assertEqual(bundle['feature_signature'], feature_signature())

    def test_completed_training_reuses_result_without_touching_test(self):
        with patch('one_hour_trend.fit', side_effect=AssertionError('Must not fit again')):
            report = train(ROOT, offline=True)
        self.assertEqual(report['horizon_hours'], 1)

    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pointer = json.loads((ROOT / 'artifacts/trend_1h_current.json').read_text())
            (root / 'artifacts').mkdir()
            (root / 'artifacts/trend_1h_current.json').write_text(json.dumps(pointer))
            for entry in pointer['files'].values():
                target = root / entry['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / entry['path'], target)
            (root / pointer['files']['model']['path']).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                verified_model(root)


if __name__ == '__main__':
    unittest.main()
