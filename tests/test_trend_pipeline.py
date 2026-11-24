"""Independent causal/split/metric checks; all market values are test fixtures."""
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from trend_metrics import OUTCOMES, metrics, quality_checks
from trend_pipeline import (FEATURES, build_dataset, feature_frame, fit_temperature,
                            partitions, temperature_scale, train_research)


def market_fixture(count=360):
    opens = pd.date_range('2025-01-01T00:00:00Z', periods=count, freq='h')
    sequence = np.arange(count, dtype=float)
    close = 1. + sequence * .00002 + .01 * np.sin(sequence / 7.)
    return pd.DataFrame(dict(open_time=opens, close_time=opens + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        open=close, high=close + .005, low=close - .005, close=close,
        volume=100. + 20. * np.cos(sequence / 11.), quote_volume=close * 100.))


class CausalFeatureTests(unittest.TestCase):
    def test_appending_or_changing_future_prices_cannot_change_prior_features(self):
        whole = market_fixture(400)
        prefix = whole.iloc[:200].copy()
        expected = feature_frame(prefix)[FEATURES]
        actual = feature_frame(whole).iloc[:200][FEATURES]
        pd.testing.assert_frame_equal(expected, actual)
        changed = whole.copy()
        changed.loc[250:, ['open', 'high', 'low', 'close']] *= 3.
        changed.loc[250:, 'volume'] *= 7.
        pd.testing.assert_frame_equal(expected, feature_frame(changed).iloc[:200][FEATURES])

    def test_targets_use_exact_24h_future_closes_and_drop_unknown_outcomes(self):
        whole = market_fixture()
        result = build_dataset(whole)
        prices = whole.set_index('close_time').close
        self.assertGreater(len(result), 0)
        self.assertTrue((result.target_end - result.close_time).eq(pd.Timedelta(hours=24)).all())
        np.testing.assert_allclose(result.future_close, result.target_end.map(prices))
        np.testing.assert_allclose(result.future_return_24h, result.future_close / result.close - 1.)
        self.assertEqual(result.close_time.max(), whole.close_time.iloc[-25])
        self.assertFalse(result.future_close.isna().any())

    def test_outcome_boundary_definitions_are_not_prediction_confidences(self):
        frame = market_fixture()
        for origin, future in [(140, 1.01), (180, .99), (220, 1.005)]:
            frame.loc[origin, ['open', 'close']] = 1.
            frame.loc[origin, 'high'], frame.loc[origin, 'low'] = 1.02, .98
            frame.loc[origin + 24, ['open', 'close']] = future
            frame.loc[origin + 24, 'high'], frame.loc[origin + 24, 'low'] = future + .01, future - .01
        result = build_dataset(frame).set_index('close_time')
        for origin, expected in [(140, 'UP'), (180, 'DOWN'), (220, 'NEUTRAL')]:
            self.assertEqual(result.loc[frame.close_time.iloc[origin], 'target'], expected)

    def test_supervised_target_fields_cannot_enter_the_feature_whitelist(self):
        forbidden = {'future_close', 'future_return_24h', 'target', 'target_end', 'outcome', 'true_outcome'}
        self.assertFalse(forbidden.intersection(FEATURES))
        self.assertEqual(len(FEATURES), len(set(FEATURES)))
        self.assertTrue({'rsi', 'ema20_distance', 'ema50_distance', 'macd_relative',
                         'volume_change_1h', 'return_24h', 'volatility_24h'}.issubset(FEATURES))

    def test_gaps_duplicates_and_forming_candles_are_rejected(self):
        whole = market_fixture()
        gap = whole.drop(index=150).reset_index(drop=True)
        duplicate = pd.concat([whole, whole.iloc[-1:]], ignore_index=True)
        for frame in (gap, duplicate):
            with self.assertRaises(ValueError):
                build_dataset(frame)
        with self.assertRaises(ValueError):
            build_dataset(whole, now=whole.open_time.iloc[-1])

    def test_malformed_ohlc_and_candle_duration_are_rejected(self):
        whole = market_fixture()
        bad_range = whole.copy()
        bad_range.loc[150, 'low'] = bad_range.loc[150, 'close'] + .1
        bad_duration = whole.copy()
        bad_duration.loc[150, 'open_time'] += pd.Timedelta(minutes=1)
        for frame in (bad_range, bad_duration):
            with self.assertRaises(ValueError):
                feature_frame(frame)


class ChronologicalSplitTests(unittest.TestCase):
    def observations(self, stride=25):
        times = pd.date_range('2024-01-01T00:00:00Z', periods=300, freq=f'{stride}h')
        return pd.DataFrame(dict(close_time=times, target_end=times + pd.Timedelta(hours=24),
                                 target=[OUTCOMES[i % 3] for i in range(len(times))]))

    def test_train_and_calibration_targets_mature_strictly_before_next_partition(self):
        data = self.observations()
        test_start = data.close_time.iloc[260]
        train, calibration = partitions(data, test_start)
        self.assertLess(train.target_end.max(), calibration.close_time.min())
        self.assertLess(calibration.target_end.max(), test_start)
        self.assertEqual(len(calibration), 45)
        self.assertFalse(set(train.close_time).intersection(calibration.close_time))
        self.assertTrue((data.close_time.diff().dropna() >= pd.Timedelta(hours=24)).all())

    def test_equal_horizon_boundaries_are_purged_instead_of_silently_shared(self):
        data = self.observations(stride=24)
        test_start = data.close_time.iloc[260]
        train, calibration = partitions(data, test_start)
        immediate_predecessor = test_start - pd.Timedelta(hours=24)
        self.assertNotIn(immediate_predecessor, set(calibration.close_time))
        self.assertLess(calibration.target_end.max(), test_start)
        self.assertLess(train.target_end.max(), calibration.close_time.min())

    def test_unknown_future_labels_never_enter_train_or_calibration(self):
        data = self.observations()
        cutoff = data.close_time.iloc[260]
        expected = partitions(data, cutoff)
        changed = data.copy()
        changed.loc[changed.close_time >= cutoff, 'target'] = 'SHOULD_NEVER_BE_SEEN'
        actual = partitions(changed, cutoff)
        for before, after in zip(expected, actual):
            pd.testing.assert_frame_equal(before, after)

    def test_sparse_or_missing_class_training_refuses_to_fit(self):
        data = self.observations()
        with self.assertRaises(ValueError):
            partitions(data, data.close_time.iloc[50])
        data['target'] = 'UP'
        with self.assertRaises(ValueError):
            partitions(data, data.close_time.iloc[260])


class CalibrationAndMetricsTests(unittest.TestCase):
    def test_temperature_scaling_reduces_overconfident_calibration_loss(self):
        y = np.array(OUTCOMES * 20)
        p = np.tile([.95, .025, .025], (len(y), 1))
        original = p.copy()
        temperature = fit_temperature(p, y)
        adjusted = temperature_scale(p, temperature)
        self.assertGreater(temperature, 1.)
        self.assertLess(metrics(y, adjusted)['log_loss'], metrics(y, p)['log_loss'])
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.)
        np.testing.assert_array_equal(p, original)

    def test_logloss_uses_explicit_up_down_neutral_column_mapping(self):
        y = np.array(['UP', 'DOWN', 'NEUTRAL', 'UP'])
        p = np.array([[.7, .1, .2], [.2, .7, .1], [.1, .2, .7], [.6, .3, .1]])
        result = metrics(y, p)
        self.assertAlmostEqual(result['log_loss'], -np.log([.7, .7, .7, .6]).mean())
        self.assertAlmostEqual(result['accuracy'], 1.)
        self.assertEqual(result['confusion_matrix'], [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        targets = np.eye(3)[[0, 1, 2, 0]]
        self.assertAlmostEqual(result['brier_score'], np.square(p - targets).sum(axis=1).mean())

    def test_good_calibration_alone_does_not_authorize_probability_display(self):
        y = np.array(OUTCOMES * 40)
        uniform = np.full((len(y), 3), 1 / 3)
        majority = np.tile([1., 0., 0.], (len(y), 1))
        selected = metrics(y, uniform)
        baseline = dict(majority=metrics(y, majority), training_prior=metrics(y, uniform),
                        technical_rule=metrics(y, majority))
        checks = quality_checks(selected, baseline)
        self.assertTrue(checks['checks']['calibration_ece_at_most_0_10'])
        self.assertFalse(checks['passed'])
        self.assertIn('balanced_accuracy_at_least_0_38', checks['failures'])
        self.assertIn('brier_better_than_training_prior', checks['failures'])

    def test_invalid_probability_vectors_are_refused(self):
        for p in ([[float('nan'), .5, .5]], [[.3, .3, .3]], [[-.1, .5, .6]], [[.5, .5]]):
            with self.assertRaises(ValueError):
                metrics(['UP'], p)


class HoldoutSelectionTests(unittest.TestCase):
    def test_flipping_final_holdout_labels_cannot_select_a_different_family(self):
        """Controlled labels and cheap dummy models test selection, never performance."""
        raw = market_fixture(7000)
        data = build_dataset(raw)
        sampled = data.iloc[::25]
        test_count = max(30, int(len(sampled) * .25))
        development = sampled.iloc[:-test_count]
        fold_size = min(30, max(10, (len(development) - 140) // 3))
        start_walk_forward = development.close_time.iloc[-3 * fold_size]
        start_holdout = sampled.close_time.iloc[-test_count]
        data['target'] = [OUTCOMES[i % 3] for i in range(len(data))]
        # The development evidence prefers the UP model in both runs.
        data.loc[data.close_time >= start_walk_forward, 'target'] = 'UP'
        flipped = data.copy()
        flipped.loc[flipped.close_time >= start_holdout, 'target'] = 'DOWN'

        def cheap_bundle(family, train, calibration, features):
            label = 'UP' if family == 'LogisticRegression' else 'DOWN'
            estimator = DummyClassifier(strategy='constant', constant=label).fit(train[features], train.target)
            return dict(estimator=estimator, temperature=1., features=features, family=family,
                        training_prior=[float(train.target.eq(label).mean()) for label in OUTCOMES],
                        sklearn_version=sklearn.__version__)

        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with patch('trend_pipeline.audit_news', return_value={'ready': False}), patch('trend_pipeline.fit_bundle', side_effect=cheap_bundle), patch('trend_pipeline.feature_signature', return_value='controlled-test-signature'):
                with patch('trend_pipeline.build_dataset', return_value=data):
                    first = train_research(Path(folder) / 'first', candles=raw)
                with patch('trend_pipeline.build_dataset', return_value=flipped):
                    second = train_research(Path(folder) / 'second', candles=raw)
        self.assertEqual(first['selected_model'], 'LogisticRegression')
        self.assertEqual(second['selected_model'], 'LogisticRegression')
        self.assertEqual(first['candidates']['RandomForest']['holdout']['accuracy'], 0.)
        self.assertEqual(second['candidates']['RandomForest']['holdout']['accuracy'], 1.)


if __name__ == '__main__':
    unittest.main()
