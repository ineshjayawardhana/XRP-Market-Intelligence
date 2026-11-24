"""Synthetic fixtures test experimental feature timing, never model performance."""
import unittest

import numpy as np
import pandas as pd

from trend_experiment_features import (EXPANDED_FEATURES, EXTRA_FEATURES, FEATURE_GROUPS,
    causal_regimes, experiment_dataset, experiment_features, segmented_experiment_dataset)
from trend_pipeline import FEATURES


def candles(count=700):
    opens = pd.date_range('2023-01-01T00:00:00Z', periods=count, freq='h')
    steps = np.arange(count, dtype=float)
    close = 2. + .03 * np.sin(steps / 9.) + .00002 * steps
    return pd.DataFrame(dict(open_time=opens, close_time=opens + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        open=close, high=close + .01, low=close - .01, close=close,
        volume=100. + 20. * np.cos(steps / 13.), quote_volume=close * 100.))


class ExperimentFeatureTests(unittest.TestCase):
    def test_expanded_whitelist_is_moderate_and_excludes_targets(self):
        self.assertTrue(45 <= len(EXPANDED_FEATURES) <= 55)
        self.assertEqual(len(EXPANDED_FEATURES), len(set(EXPANDED_FEATURES)))
        self.assertEqual(EXPANDED_FEATURES[:len(FEATURES)], FEATURES)
        self.assertEqual(set(EXTRA_FEATURES), set(EXPANDED_FEATURES) - set(FEATURES))
        self.assertEqual(sum(map(len, FEATURE_GROUPS.values())), len(EXPANDED_FEATURES))
        self.assertFalse({'future_close', 'future_return_24h', 'target', 'target_end'}.intersection(EXPANDED_FEATURES))

    def test_prefix_and_future_mutations_do_not_change_past_features(self):
        full = candles()
        expected = experiment_features(full.iloc[:400])[EXPANDED_FEATURES]
        pd.testing.assert_frame_equal(expected, experiment_features(full).iloc[:400][EXPANDED_FEATURES])
        changed = full.copy()
        changed.loc[400:, ['open', 'high', 'low', 'close']] *= 2.
        changed.loc[400:, 'volume'] *= 5.
        pd.testing.assert_frame_equal(expected, experiment_features(changed).iloc[:400][EXPANDED_FEATURES])

    def test_target_horizon_matches_exact_prices_and_unknown_rows_are_excluded(self):
        raw = candles()
        dataset = experiment_dataset(raw)
        self.assertGreater(len(dataset), 0)
        self.assertTrue((dataset.target_end - dataset.close_time).eq(pd.Timedelta(hours=24)).all())
        np.testing.assert_allclose(dataset.future_close, dataset.target_end.map(raw.set_index('close_time').close))
        np.testing.assert_allclose(dataset.future_return_24h, dataset.future_close / dataset.close - 1)
        self.assertEqual(dataset.close_time.max(), raw.close_time.iloc[-25])
        self.assertTrue(np.isfinite(dataset[EXPANDED_FEATURES].to_numpy()).all())

    def test_lagged_returns_and_changes_have_the_stated_time_offsets(self):
        frame = experiment_features(candles())
        index = 400
        for lag in (1, 3, 6, 24):
            self.assertAlmostEqual(frame.loc[index, f'return_1h_lag_{lag}h'], frame.loc[index - lag, 'return_1h'])
        for lag in (1, 6, 24):
            self.assertAlmostEqual(frame.loc[index, f'rsi_change_{lag}h'], (frame.loc[index, 'rsi'] - frame.loc[index - lag, 'rsi']) / 100)
            self.assertAlmostEqual(frame.loc[index, f'macd_hist_change_{lag}h'], frame.loc[index, 'macd_hist_relative'] - frame.loc[index - lag, 'macd_hist_relative'])

    def test_high_low_and_volume_reference_excludes_current_candle(self):
        raw = candles()
        frame = experiment_features(raw)
        index = 400
        for hours in (24, 168):
            prior = raw.iloc[index - hours:index]
            self.assertAlmostEqual(frame.loc[index, f'distance_high_{hours}h'], raw.close.iloc[index] / prior.high.max() - 1)
            self.assertAlmostEqual(frame.loc[index, f'distance_low_{hours}h'], raw.close.iloc[index] / prior.low.min() - 1)
            self.assertAlmostEqual(frame.loc[index, f'volume_relative_{hours}h'], raw.volume.iloc[index] / prior.volume.mean())

    def test_missing_or_forming_candles_are_rejected_and_input_unchanged(self):
        raw = candles()
        before = raw.copy(deep=True)
        experiment_dataset(raw)
        pd.testing.assert_frame_equal(raw, before)
        with self.assertRaises(ValueError):
            experiment_features(raw.drop(index=300))
        with self.assertRaises(ValueError):
            experiment_features(raw, now=raw.open_time.iloc[-1])

    def test_zero_volume_denominator_is_missing_and_not_filled(self):
        raw = candles()
        raw.loc[200:367, 'volume'] = 0.
        frame = experiment_features(raw)
        self.assertTrue(pd.isna(frame.loc[368, 'volume_relative_168h']))
        self.assertNotIn(raw.close_time.iloc[368], set(experiment_dataset(raw).close_time))
        self.assertFalse(np.isinf(frame[EXPANDED_FEATURES].to_numpy()).any())

    def test_volatility_regime_uses_training_threshold_only(self):
        frame = experiment_features(candles())
        training, testing = frame.iloc[200:400], frame.iloc[400:]
        baseline = causal_regimes(testing, training)
        changed = testing.copy()
        changed.loc[changed.index[100]:, 'volatility_24h'] *= 100
        actual = causal_regimes(changed, training)
        self.assertEqual(actual.attrs['training_volatility_median'], training.volatility_24h.median())
        self.assertEqual(baseline.attrs['training_volatility_median'], actual.attrs['training_volatility_median'])
        pd.testing.assert_frame_equal(baseline.iloc[:100], actual.iloc[:100])

    def test_segmented_dataset_restarts_warmup_and_never_crosses_gap_targets(self):
        raw = candles(900)
        gap = raw.drop(index=450).reset_index(drop=True)
        result = segmented_experiment_dataset(gap)
        first = result[result.contiguous_segment.eq(1)]
        second = result[result.contiguous_segment.eq(2)]
        self.assertGreater(len(first), 0)
        self.assertGreater(len(second), 0)
        self.assertLessEqual(first.target_end.max(), raw.close_time.iloc[449])
        self.assertGreaterEqual(second.close_time.min(), raw.close_time.iloc[451] + pd.Timedelta(hours=168))
        self.assertEqual(result.attrs['missing_hours'], 1)
        self.assertEqual(len(result.attrs['gaps']), 1)
        self.assertEqual(len(result.attrs['segments']), 2)
        with self.assertRaises(ValueError):
            experiment_dataset(gap)  # Production-style strict adapter remains strict.

    def test_segment_restart_is_independent_of_prices_before_gap(self):
        raw = candles(900).drop(index=450).reset_index(drop=True)
        original = segmented_experiment_dataset(raw)
        changed = raw.copy()
        changed.loc[:449, ['open', 'high', 'low', 'close']] *= 3.
        changed.loc[:449, 'volume'] *= 5.
        actual = segmented_experiment_dataset(changed)
        pd.testing.assert_frame_equal(original.loc[original.contiguous_segment.eq(2), EXPANDED_FEATURES],
                                      actual.loc[actual.contiguous_segment.eq(2), EXPANDED_FEATURES])

    def test_segmented_dataset_future_append_does_not_change_eligible_past_features(self):
        raw = candles(900).drop(index=450).reset_index(drop=True)
        prefix = segmented_experiment_dataset(raw.iloc[:750])
        whole = segmented_experiment_dataset(raw)
        common = whole[whole.close_time.isin(prefix.close_time)]
        np.testing.assert_array_equal(prefix.close_time, common.close_time)
        np.testing.assert_allclose(prefix[EXPANDED_FEATURES], common[EXPANDED_FEATURES])

    def test_segmented_adapter_validates_even_short_segments_and_has_empty_schema(self):
        raw = candles(120).drop(index=60).reset_index(drop=True)
        empty = segmented_experiment_dataset(raw)
        self.assertTrue(empty.empty)
        self.assertTrue(set(EXPANDED_FEATURES + ['target', 'target_end', 'contiguous_segment']).issubset(empty))
        for corrupted in (pd.concat([raw, raw.iloc[-1:]], ignore_index=True), raw.iloc[::-1]):
            with self.assertRaises(ValueError):
                segmented_experiment_dataset(corrupted)
        bad = raw.copy()
        bad.loc[20, 'low'] = bad.close.iloc[20] + .1
        with self.assertRaises(ValueError):
            segmented_experiment_dataset(bad)
        with self.assertRaises(ValueError):
            segmented_experiment_dataset(raw, now=raw.open_time.iloc[-1])


if __name__ == '__main__':
    unittest.main()
