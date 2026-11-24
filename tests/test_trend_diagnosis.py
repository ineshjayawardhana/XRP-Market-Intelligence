"""Diagnostic math uses isolated fixtures, never production accuracy claims."""
import unittest

import numpy as np
import pandas as pd

from trend_diagnosis import (causal_regime_labels, class_distribution, feature_diagnostics,
                             paired_accuracy_interval, regime_analysis, wilson_interval)


class DiagnosisTests(unittest.TestCase):
    def test_distribution_retains_all_outcomes(self):
        distribution = class_distribution(['UP', 'UP', 'DOWN'])
        self.assertEqual(distribution['NEUTRAL']['count'], 0)
        self.assertAlmostEqual(distribution['UP']['fraction'], 2 / 3)

    def test_wilson_interval_known_symmetric_case_and_invalid_inputs(self):
        interval = wilson_interval(50, 100)
        self.assertAlmostEqual(interval['lower'], .40383153, places=6)
        self.assertAlmostEqual(interval['upper'], .59616847, places=6)
        for correct, count in [(0, 0), (2, 1), (-1, 10)]:
            with self.assertRaises(ValueError):
                wilson_interval(correct, count)

    def test_paired_bootstrap_is_reproducible_and_identical_predictions_have_zero_delta(self):
        y = np.array(['UP', 'DOWN', 'NEUTRAL'] * 15)
        first = paired_accuracy_interval(y, y, y, repeats=100)
        second = paired_accuracy_interval(y, y, y, repeats=100)
        self.assertEqual(first, second)
        self.assertEqual((first['accuracy_difference'], first['lower'], first['upper']), (0., 0., 0.))
        with self.assertRaises(ValueError):
            paired_accuracy_interval([], [], [])

    def frames(self):
        train = pd.DataFrame(dict(a=[0., 1., 2., 3., 4., 5.], b=[0., 2., 4., 6., 8., 10.],
                                  constant=[1.] * 6, volatility_24h=np.arange(6) + 1., ema_gap=np.arange(6) * .01))
        test = train.copy()
        test['close_time'] = pd.date_range('2025-01-01T00:00:00Z', periods=6, freq='25h')
        test['target'] = ['UP', 'DOWN', 'NEUTRAL'] * 2
        return train, test

    def test_feature_scale_and_redundancy_fit_training_only(self):
        train, test = self.frames()
        test['a'] += 10
        result = feature_diagnostics(train, test, ['a', 'b', 'constant'])
        self.assertIn('a', result['mean_shift_flags'])
        self.assertEqual(result['constant_training_features'], ['constant'])
        self.assertAlmostEqual(result['highly_correlated_training_pairs'][0]['correlation'], 1.)

    def test_regime_thresholds_are_invariant_to_test_outcomes(self):
        train, test = self.frames()
        labels, thresholds = causal_regime_labels(train, test)
        changed = test.copy()
        changed['target'] = 'UP'
        changed['future_return_24h'] = 999
        newer, same_thresholds = causal_regime_labels(train, changed)
        self.assertEqual(thresholds, same_thresholds)
        for name in labels:
            np.testing.assert_array_equal(labels[name], newer[name])

    def test_regime_groups_are_explicitly_small_and_descriptive(self):
        train, test = self.frames()
        p = np.array([[.4, .3, .3]] * len(test))
        result = regime_analysis(train, test, p)
        self.assertEqual(sum(row['sample_size'] for row in result['by_utc_month']), len(test))
        self.assertTrue(all(row['small_subgroup'] for row in result['by_utc_month']))
        self.assertIn('Post-hoc', result['limitation'])

    def test_regime_baselines_use_training_majority_without_test_prevalence_fit(self):
        train, test = self.frames()
        train['target'] = 'UP'
        test['target'] = 'DOWN'
        p = np.array([[.4, .3, .3]] * len(test))
        result = regime_analysis(train, test, p)
        majority = result['by_utc_month'][0]['baselines']['training_majority']
        self.assertEqual(majority['accuracy'], 0.)
        self.assertEqual(majority['confusion_matrix'][1][0], len(test))


if __name__ == '__main__':
    unittest.main()
