"""Independent isolated-experiment checks. All numerical data are fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import trend_experiments as experiment
from trend_experiment_features import causal_regimes
from trend_metrics import OUTCOMES, metrics


def candles(count=9600):
    opens = pd.date_range('2021-12-20T00:00:00Z', periods=count, freq='h')
    step = np.arange(count)
    price = 2 + .1 * np.sin(step / 30) + .03 * np.cos(step / 7)
    return pd.DataFrame(dict(open_time=opens, close_time=opens + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        open=price, close=price, high=price + .01, low=price - .01,
        volume=100 + 10 * np.sin(step / 19), quote_volume=price * 100))


def origins(count=400, start='2022-01-01T00:59:59.999Z'):
    times = pd.date_range(start, periods=count, freq='25h')
    step = np.arange(count)
    return pd.DataFrame(dict(close_time=times, target_end=times + pd.Timedelta(hours=24),
        x=step, target=np.array(OUTCOMES)[step % 3], future_return_24h=np.where(step % 3 == 0, .02, -.02),
        technical=np.array(['Bullish', 'Bearish', 'Neutral'])[step % 3],
        ema_gap=np.sin(step) * .01, volatility_24h=.01 + step * .00001))


def small_protocol():
    return dict(grid_anchor='2022-01-01T00:59:59.999Z', training_start='2022-01-01T00:00:00Z',
        reserve_start='2024-05-01T00:00:00Z', reserve_end_exclusive='2024-11-01T00:00:00Z',
        horizon_hours=24, stride_hours=25, calibration_origins=45,
        inner_folds=2, inner_test_origins=20, outer_folds=2, outer_test_origins=30,
        feature_sets={'Fixture': ['x']}, grids={'LogisticRegression': [{'C': .1}, {'C': 1.}]})


def cheap_bundle(family, params, train, calibration, features):
    if not train.target_end.max() < calibration.close_time.min():
        raise AssertionError('Labels cross training/calibration boundary')
    return dict(family=family, params=params, features=features, temperature=1.,
        training_prior=[float(train.target.eq(label).mean()) for label in OUTCOMES])


def cheap_predictions(bundle, frame, calibrated=True):
    called = np.asarray(frame.x, dtype=int) % 3
    if bundle['params']['C'] != .1:
        called = (called + 1) % 3
    p = np.ones((len(frame), 3)) * .1
    p[np.arange(len(frame)), called] = .8
    return p


class ReservationTests(unittest.TestCase):
    def test_protocol_is_frozen_and_overlapping_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = small_protocol()
            identifier, entry = experiment.reserve_protocol(directory, protocol)
            path = Path(directory) / 'protocols' / (identifier + '.json')
            before = path.read_bytes()
            same, _ = experiment.reserve_protocol(directory, protocol)
            self.assertEqual(same, identifier)
            self.assertEqual(path.read_bytes(), before)
            changed = dict(protocol, stride_hours=26)
            with self.assertRaisesRegex(ValueError, 'reserve|protocol|committed'):
                experiment.reserve_protocol(directory, changed)

    def test_development_only_does_not_rewrite_existing_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = small_protocol()
            identifier, _ = experiment.reserve_protocol(directory, protocol)
            path = Path(directory) / 'protocols' / (identifier + '.json')
            before = path.read_bytes()
            registry = (Path(directory) / 'reserve_registry.json').read_bytes()
            experiment.reserve_protocol(directory, protocol, development_only=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual((Path(directory) / 'reserve_registry.json').read_bytes(), registry)

    def test_evaluated_reserve_with_missing_report_cannot_be_scored_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = small_protocol()
            folder = root / 'data' / 'trend_experiments'
            identifier, _ = experiment.reserve_protocol(folder, protocol)
            experiment.update_reserve(folder, identifier, status='evaluated', report_sha256='missing')
            with patch('trend_experiments.protocol_definition', return_value=protocol):
                with self.assertRaisesRegex(ValueError, 'report|evaluated|reserve'):
                    experiment.run_investigation(root, offline=True, progress=lambda _: None)

    def test_opened_reserve_cannot_retrain_and_reuse_its_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = small_protocol()
            folder = root / 'data' / 'trend_experiments'
            identifier, _ = experiment.reserve_protocol(folder, protocol)
            experiment.update_reserve(folder, identifier, status='opened', model_sha256='frozen')
            with patch('trend_experiments.protocol_definition', return_value=protocol), patch('trend_experiments.collect_candles') as collection:
                with self.assertRaisesRegex(ValueError, 'opened|reserve|recovery'):
                    experiment.run_investigation(root, progress=lambda _: None)
            collection.assert_not_called()

    def test_prior_production_exposure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'data' / 'trend'
            folder.mkdir(parents=True)
            pd.DataFrame({'close_time': ['2024-06-01T00:00:00Z']}).to_csv(folder / 'xrp_hourly.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'already'):
                experiment.check_unseen_reserve(root, small_protocol())


class UnopenedProtocolAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'data' / 'trend_experiments'
        self.protocol = small_protocol()
        self.identifier, _ = experiment.reserve_protocol(self.folder, self.protocol)
        self.run = self.root / 'artifacts' / 'trend_experiments' / self.identifier[:16]
        self.run.mkdir(parents=True)
        (self.run / 'protocol.json').write_text('{}')
        (self.run / 'production_manifest.json').write_text('[]')

    def test_unfitted_metadata_only_reservation_can_be_explicitly_amended(self):
        frozen = self.folder / 'protocols' / (self.identifier + '.json')
        before = frozen.read_bytes()
        reason = 'Public candle gap found before features, fitting or reserve-target construction.'
        experiment.supersede_unopened_protocol(self.folder, self.identifier, reason)
        registry = json.loads((self.folder / 'reserve_registry.json').read_text())
        old = registry['entries'][0]
        self.assertEqual(old['status'], 'superseded_unopened')
        self.assertEqual(old['reason'], reason)
        self.assertIn('superseded_at', old)
        self.assertEqual(frozen.read_bytes(), before)
        changed = dict(self.protocol, gap_policy='separate contiguous segments; no interpolation')
        new_identifier, new_entry = experiment.reserve_protocol(self.folder, changed)
        self.assertNotEqual(new_identifier, self.identifier)
        self.assertEqual(new_entry['status'], 'reserved')

    def test_opened_and_evaluated_reserves_cannot_be_amended(self):
        for status in ('opened', 'evaluated'):
            with self.subTest(status=status):
                experiment.update_reserve(self.folder, self.identifier, status=status)
                with self.assertRaises(ValueError):
                    experiment.supersede_unopened_protocol(self.folder, self.identifier, 'Source gap')

    def test_model_features_predictions_results_or_unexpected_files_prevent_amendment(self):
        for filename in ('model.joblib', 'selection.json', 'development_dataset.csv', 'outer_Fixture.csv',
                         'final_predictions.csv', 'report.json', 'unexpected.txt'):
            with self.subTest(filename=filename):
                path = self.run / filename
                path.write_text('Evidence that this run is not metadata-only')
                with self.assertRaises(ValueError):
                    experiment.supersede_unopened_protocol(self.folder, self.identifier, 'Source gap')
                path.unlink()

    def test_amendment_requires_nonblank_reason_and_known_identifier(self):
        for reason in ('', '   '):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                experiment.supersede_unopened_protocol(self.folder, self.identifier, reason)
        with self.assertRaises(ValueError):
            experiment.supersede_unopened_protocol(self.folder, 'not-a-reserved-protocol', 'Source gap')


class ChronologicalExperimentTests(unittest.TestCase):
    def test_aligned_grid_and_targets_stay_inside_requested_period(self):
        protocol = small_protocol()
        raw = candles()
        end = '2022-11-01T00:00:00Z'
        data = experiment.aligned_origins(raw, protocol, end)
        self.assertTrue(data.target_end.lt(pd.Timestamp(end)).all())
        self.assertTrue(data.close_time.diff().dropna().eq(pd.Timedelta(hours=25)).all())
        offsets = (data.close_time - pd.Timestamp(protocol['grid_anchor'])) / pd.Timedelta(hours=1)
        self.assertTrue(np.isclose(offsets % 25, 0).all())
        changed = raw.copy()
        later = changed.close_time >= pd.Timestamp(end)
        changed.loc[later, ['open', 'close', 'high', 'low']] *= 10
        pd.testing.assert_frame_equal(data, experiment.aligned_origins(changed, protocol, end))

    def test_histogram_boosting_does_not_randomly_split_for_early_stopping(self):
        model = experiment.make_estimator('HistGradientBoosting', {'max_leaf_nodes': 7})
        self.assertFalse(model.early_stopping)
        self.assertEqual(model.random_state, 42)

    def test_tuning_uses_only_mature_earlier_training_and_calibration(self):
        data, protocol = origins(), small_protocol()
        cutoffs = []
        def fit(family, params, train, calibration, features):
            cutoffs.append((train.target_end.max(), calibration.close_time.min(), calibration.target_end.max()))
            return cheap_bundle(family, params, train, calibration, features)
        with patch('trend_experiments.fit_candidate', side_effect=fit), patch('trend_experiments.predict_bundle', side_effect=cheap_predictions):
            params, trials = experiment.tune(data, 'LogisticRegression', ['x'], protocol)
        self.assertEqual(params, {'C': .1})
        for trial in trials:
            for fold in trial['folds']:
                self.assertLess(pd.Timestamp(fold['train_target_end']), pd.Timestamp(fold['calibration_start']))
                self.assertLess(pd.Timestamp(fold['calibration_target_end']), pd.Timestamp(fold['test_start']))
        self.assertEqual(len(cutoffs), 4)

    def test_outer_tuning_receives_no_outer_validation_labels(self):
        data, protocol = origins(), small_protocol()
        actual_tune = experiment.tune
        outer_starts = [validation.close_time.min() for _, validation in experiment.rolling_blocks(data, 2, 30)]
        calls = []
        def restricted_tune(earlier, family, features, received):
            expected_start = outer_starts[len(calls)]
            self.assertLess(earlier.target_end.max(), expected_start)
            calls.append(expected_start)
            return actual_tune(earlier, family, features, received)
        with patch('trend_experiments.fit_candidate', side_effect=cheap_bundle), patch('trend_experiments.predict_bundle', side_effect=cheap_predictions), patch('trend_experiments.tune', side_effect=restricted_tune):
            selected, candidates = experiment.nested_development(data, protocol)
        self.assertEqual(selected, 'LogisticRegression/Fixture')
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(candidates[selected]['predictions']), 60)

    def test_final_labels_are_constructed_only_after_frozen_selection_and_open_ledger(self):
        protocol = small_protocol()
        development, final = origins(), origins(120, '2024-05-01T00:59:59.999Z')
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'data' / 'trend_experiments'
            cache = folder / 'cache'
            cache.mkdir(parents=True)
            candles(200).to_csv(cache / 'xrp_hourly.csv', index=False)
            run = root / 'artifacts' / 'trend_experiments' / experiment.digest(protocol)[:16]
            def construct(raw, received, end):
                calls.append(end)
                if end == protocol['reserve_start']:
                    return development.copy()
                self.assertTrue((run / 'selection.json').exists())
                self.assertTrue((run / 'model.joblib').exists())
                registry = json.loads((folder / 'reserve_registry.json').read_text())
                self.assertEqual(registry['entries'][0]['status'], 'opened')
                self.assertIn('raw_cache_sha256', registry['entries'][0])
                return pd.concat([development, final], ignore_index=True)
            def nested(train, received, progress):
                self.assertLess(train.target_end.max(), pd.Timestamp(protocol['reserve_start']))
                bundle = cheap_bundle('LogisticRegression', {'C': .1}, train.iloc[:-45], train.iloc[-45:], ['x'])
                rows = experiment.score_rows(bundle, train.iloc[-60:], train.iloc[:-45], train.iloc[-45:])
                baselines = {key: experiment.rows_metrics(rows, key) for key in ('majority', 'training_prior', 'technical_rule')}
                return 'LogisticRegression/Fixture', {'LogisticRegression/Fixture': dict(family='LogisticRegression',
                    feature_set='Fixture', metrics=experiment.rows_metrics(rows), baselines=baselines,
                    macro_f1_gain_vs_majority_by_fold=[.1, .1], macro_f1_gain_vs_technical_by_fold=[.1, .1], predictions=rows)}
            with patch('trend_experiments.protocol_definition', return_value=protocol), patch('trend_experiments.aligned_origins', side_effect=construct), patch('trend_experiments.nested_development', side_effect=nested), patch('trend_experiments.tune', return_value=({'C': .1}, [])), patch('trend_experiments.fit_candidate', side_effect=cheap_bundle), patch('trend_experiments.predict_bundle', side_effect=cheap_predictions), patch('trend_experiments.diagnose_existing', return_value={}), patch('trend_experiments.production_manifest', return_value=[]):
                report = experiment.run_investigation(root, offline=True, progress=lambda _: None)
            self.assertEqual(calls, [protocol['reserve_start'], protocol['reserve_end_exclusive']])
            self.assertTrue(report['selected_before_final_reserve'])
            self.assertTrue(report['final_reserve_scored'])
            self.assertFalse(report['production_replaced'])

    def test_block_intervals_are_paired_deterministic_and_labeled_differences(self):
        labels = np.array(OUTCOMES)[np.arange(60) % 3]
        correct, wrong = np.arange(60) % 3, (np.arange(60) + 1) % 3
        first = experiment.paired_block_intervals(labels, correct, wrong, repeats=100, seed=7)
        self.assertEqual(first, experiment.paired_block_intervals(labels, correct, wrong, repeats=100, seed=7))
        for metric in first.values():
            self.assertAlmostEqual(metric['difference'], 1.)
            self.assertGreater(metric['lower'], 0)
        equal = experiment.paired_block_intervals(labels, correct, correct, repeats=30)
        self.assertTrue(all(value == 0 for result in equal.values() for value in result.values()))

    def test_regime_threshold_is_fitted_only_on_training(self):
        train, evaluation = origins(100), origins(20, '2024-01-01T00:00:00Z')
        first = causal_regimes(evaluation, train)
        evaluation.volatility_24h *= 100
        second = causal_regimes(evaluation, train)
        self.assertEqual(first.attrs['training_volatility_median'], second.attrs['training_volatility_median'])
        self.assertAlmostEqual(first.attrs['training_volatility_median'], float(train.volatility_24h.median()))


if __name__ == '__main__':
    unittest.main()
