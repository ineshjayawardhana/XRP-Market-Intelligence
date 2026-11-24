"""Dashboard inference display tests; all probability values are isolated fixtures."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import dashboard_data  # Import native dependencies before isolating sys.modules.


class TrendDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'XRP_DASHBOARD_ROOT': str(self.root),
            'XRP_DASHBOARD_CACHE_DIR': str(self.root / 'cache'), 'XRP_DASHBOARD_OFFLINE': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)
        import streamlit as st
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def fixture(self, validated=False):
        return dict(validated=validated, probabilities={'UP': .6, 'DOWN': .2, 'NEUTRAL': .2},
            experimental_label=None if validated else 'UP', message='Predictive probabilities are not yet validated.',
            limitation='Isolated test fixture; not production results.', model_version='fixture-version',
            feature_as_of=pd.Timestamp.now(tz='UTC').isoformat(), generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
            evaluated_at='2026-11-01T00:00:00Z', sample_size=120, test_start='2026-01-01T00:00:00Z',
            test_end='2026-10-01T00:00:00Z', quality_failures=[] if validated else ['Fixture calibration did not pass.'])

    def run_app(self, result):
        from streamlit.testing.v1 import AppTest
        module = types.ModuleType('one_hour_trend')
        module.live_forecast = lambda candles, root: result
        with patch.dict(sys.modules, {'one_hour_trend': module}):
            return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)

    def test_failed_gate_exposes_class_but_never_probability_numbers(self):
        app = self.run_app(self.fixture(False))
        self.assertFalse(app.exception)
        values = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(values['UP'], 'Not validated')
        self.assertEqual(values['DOWN'], 'Not validated')
        self.assertEqual(values['NEUTRAL'], 'Not validated')
        self.assertNotIn('60.0%', values.values())
        self.assertTrue(any('Experimental model signal' in item.value and 'UP' in item.value for item in app.warning))
        self.assertTrue(any('fixture-version' in item.value for item in app.caption))

    def test_passed_gate_displays_only_returned_research_probabilities(self):
        app = self.run_app(self.fixture(True))
        self.assertFalse(app.exception)
        values = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(values['UP'], '60.0%')
        self.assertEqual(values['DOWN'], '20.0%')
        self.assertEqual(values['NEUTRAL'], '20.0%')
        self.assertFalse(any('Experimental model signal' in item.value for item in app.warning))

    def test_advanced_view_renders_recorded_metrics_and_failed_gate(self):
        metrics = dict(sample_size=120, accuracy=.5, balanced_accuracy=.5, macro_f1=.49,
            calibration_ece=.15, brier_score=.7, log_loss=1.2,
            outcome_distribution={'UP': 40, 'DOWN': 40, 'NEUTRAL': 40},
            confusion_matrix=[[20, 10, 10], [10, 20, 10], [10, 10, 20]],
            classification_report={'UP': {'precision': .5, 'recall': .5, 'f1-score': .5, 'support': 40}},
            reliability_bins=[{'count': 120, 'mean_confidence': .65, 'accuracy': .5}])
        fold = dict(fold=0, train_end='2025-01-01T00:00:00Z', test_start='2025-02-01T00:00:00Z',
                    test_end='2025-03-01T00:00:00Z', train_samples=120, test_samples=30,
                    accuracy=.5, balanced_accuracy=.5, macro_f1=.49)
        report = dict(schema_version=2, model_version='fixture-version', evaluated_at='2026-11-01T00:00:00Z',
            evaluation_type='retrospective_replay', selected_model='LogisticRegression', selected_before_holdout=True,
            data=dict(hourly_candles=20000, daily_samples=800, source='isolated test fixture',
                      start='2024-01-01T00:00:00Z', end='2026-10-01T00:00:00Z'),
            split=dict(train_start='2024-01-01T00:00:00Z', train_end='2025-01-01T00:00:00Z',
                       calibration_start='2025-01-03T00:00:00Z', calibration_end='2025-05-01T00:00:00Z',
                       test_start='2025-05-03T00:00:00Z', test_end='2026-10-01T00:00:00Z',
                       train_samples=500, calibration_samples=100, test_samples=120, purge_hours=24),
            candidates={'LogisticRegression': {'holdout': metrics, 'walk_forward': dict(metrics, folds=[fold])}},
            baselines={'majority': metrics, 'training_prior': metrics, 'technical_rule': metrics}, holdout=metrics,
            quality=dict(passed=False, failures=['Fixture calibration did not pass.'], checks={'ece': False}),
            news={'historical_comparison_ready': False}, limitations=['Test fixture, not actual market results.'])
        folder = self.root / 'artifacts'
        folder.mkdir()
        (folder / 'trend_24h_report.json').write_text(json.dumps(report), encoding='utf-8')
        app = self.run_app(self.fixture(False))
        app.radio[0].set_value('Advanced research').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('remains experimental' in item.value for item in app.warning))
        self.assertTrue(any('LogisticRegression' in item.value for item in app.markdown))
        self.assertGreaterEqual(len(app.dataframe), 7)
        values = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(values['Accuracy'], '0.500')
        self.assertEqual(values['Held-out examples'], '120')


if __name__ == '__main__':
    unittest.main()
