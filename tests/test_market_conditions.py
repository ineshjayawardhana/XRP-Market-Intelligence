"""Causality, uncertainty, event timing and read-only research UI fixtures."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_conditions_study import (bootstrap_indices, build_study, describe_group, news_associations, regime_rows)


def candles(n=1500, start='2022-01-01T00:00Z'):
    times = pd.date_range(start, periods=n, freq='h')
    t = np.arange(n)
    close = 2 + .1 * np.sin(t / 30) + .03 * np.cos(t / 11)
    return pd.DataFrame(dict(open_time=times, close_time=times + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        open=close, high=close + .01, low=close - .01, close=close, volume=100 + 10 * np.sin(t / 7), quote_volume=close * 100))


class StudyTests(unittest.TestCase):
    def test_regimes_are_prefix_invariant_and_targets_exact(self):
        raw = candles(2000)
        prefix = regime_rows(raw.iloc[:1500].copy())
        whole = regime_rows(raw)
        matched = whole[whole.close_time.isin(prefix.close_time)]
        columns = ['close_time', 'regime', 'high_volatility', 'high_volatility_threshold', 'future_close', 'target']
        pd.testing.assert_frame_equal(prefix[columns].reset_index(drop=True), matched[columns].reset_index(drop=True))
        self.assertTrue(prefix.close_time.diff().dropna().ge(pd.Timedelta(hours=25)).all())
        self.assertTrue((prefix.target_end - prefix.close_time).eq(pd.Timedelta(hours=24)).all())

    def test_future_mutation_cannot_change_prior_regimes(self):
        raw = candles(2000)
        first = regime_rows(raw)
        changed = raw.copy()
        changed.loc[1600:, ['open', 'high', 'low', 'close']] *= 3
        second = regime_rows(changed)
        keep = first.close_time < raw.close_time.iloc[1576]
        cols = ['regime', 'high_volatility', 'high_volatility_threshold']
        pd.testing.assert_frame_equal(first.loc[keep, cols], second.loc[keep, cols])

    def test_gap_and_incomplete_candle_windows_are_excluded(self):
        raw = candles(3500)
        raw.loc[1500, 'close_time'] -= pd.Timedelta(minutes=20)
        raw = raw.drop(index=1501).reset_index(drop=True)
        result = regime_rows(raw)
        gap = pd.Timestamp('2022-03-04T12:00Z')
        # No outcome spans missing/partial observations; warm-up restarts.
        bad_time = pd.Timestamp('2022-01-01T00:00Z') + pd.Timedelta(hours=1500)
        self.assertFalse(((result.close_time < bad_time) & (result.target_end > bad_time)).any())
        after = result[result.close_time > bad_time]
        self.assertTrue((after.close_time > bad_time + pd.Timedelta(hours=720)).all())
        self.assertEqual(result.attrs['incomplete_candles_excluded'], 1)

    def test_bootstrap_blocks_do_not_bridge_source_gaps(self):
        times = pd.date_range('2023-01-01T00:00Z', periods=12, freq='25h').append(pd.date_range('2023-03-01T00:00Z', periods=12, freq='25h'))
        indices = bootstrap_indices(times, repeats=20, block=7)
        np.testing.assert_array_equal(indices, bootstrap_indices(times, repeats=20, block=7))
        for row in indices:
            for start in range(0, 21, 7):
                selected = times[row[start:start + 7]]
                self.assertTrue((np.diff(selected.asi8) == pd.Timedelta(hours=25).value).all())

    def test_counts_frequencies_intervals_and_same_origin_baselines(self):
        frame = pd.DataFrame(dict(target=['UP', 'DOWN', 'NEUTRAL'] * 20, technical_outcome=['UP'] * 60))
        indices = bootstrap_indices(pd.date_range('2023-01-01T00:00Z', periods=60, freq='25h'), repeats=100)
        group = describe_group(frame, np.ones(60, bool), indices, 'DOWN')
        self.assertEqual(sum(value['count'] for value in group['outcomes'].values()), 60)
        for outcome in group['outcomes'].values():
            self.assertAlmostEqual(outcome['historical_frequency'], 1 / 3)
            self.assertLessEqual(outcome['block_95']['lower'], outcome['block_95']['upper'])
        self.assertEqual(group['baselines']['Reference-majority']['matches'], 20)
        empty = describe_group(frame, np.zeros(60, bool), indices, 'DOWN')
        self.assertIsNone(empty['outcomes']['UP']['historical_frequency'])

    def test_full_study_keeps_reference_earlier_and_highvol_overlap_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            report, frame = build_study(candles(24000, '2021-11-01T00:00Z'), Path(folder), repeats=20)
        self.assertLess(pd.Timestamp(report['reference']['end']) + pd.Timedelta(hours=24), pd.Timestamp(report['evaluation']['start']))
        self.assertEqual(sum(report['groups'][key]['samples'] for key in ('Bullish', 'Bearish', 'Sideways')), len(frame))
        self.assertFalse(report['model_validation_status_changed'])
        self.assertFalse(report['news']['estimable'])

    def test_news_uses_entry_after_availability_and_excludes_unknown_outcomes(self):
        raw = candles(2000)
        available = raw.close_time.iloc[1200] + pd.Timedelta(minutes=10)
        events = [dict(id='fixture', role='gold', label='Neutral_Market_News', feature_available_at=available),
                  dict(id='late', role='silver', label='Hype_Speculation', feature_available_at=raw.close_time.iloc[-1])]
        with patch('market_conditions_study.load_timed_news', return_value=(events, {}, {}, {})), patch('market_conditions_study.audit_news', return_value={}):
            result = news_associations(Path('.'), raw, regime_rows(raw))
        self.assertEqual(result['matured_nonoverlapping_events'], 1)
        self.assertGreaterEqual(pd.Timestamp(result['events'][0]['entry_at']), available)
        self.assertGreater(result['excluded']['no_mature_gap_free_outcome'], 0)
        self.assertIsNone(result['mean_event_minus_control_return'])


class ResearchUITests(unittest.TestCase):
    def test_new_view_missing_report_and_actual_fixture_both_themes(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'XRP_DASHBOARD_ROOT': folder,
                'XRP_DASHBOARD_OFFLINE': '1', 'XRP_DASHBOARD_CACHE_DIR': str(Path(folder) / 'cache')}):
            st.cache_data.clear()
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            app.radio[0].set_value('Market Conditions Research').run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any('No readable study' in value.value for value in app.info))
            # Real report copied to a temporary root; never mutate source data.
            original = Path(__file__).resolve().parents[1] / 'reports' / 'market_conditions'
            if (original / 'report.json').exists():
                import shutil
                shutil.copytree(original, Path(folder) / 'reports' / 'market_conditions')
                for theme in ('Light', 'Dark'):
                    next(box for box in app.selectbox if box.label == 'Appearance').select(theme).run(timeout=30)
                    self.assertFalse(app.exception)
                    self.assertTrue(app.get('plotly_chart'))
                    self.assertTrue(any('not validated future probabilities' in value.value for value in app.warning))
                app.selectbox(key='regime_study_year').select('2024').run(timeout=30)
                self.assertFalse(app.exception)
                registry = Path(folder) / 'data' / 'trend_experiments' / 'reserve_registry.json'
                registry.parent.mkdir(parents=True)
                registry.write_text(json.dumps(dict(entries=[dict(status='reserved', protocol_sha256='fixture',
                    start='2024-06-01T00:00Z', end='2024-07-01T00:00Z')])))
                app.run(timeout=30)
                self.assertTrue(any('Study withheld' in value.value for value in app.warning))
            st.cache_data.clear()


if __name__ == '__main__':
    unittest.main()
