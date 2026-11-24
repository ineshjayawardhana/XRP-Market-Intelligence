"""Independent audit and future-reserve safeguards; only temporary data writes."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit_market_conditions import candle_inventory, window_overlap
from market_conditions_study import active_reserves, bootstrap_indices, build_study, outside_reserves
from trend_market_data import save_frame


def candles():
    times = pd.date_range('2021-11-01T00:00Z', periods=24000, freq='h')
    values = 2 + .1 * np.sin(np.arange(len(times)) / 30)
    return pd.DataFrame(dict(open_time=times, close_time=times + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        open=values, close=values, high=values + .01, low=values - .01, volume=100., quote_volume=values * 100))


class RegimeAuditTests(unittest.TestCase):
    def test_interval_overlap_detected_when_exact_origins_differ(self):
        starts = pd.date_range('2023-01-01T00:00Z', periods=3, freq='25h')
        frame = pd.DataFrame(dict(close_time=starts, target_end=starts + pd.Timedelta(hours=24)))
        ml = starts + pd.Timedelta(hours=1)
        result = window_overlap(frame, ml, ml + pd.Timedelta(hours=24))
        self.assertEqual(result['exact_origin_matches'], 0)
        self.assertEqual(result['descriptive_observations_overlapping'], 3)
        self.assertEqual(result['ml_intervals_overlapping'], 3)

    def test_unfinished_reserve_is_excluded_with_boundary_purge(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data' / 'trend_experiments' / 'reserve_registry.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(entries=[dict(status='reserved', protocol_sha256='fixture',
                start='2023-06-01T00:00Z', end='2023-07-01T00:00Z')])))
            report, rows = build_study(candles(), Path(folder), repeats=20)
            self.assertGreater(report['ml_separation']['excluded_observations'], 0)
            self.assertFalse(((rows.close_time < pd.Timestamp('2023-07-01T00:00Z')) &
                              (rows.target_end >= pd.Timestamp('2023-06-01T00:00Z'))).any())
            self.assertEqual(report['ml_separation']['active_reserve_windows'], 1)

    def test_evaluated_reserve_is_descriptive_history_not_new_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data' / 'trend_experiments' / 'reserve_registry.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(entries=[dict(status='evaluated')])))
            self.assertEqual(active_reserves(folder), [])
            path.write_text(json.dumps(dict(entries=[dict(status='unknown')])))
            with self.assertRaises(ValueError):
                active_reserves(folder)

    def test_bootstrap_does_not_silently_assume_independence(self):
        times = pd.date_range('2023-01-01T00:00Z', periods=10, freq='50h')
        with self.assertRaisesRegex(ValueError, 'independent-row fallback'):
            bootstrap_indices(times)

    def test_inventory_distinguishes_raw_partial_and_duplicate_rows(self):
        raw = candles().iloc[:200].copy()
        raw.loc[10, 'close_time'] -= pd.Timedelta(minutes=20)
        raw = pd.concat([raw, raw.iloc[-1:]], ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.csv'
            save_frame(path, raw)
            _, result = candle_inventory(path)
        self.assertEqual(result['rows'], 201)
        self.assertEqual(result['unique_hour_timestamps'], 200)
        self.assertEqual(result['duplicate_open_timestamps'], 1)
        self.assertEqual(result['incomplete_rows'], 1)

    def test_reserve_filter_includes_exact_target_boundary(self):
        starts = pd.Series(pd.to_datetime(['2023-05-31T00:00Z', '2023-05-30T00:00Z'], utc=True))
        ends = starts + pd.Timedelta(hours=24)
        windows = [dict(start=pd.Timestamp('2023-06-01T00:00Z'), end=pd.Timestamp('2023-07-01T00:00Z'))]
        self.assertEqual(outside_reserves(starts, ends, windows).tolist(), [False, True])


if __name__ == '__main__':
    unittest.main()
