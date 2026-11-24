"""Public collector resume/gap/cooldown checks; no network is used."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from experiment_market_data import collect_experiment_candles
from trend_market_data import save_frame


class ExperimentalCollectionTests(unittest.TestCase):
    def fixture(self):
        end = pd.Timestamp.now(tz='UTC').floor('h')
        opens = pd.date_range(end - pd.Timedelta(days=8), end - pd.Timedelta(hours=1), freq='h')
        return pd.DataFrame(dict(open_time=opens, close_time=opens + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            open=1., high=1.1, low=.9, close=1., volume=10., quote_volume=10.))

    def test_forward_resume_retains_gap_and_does_not_redownload_completed_cache(self):
        raw = self.fixture().drop(index=50).reset_index(drop=True)
        seed = raw.iloc[:100]
        session = Mock()
        def fetch(url, params, **kwargs):
            start = pd.to_datetime(params['startTime'], unit='ms', utc=True)
            rows = raw[raw.open_time >= start]
            response = Mock(status_code=200)
            response.json.return_value = [[int(row.open_time.timestamp() * 1000), row.open, row.high, row.low,
                row.close, row.volume, int(row.close_time.timestamp() * 1000), row.quote_volume, 0, 0, 0, 0]
                for row in rows.itertuples()]
            return response
        session.get.side_effect = fetch
        with tempfile.TemporaryDirectory() as folder, patch('experiment_market_data.requests.Session') as constructor:
            constructor.return_value.__enter__.return_value = session
            save_frame(Path(folder) / 'xrp_hourly.csv', seed)
            frame = collect_experiment_candles(folder, days=1)
            self.assertEqual(len(frame), len(raw))
            quality = json.loads((Path(folder) / 'data_quality.json').read_text())
            self.assertEqual(quality['missing_hour_count'], 1)
            calls = session.get.call_count
            collect_experiment_candles(folder, days=1)
            self.assertEqual(session.get.call_count, calls)

    def test_429_cooldown_is_respected_without_retry(self):
        session = Mock()
        session.get.return_value = Mock(status_code=429, headers={'Retry-After': '120'})
        with tempfile.TemporaryDirectory() as folder, patch('experiment_market_data.requests.Session') as constructor:
            constructor.return_value.__enter__.return_value = session
            with self.assertRaisesRegex(RuntimeError, 'rate limit'):
                collect_experiment_candles(folder, days=1)
            with self.assertRaisesRegex(RuntimeError, 'cooldown'):
                collect_experiment_candles(folder, days=1)
            self.assertEqual(session.get.call_count, 1)

    def test_invalid_existing_cache_is_rejected_before_requests(self):
        raw = self.fixture()
        with tempfile.TemporaryDirectory() as folder, patch('experiment_market_data.requests.Session') as constructor:
            save_frame(Path(folder) / 'xrp_hourly.csv', pd.concat([raw, raw.iloc[-1:]], ignore_index=True))
            with self.assertRaisesRegex(ValueError, 'duplicated'):
                collect_experiment_candles(folder, days=1)
            constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
