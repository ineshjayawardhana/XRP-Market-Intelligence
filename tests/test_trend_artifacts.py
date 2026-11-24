"""Temporary synthetic fixtures verify plumbing; never research performance claims."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from forecast_validation import validated_forecast, validated_research
from trend_market_data import collect_candles, read_candles
from trend_pipeline import live_forecast, train_research


def market_fixture(days=600):
    count = (days + 7) * 24
    close_time = pd.date_range(end=pd.Timestamp.now(tz='UTC').floor('h') - pd.Timedelta(milliseconds=1), periods=count, freq='h')
    step = np.arange(count)
    close = 2 + .16 * np.sin(step / 32) + .02 * np.cos(step / 11)
    return pd.DataFrame(dict(open_time=close_time - pd.Timedelta(hours=1) + pd.Timedelta(milliseconds=1),
        close_time=close_time, open=close, high=close + .01, low=close - .01, close=close,
        volume=100 + 10 * np.sin(step / 17), quote_volume=(100 + 10 * np.sin(step / 17)) * close))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.raw = market_fixture()
        cls.report = train_research(cls.root, days=600, candles=cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_exact_saved_model_reproduces_metrics_and_live_outputs_obey_gate(self):
        evidence = validated_research(self.root / 'artifacts')
        self.assertTrue(evidence['integrity_passed'], evidence['limitation'])
        self.assertEqual(evidence['sample_size'], self.report['holdout']['sample_size'])
        forecast = live_forecast(self.raw.tail(888), self.root)
        self.assertIn(forecast['experimental_label'], ['UP', 'DOWN', 'NEUTRAL'])
        self.assertEqual(forecast['validated'], evidence['validated'])
        if evidence['validated']:
            self.assertAlmostEqual(sum(forecast['probabilities'].values()), 1.)
        else:
            self.assertIsNone(forecast['probabilities'])
        stale = self.raw.tail(888).copy()
        stale.close_time -= pd.Timedelta(days=1)
        stale.open_time -= pd.Timedelta(days=1)
        self.assertIsNone(live_forecast(stale, self.root)['experimental_label'])

    def test_modified_artifact_and_future_feature_code_fail_closed(self):
        from unittest.mock import patch
        with patch('trend_pipeline.feature_signature', return_value='changed'):
            self.assertFalse(validated_research(self.root / 'artifacts')['integrity_passed'])
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / 'artifacts'
            shutil.copytree(self.root / 'artifacts', copy)
            contract = json.loads((copy / 'trend_24h_validation.json').read_text())
            (copy / contract['oos_predictions_file']).write_text('modified')
            evidence = validated_research(copy)
            self.assertFalse(evidence['integrity_passed'])
            self.assertIn('modified', evidence['limitation'])

    def test_historical_replay_never_claims_predictions_generated_in_past(self):
        contract = json.loads((self.root / 'artifacts' / 'trend_24h_validation.json').read_text())
        frame = pd.read_csv(self.root / 'artifacts' / contract['oos_predictions_file'])
        self.assertNotIn('prediction_generated_at', frame)
        self.assertTrue(frame.evaluation_type.eq('retrospective_replay').all())
        computed = pd.to_datetime(frame.prediction_computed_at, utc=True)
        self.assertTrue((computed >= pd.to_datetime(frame.target_end, utc=True)).all())


class CollectorTests(unittest.TestCase):
    def test_incremental_resume_without_redownloading_existing_pages(self):
        raw = market_fixture(180)
        current = raw.close_time.max().ceil('h')
        def fetch(url, params, **kwargs):
            start = pd.to_datetime(params['startTime'], unit='ms', utc=True)
            end = pd.to_datetime(params['endTime'], unit='ms', utc=True)
            rows = raw[raw.open_time.between(start, end)].head(1000)
            values = [[int(row.open_time.timestamp() * 1000), row.open, row.high, row.low,
                row.close, row.volume, int(row.close_time.timestamp() * 1000), row.quote_volume, 0, 0, 0, 0]
                for row in rows.itertuples()]
            response = Mock(status_code=200)
            response.json.return_value = values
            return response
        client = Mock()
        client.get.side_effect = fetch
        with tempfile.TemporaryDirectory() as folder:
            first = collect_candles(folder, days=180, now=current, session=client, pause=0)
            self.assertEqual(len(first), len(raw))
            calls = client.get.call_count
            second = collect_candles(folder, days=180, now=current, session=client, pause=0)
            self.assertEqual(client.get.call_count, calls)
            self.assertEqual(len(second), len(first))

    def test_rate_limit_persists_cooldown_without_retrying(self):
        client = Mock()
        client.get.return_value = Mock(status_code=429, headers={'Retry-After': '120'})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 429'):
                collect_candles(folder, session=client, pause=0)
            self.assertEqual(client.get.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, 'cooldown'):
                collect_candles(folder, session=client, pause=0)
            self.assertEqual(client.get.call_count, 1)


if __name__ == '__main__':
    unittest.main()
