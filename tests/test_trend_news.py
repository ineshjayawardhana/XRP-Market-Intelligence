"""Point-in-time news tests use temporary fixtures, never research datasets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ai_labeling import atomic_csv
from news_label_safety import sha256
from trend_news import audit_news, news_features_asof


class TrendNewsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'data' / 'multisource'
        self.research = self.folder / 'research'
        self.research.mkdir(parents=True)

    def write(self, source, silver=(), gold=()):
        for path, rows in [(self.folder / 'xrp_real_labeled.csv', source),
                           (self.research / 'silver_dataset.csv', silver),
                           (self.research / 'gold_dataset.csv', gold)]:
            if rows:
                fields = list(dict.fromkeys(key for row in rows for key in row))
                atomic_csv(path, fields, list(rows))

    def row(self, ident='a', at='2026-01-01T00:00:00Z'):
        return dict(id=ident, headline='XRP fixture headline ' + ident,
                    url='https://fixture.example/' + ident, source='fixture', label='',
                    collected_at=at, availability_at=at)

    def silver(self, row, **changes):
        return dict(row, label='Neutral_Market_News', label_status='AI_PROVISIONAL',
                    classified_at=row['collected_at'], **changes)

    def test_missing_news_is_ineligible_not_fake_zero_coverage_success(self):
        report = audit_news(self.root)
        self.assertFalse(report['ready'])
        self.assertEqual(report['eligible_records'], 0)
        json.dumps(report)

    def test_publication_and_first_seen_cannot_replace_label_timestamp(self):
        source = self.row()
        silver = self.silver(source)
        silver.pop('classified_at')
        silver.update(published_at='2025-01-01T00:00:00Z', gdelt_first_seen_at='2025-01-01T00:00:00Z')
        self.write([source], [silver])
        report = audit_news(self.root)
        self.assertEqual(report['eligible_records'], 0)
        self.assertEqual(report['exclusion_reasons']['missing_explicit_label_availability'], 1)

    def test_later_context_and_label_time_delay_feature_availability(self):
        source = self.row()
        silver = self.silver(source)
        silver['context_retrieved_at'] = '2026-01-05T00:00:00Z'
        silver['classified_at'] = '2026-01-04T00:00:00Z'
        self.write([source], [silver])
        report = audit_news(self.root)
        self.assertEqual(report['feature_start'], '2026-01-05T00:00:00+00:00')
        self.assertFalse(report['ready'])

    def test_legacy_human_annotation_without_review_time_is_not_historically_available(self):
        source = self.row()
        source['label'] = 'Neutral_Market_News'
        gold = dict(source, label_status='HUMAN_VERIFIED', context_retrieved_at='2026-01-01T00:00:00Z')
        self.write([source], gold=[gold])
        self.assertEqual(audit_news(self.root)['eligible_records'], 0)

    def test_gold_cannot_forge_human_status_or_override_skip(self):
        source = self.row()
        gold = dict(source, label='Real_Adoption', label_status='HUMAN_VERIFIED', human_reviewed_at=source['collected_at'])
        self.write([source], gold=[gold])
        self.assertEqual(audit_news(self.root)['eligible_records'], 0)
        source['label'] = 'SKIP'
        self.write([source], gold=[gold])
        self.assertEqual(audit_news(self.root)['exclusion_reasons']['protected_human_exclusion'], 1)

    def test_naive_and_future_label_timestamps_are_refused(self):
        source = self.row()
        silver = self.silver(source)
        silver['classified_at'] = '2026-01-01T00:00:00'
        self.write([source], [silver])
        self.assertEqual(audit_news(self.root)['eligible_records'], 0)
        silver['classified_at'] = '2099-01-01T00:00:00Z'
        self.write([source], [silver])
        self.assertEqual(audit_news(self.root)['exclusion_reasons']['future_feature_availability'], 1)

    def test_duplicate_headlines_do_not_create_independent_events(self):
        first, second = self.row('a'), self.row('b')
        second['headline'] = first['headline']
        self.write([first, second], [self.silver(first), self.silver(second)])
        report = audit_news(self.root)
        self.assertEqual(report['eligible_records'], 1)
        self.assertEqual(report['exclusion_reasons']['duplicate_article'], 1)

    def test_coverage_minima_and_causal_separate_gold_silver_features(self):
        # Every other day for 122 days; enough independently timed news days.
        times = pd.date_range('2025-01-01T00:00:00Z', periods=62, freq='2D')
        sources = [self.row(str(i), stamp.isoformat()) for i, stamp in enumerate(times)]
        silver = [self.silver(row) for row in sources]
        self.write(sources, silver)
        predictions = pd.date_range(times[0] - pd.Timedelta(hours=1), times[-1] + pd.Timedelta(hours=25), freq='h')
        report = audit_news(self.root, predictions)
        self.assertTrue(report['ready'], report['reasons'])
        before = sha256(self.research / 'silver_dataset.csv')
        features = news_features_asof(self.root, predictions)
        self.assertEqual(features.iloc[0]['silver_Neutral_Market_News_24h'], 0)
        self.assertEqual(features.iloc[1]['silver_Neutral_Market_News_24h'], 1)
        self.assertEqual(features.iloc[25]['silver_Neutral_Market_News_24h'], 0)
        self.assertEqual(features.filter(like='gold_').to_numpy().sum(), 0)
        self.assertEqual(sha256(self.research / 'silver_dataset.csv'), before)

    def test_unready_builder_and_naive_prediction_time_raise(self):
        with self.assertRaisesRegex(ValueError, 'not ready'):
            news_features_asof(self.root, ['2026-01-01T00:00:00Z'])
        with self.assertRaisesRegex(ValueError, 'timezone'):
            audit_news(self.root, ['2026-01-01T00:00:00'])


if __name__ == '__main__':
    unittest.main()
