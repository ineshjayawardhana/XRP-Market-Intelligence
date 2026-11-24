"""Batch and chronological preparation tests; never fit a model or modify real news."""
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import ai_labeling as workflow
from batch_news_labeling import AUTO_FIELDS, REVIEW_FIELDS, batch, batch_review, classify
from news_label_safety import canonical_url, duplicate_groups, sha256, utc_date
from prepare_real_training import load_verified_partitions, prepare


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.fields = ['id', 'date', 'headline', 'source', 'url', 'timestamp_type', 'label']
        self.rows = [dict(id=str(i), date=f'2020-01-{i+1:02d}T12:00:00+00:00',
                         headline=title, source='fixture.example', url=f'https://fixture.example/{i}',
                         timestamp_type='FIRST_SEEN', label=label)
                     for i, (title, label) in enumerate([
                         ('Privy Adds Programmable Wallet Tools for XRP Ledger Developers', ''),
                         ('XRP Support and Resistance Levels Frame Market Trading', ''),
                         ('Cross River Bank partners with Ripple to integrate RLUSD', ''),
                         ('Will XRP reach four dollars by 2028?', ''),
                         ('Existing XRP neutral report', 'Neutral_Market_News'),
                         ('Existing XRP skipped report', 'SKIP')])]
        self.source = self.folder / 'xrp_real_labeled.csv'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        self.suggestions = [dict(id=r['id'], headline=r['headline'], url=r['url'],
                                suggested_label=label, confidence_estimate=score, reasoning='Fixture assessment',
                                needs_manual_review='True', review_status='pending', context_status='retrieved_unverified',
                                article_verified='False', suggestion_method='fixture')
                            for r, label, score in zip(self.rows, ['Real_Adoption', 'Neutral_Market_News',
                                                                 'Real_Adoption', workflow.REVIEW], [.9, .85, .8, .4])]
        workflow.atomic_csv(self.folder / 'xrp_ai_suggestions.csv', workflow.SUGGESTION_FIELDS, self.suggestions)
        self.proposals = [dict(row) for row in self.suggestions]
        (self.folder / 'xrp_ai_proposals.json').write_text(json.dumps(self.proposals))
        words = [
            'XRP Ledger developers can create wallets and sign transactions with the new programmable wallet tools. '
            'These application interfaces integrate payments and asset transfers through documented operational features. '
            'The product focuses on software builders rather than making any assertion about investment returns.',
            'XRP traded between support and resistance during a volatile market session. Volume indicators show measured '
            'price activity with moving averages providing conditional context for investors. Prices may rise or fall '
            'and this technical analysis does not promise any particular future investment return.',
            '',
            'XRP could reach a proposed valuation over several years but the available discussion is incomplete. '
            'Market supply and adoption assumptions introduce several uncertainties that cannot establish one clear '
            'category from this excerpt alone. A price forecast may overlap with balanced financial analysis.']
        self.contexts = [dict(id=r['id'], headline=r['headline'], url=r['url'],
                              status='retrieved_unverified' if text else 'unavailable', excerpt=text,
                              fetched_at='2026-11-15T00:00:00+00:00', article_verified=False)
                         for r, text in zip(self.rows, words)]
        (self.folder / 'xrp_article_context.json').write_text(json.dumps(self.contexts))

    def test_batch_all_records_provisional_and_inputs_byte_unchanged(self):
        inputs = {name: sha256(self.folder / name) for name in ('xrp_real_labeled.csv', 'xrp_ai_suggestions.csv')}
        summary = batch(self.folder)
        self.assertEqual(summary['total_records'], 4)
        self.assertEqual(summary['automatically_classified'], 2)
        self.assertEqual(summary['uncertain'], 2)
        self.assertEqual(summary['records_using_excerpts'], 3)
        for name, digest in inputs.items():
            self.assertEqual(sha256(self.folder / name), digest)
        rows = workflow.read_csv(self.folder / 'xrp_auto_labeled.csv')[1]
        self.assertTrue(all(row['label_status'] == 'AI_PROVISIONAL' for row in rows))
        self.assertTrue(all(row['article_verified'] == 'False' for row in rows))
        self.assertEqual(rows[2]['predicted_label'], workflow.REVIEW)
        self.assertNotIn('label', workflow.read_csv(self.folder / 'xrp_auto_labeled.csv')[0])
        self.assertFalse((self.folder / 'xrp_human_reviews.csv').exists())
        self.assertTrue((self.folder / 'xrp_batch_report.md').exists())
        self.assertTrue(list((self.folder / 'backups').glob('xrp_real_labeled_*.csv')))

    def test_rerun_reproducible_categories_and_protected_ledger(self):
        decision = workflow.decide(self.suggestions[1], 'a', 'Reviewer')
        ledger_path = self.folder / 'xrp_human_reviews.csv'
        workflow.atomic_csv(ledger_path, workflow.DECISION_FIELDS, [decision])
        digest = sha256(ledger_path)
        first = batch(self.folder)
        second = batch(self.folder)
        self.assertEqual(first['class_counts'], second['class_counts'])
        self.assertEqual(sha256(ledger_path), digest)

    def test_conflicting_human_label_flags_without_overwrite(self):
        self.rows[0]['label'] = 'Negative_FUD'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        before = self.source.read_bytes()
        batch(self.folder)
        auto = workflow.read_csv(self.folder / 'xrp_auto_labeled.csv')[1]
        self.assertEqual(auto[0]['predicted_label'], workflow.REVIEW)
        self.assertIn('conflict_with_protected_decision', auto[0]['review_flags'])
        self.assertEqual(self.source.read_bytes(), before)

    def test_conflict_between_saved_assessments_abstains(self):
        self.proposals[0]['suggested_label'] = 'Negative_FUD'
        (self.folder / 'xrp_ai_proposals.json').write_text(json.dumps(self.proposals))
        batch(self.folder)
        row = workflow.read_csv(self.folder / 'xrp_auto_labeled.csv')[1][0]
        self.assertEqual(row['predicted_label'], workflow.REVIEW)
        self.assertIn('saved_assessment_conflict', row['review_flags'])

    def test_new_articles_use_excerpt_rules_without_proposals(self):
        (self.folder / 'xrp_ai_proposals.json').unlink()
        (self.folder / 'xrp_ai_suggestions.csv').unlink()
        batch(self.folder)
        row = workflow.read_csv(self.folder / 'xrp_auto_labeled.csv')[1][0]
        self.assertEqual(row['predicted_label'], 'Real_Adoption')
        self.assertEqual(row['classification_method'], 'conservative_evidence_rules_v1')

    def test_new_article_cannot_claim_product_from_unsupported_excerpt(self):
        context = dict(self.contexts[0], excerpt=self.contexts[1]['excerpt'])
        result = classify(self.rows[0], None, context)
        self.assertEqual(result[0], workflow.REVIEW)

    def test_protected_skip_in_suggestions_remains_skipped(self):
        self.rows[0]['label'] = 'SKIP'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        before = self.source.read_bytes()
        batch(self.folder)
        row = workflow.read_csv(self.folder / 'xrp_auto_labeled.csv')[1][0]
        self.assertEqual(row['predicted_label'], workflow.REVIEW)
        self.assertEqual(row['protected_human_label'], 'SKIP')
        self.assertEqual(self.source.read_bytes(), before)

    def test_stale_context_and_low_confidence_abstain(self):
        self.contexts[0]['url'] = 'https://fixture.example/changed'
        result = classify(self.rows[0], self.proposals[0], self.contexts[0])
        self.assertEqual(result[0], workflow.REVIEW)
        self.proposals[1]['confidence_estimate'] = .6
        result = classify(self.rows[1], self.proposals[1], self.contexts[1])
        self.assertEqual(result[0], workflow.REVIEW)
        self.assertIn('low_heuristic_confidence', result[4])

    def test_batch_review_accept_correct_reject_together(self):
        batch(self.folder)
        choices = workflow.read_csv(self.folder / 'xrp_batch_decisions_template.csv')[1]
        choices[0]['action'] = 'accept'
        choices[1]['action'] = 'reject'
        choices[2].update(action='correct', human_label='Real_Adoption')
        path = self.folder / 'my_decisions.csv'
        workflow.atomic_csv(path, REVIEW_FIELDS, choices)
        before = self.source.read_bytes()
        self.assertEqual(batch_review(self.folder, path, 'Reviewer'), 3)
        ledger = workflow.load_decisions(self.folder / 'xrp_human_reviews.csv')
        self.assertEqual([row['review_status'] for row in ledger], ['accepted', 'rejected', 'corrected'])
        self.assertEqual(self.source.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'protected'):
            batch_review(self.folder, path, 'Reviewer')

    def test_invalid_batch_is_all_or_nothing(self):
        batch(self.folder)
        choices = workflow.read_csv(self.folder / 'xrp_batch_decisions_template.csv')[1]
        choices[0]['action'] = 'accept'
        choices[2]['action'] = 'accept'  # REVIEW_REQUIRED cannot become a human class.
        path = self.folder / 'bad_decisions.csv'
        workflow.atomic_csv(path, REVIEW_FIELDS, choices)
        with self.assertRaises(ValueError):
            batch_review(self.folder, path, 'Reviewer')
        self.assertFalse((self.folder / 'xrp_human_reviews.csv').exists())

    def test_sparse_preparation_includes_saved_human_decision(self):
        decision = workflow.decide(self.suggestions[1], 'a', 'Reviewer')
        workflow.atomic_csv(self.folder / 'xrp_human_reviews.csv', workflow.DECISION_FIELDS, [decision])
        before = self.source.read_bytes()
        batch(self.folder)
        manifest = prepare(self.folder)
        self.assertFalse(manifest['ready_for_training'])
        self.assertEqual(manifest['human_annotation_count'], 2)
        self.assertFalse(manifest['test_reservation_frozen'])
        self.assertEqual(self.source.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'Training blocked'):
            load_verified_partitions(self.folder, self.folder / 'training_prepared')

    def test_source_ai_status_is_excluded_even_if_label_column_present(self):
        fields = self.fields + ['label_status']
        rows = [dict(row, label_status='AI_PROVISIONAL' if row['id'] == '4' else '') for row in self.rows]
        workflow.atomic_csv(self.source, fields, rows)
        manifest = prepare(self.folder)
        self.assertEqual(manifest['human_annotation_count'], 0)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.fields = ['id', 'date', 'headline', 'source', 'url', 'label']
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.rows = [dict(id=f'h{i}', date=(start + timedelta(days=i * 2)).isoformat(),
                         headline=f'XRP topic{i} event{i} account{i}', source='fixture.example',
                         url=f'https://fixture.example/story/{i}', label=workflow.LABELS[i % 4])
                     for i in range(240)]
        self.source = self.folder / 'xrp_real_labeled.csv'
        workflow.atomic_csv(self.source, self.fields, self.rows)

    def test_sufficient_data_prepares_human_only_chronological_splits(self):
        manifest = prepare(self.folder)
        self.assertTrue(manifest['ready_for_training'], manifest['issues'])
        self.assertTrue(manifest['test_reservation_frozen'])
        output = self.folder / 'training_prepared'
        train, validation, test = load_verified_partitions(self.folder, output)
        self.assertLess(train.date.max(), validation.date.min() - timedelta(hours=24))
        self.assertLess(validation.date.max(), test.date.min() - timedelta(hours=24))
        for frame in (train, validation, test):
            self.assertTrue(frame.label_status.eq('HUMAN_VERIFIED').all())
        self.assertFalse(set(train.duplicate_group) & set(test.duplicate_group))

    def test_ai_training_is_opt_in_and_never_reaches_evaluation_or_future_train(self):
        ai_sources, auto = [], []
        for index, date in enumerate(['2020-02-01T00:00:00+00:00', '2022-01-01T00:00:00+00:00']):
            row = dict(id=f'ai{index}', date=date, headline=f'XRP different{index} payment{index}',
                       source='fixture.example', url=f'https://fixture.example/ai/{index}', label='')
            ai_sources.append(row)
            values = {key: '' for key in AUTO_FIELDS}
            values.update(id=row['id'], headline=row['headline'], url=row['url'], source=row['source'],
                          timestamp=date, predicted_label='Real_Adoption', label_status='AI_PROVISIONAL',
                          needs_manual_review='False')
            auto.append(values)
        workflow.atomic_csv(self.source, self.fields, self.rows + ai_sources)
        workflow.atomic_csv(self.folder / 'xrp_auto_labeled.csv', AUTO_FIELDS, auto)
        first = prepare(self.folder)
        self.assertEqual(first['partitions']['train_ai_provisional.csv']['count'], 0)
        second = prepare(self.folder, include_provisional=True)
        self.assertEqual(second['partitions']['train_ai_provisional.csv']['count'], 1)
        output = self.folder / 'training_prepared'
        weak = workflow.read_csv(output / 'train_ai_provisional.csv')[1]
        self.assertEqual(weak[0]['id'], 'ai0')
        frames = load_verified_partitions(self.folder, output)
        self.assertFalse(any(frame.id.str.startswith('ai').any() for frame in frames))

    def test_frozen_test_is_stable_and_future_records_are_excluded(self):
        prepare(self.folder)
        output = self.folder / 'training_prepared'
        frozen = (output / 'test_human.csv').read_bytes()
        extra = dict(id='future', date='2024-01-01T00:00:00+00:00',
                     headline='XRP future unrelated alpha beta', source='fixture.example',
                     url='https://fixture.example/future', label='Real_Adoption')
        workflow.atomic_csv(self.source, self.fields, self.rows + [extra])
        manifest = prepare(self.folder)
        self.assertTrue(manifest['ready_for_training'])
        self.assertEqual((output / 'test_human.csv').read_bytes(), frozen)
        for name in ('train_human.csv', 'validation_human.csv', 'test_human.csv'):
            self.assertNotIn('future', {row['id'] for row in workflow.read_csv(output / name)[1]})

    def test_changed_frozen_test_blocks_training(self):
        prepare(self.folder)
        self.rows[-1]['label'] = 'Real_Adoption'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        manifest = prepare(self.folder)
        self.assertFalse(manifest['ready_for_training'])
        self.assertTrue(any('Frozen test' in issue for issue in manifest['issues']))

    def test_duplicate_syndicated_titles_never_cross_partitions(self):
        duplicate = dict(self.rows[0], id='syndicated', date=self.rows[-1]['date'],
                         headline=self.rows[0]['headline'] + ' | The Motley Fool',
                         url='https://another.example/syndicated')
        workflow.atomic_csv(self.source, self.fields, self.rows + [duplicate])
        manifest = prepare(self.folder)
        self.assertEqual(manifest['unique_valid_human_count'], 240)
        self.assertTrue(manifest['ready_for_training'])
        audit = workflow.read_csv(self.folder / 'training_prepared' / 'excluded_records.csv')[1]
        self.assertTrue(any(row['id'] == 'syndicated' and 'duplicate_of' in row['reason'] for row in audit))

    def test_conflicting_human_duplicates_are_quarantined(self):
        duplicate = dict(self.rows[0], id='conflict', label='Negative_FUD')
        workflow.atomic_csv(self.source, self.fields, self.rows + [duplicate])
        manifest = prepare(self.folder)
        self.assertEqual(manifest['unique_valid_human_count'], 239)
        audit = workflow.read_csv(self.folder / 'training_prepared' / 'excluded_records.csv')[1]
        self.assertEqual({row['id'] for row in audit if row['reason'] == 'conflicting_duplicate_labels'}, {'h0', 'conflict'})

    def test_same_timestamp_cannot_cross_partition_boundaries(self):
        for row in self.rows:
            row['date'] = '2020-01-01T00:00:00+00:00'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        manifest = prepare(self.folder)
        self.assertFalse(manifest['ready_for_training'])
        self.assertEqual(manifest['partitions']['train_human.csv']['count'], 0)
        self.assertEqual(manifest['partitions']['validation_human.csv']['count'], 0)

    def test_modified_outputs_or_sources_invalidate_prepared_data(self):
        prepare(self.folder)
        output = self.folder / 'training_prepared'
        with (output / 'test_human.csv').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'partition changed'):
            load_verified_partitions(self.folder, output)
        prepare(self.folder)
        with self.source.open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            load_verified_partitions(self.folder, output)

    def test_new_ledger_after_preparation_invalidates_manifest(self):
        prepare(self.folder)
        workflow.atomic_csv(self.folder / 'xrp_human_reviews.csv', workflow.DECISION_FIELDS, [])
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            load_verified_partitions(self.folder, self.folder / 'training_prepared')

    def test_malformed_and_duplicate_csv_headers_are_refused(self):
        path = self.folder / 'broken.csv'
        path.write_text('id,label,label\n1,x,y\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate CSV headers'):
            workflow.read_csv(path)
        path.write_text('id,label\n1\n')
        with self.assertRaisesRegex(ValueError, 'Malformed CSV'):
            workflow.read_csv(path)

    def test_trainer_refuses_before_fit_on_sparse_data(self):
        import train_real_model
        workflow.atomic_csv(self.source, self.fields, self.rows[:8])
        with patch.object(train_real_model, 'DATA', self.folder), patch.object(train_real_model.Pipeline, 'fit') as fit:
            with self.assertRaisesRegex(ValueError, 'Training blocked'):
                train_real_model.fit_and_evaluate()
        fit.assert_not_called()

    def test_url_canonicalization_and_aware_timestamps(self):
        self.assertEqual(canonical_url('http://www.example.com/a/?utm_source=x&ref=y'),
                         canonical_url('https://example.com/a'))
        self.assertNotEqual(canonical_url('https://example.com/article?id=1'),
                            canonical_url('https://example.com/article?id=2'))
        self.assertIsNone(utc_date('2020-01-01'))
        self.assertIsNotNone(utc_date('2020-01-01T00:00:00Z'))


if __name__ == '__main__':
    unittest.main()
