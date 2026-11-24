"""Safety and workflow tests. All writes use disposable temporary datasets."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_labeling as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.source = self.folder / 'xrp_real_labeled.csv'
        self.suggestions = self.folder / 'xrp_ai_suggestions.csv'
        self.decisions = self.folder / 'xrp_human_reviews.csv'
        self.proposals = self.folder / 'proposals.json'
        self.fields = ['id', 'date', 'headline', 'source', 'url', 'label']
        self.rows = [dict(id=str(i), date=f'2026-01-{i+1:02d}',
                          headline=f'XRP example article number {i}', source='example.org',
                          url=f'https://example.org/{i}', label=label)
                     for i, label in enumerate(['', '', 'Neutral_Market_News', 'SKIP'])]
        workflow.atomic_csv(self.source, self.fields, self.rows)
        self.original = self.source.read_bytes()
        self.proposals.write_text(json.dumps([
            dict(self.rows[0], suggested_label='Real_Adoption', confidence_estimate=.8,
                 reasoning='Named deployment. Article not verified.')]), encoding='utf-8')

    def generate(self):
        workflow.generate(self.source, self.suggestions, self.proposals, self.decisions)
        return workflow.read_csv(self.suggestions)[1]

    def save_decision(self, row, action='a'):
        decision = workflow.decide(row, action, 'Test reviewer')
        workflow.atomic_csv(self.decisions, workflow.DECISION_FIELDS, [decision])
        return decision

    def test_backup_is_byte_identical_and_unique(self):
        first, second = workflow.backup(self.source), workflow.backup(self.source)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), self.original)
        self.assertEqual(hashlib.sha256(first.read_bytes()).digest(),
                         hashlib.sha256(self.original).digest())

    def test_generate_only_blanks_no_dataset_mutation(self):
        suggestions = self.generate()
        self.assertEqual([row['id'] for row in suggestions], ['0', '1'])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertTrue(all(row['needs_manual_review'] == 'True' for row in suggestions))
        self.assertTrue(all(row['article_verified'] == 'False' for row in suggestions))
        self.assertNotIn('label', workflow.read_csv(self.suggestions)[0])

    def test_unknown_or_stale_proposals_require_review(self):
        proposals = json.loads(self.proposals.read_text())
        proposals[0]['headline'] = 'Changed headline'
        self.proposals.write_text(json.dumps(proposals))
        self.assertTrue(all(row['suggested_label'] == workflow.REVIEW for row in self.generate()))

    def test_review_required_cannot_be_accepted(self):
        row = self.generate()[1]
        with self.assertRaises(ValueError):
            workflow.decide(row, 'a', 'Reviewer')
        self.assertEqual(workflow.decide(row, '2', 'Reviewer')['human_label'], 'Hype_Speculation')

    def test_accept_correct_reject_and_invalid_input(self):
        row = self.generate()[0]
        self.assertEqual(workflow.decide(row, 'a', 'Reviewer')['review_status'], 'accepted')
        self.assertEqual(workflow.decide(row, '3', 'Reviewer')['human_label'], 'Negative_FUD')
        self.assertEqual(workflow.decide(row, 'r', 'Reviewer')['human_label'], '')
        for action, reviewer in [('bad', 'Reviewer'), ('a', '')]:
            with self.assertRaises(ValueError):
                workflow.decide(row, action, reviewer)

    def test_review_saves_each_answer_and_resumes(self):
        self.generate()
        with patch('builtins.input', side_effect=['bad', 'a', 'q']):
            workflow.review(self.source, self.suggestions, self.decisions, 'Reviewer')
        saved = workflow.load_decisions(self.decisions)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['id'], '0')
        self.assertEqual(workflow.read_csv(self.suggestions)[1][0]['needs_manual_review'], 'False')
        with patch('builtins.input', side_effect=['4']):
            workflow.review(self.source, self.suggestions, self.decisions, 'Reviewer')
        self.assertEqual(len(workflow.load_decisions(self.decisions)), 2)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_defer_and_quit_do_not_make_human_labels(self):
        self.generate()
        with patch('builtins.input', side_effect=['d', 'q']):
            workflow.review(self.source, self.suggestions, self.decisions, 'Reviewer')
        self.assertFalse(self.decisions.exists())

    def test_export_only_human_decisions_is_idempotent(self):
        row = self.generate()[0]
        self.save_decision(row)
        self.assertEqual(workflow.export_reviewed(self.source, self.decisions), 1)
        exported = workflow.read_csv(self.source)[1]
        self.assertEqual([r['label'] for r in exported],
                         ['Real_Adoption', '', 'Neutral_Market_News', 'SKIP'])
        self.assertEqual(exported[0]['label_origin'], 'human_reviewed_ai')
        for old, new in zip(self.rows, exported):
            for column in self.fields:
                if old['id'] != '0' or column != 'label':
                    self.assertEqual(old[column], new[column])
        after = self.source.read_bytes()
        self.assertEqual(workflow.export_reviewed(self.source, self.decisions), 0)
        self.assertEqual(self.source.read_bytes(), after)
        backups = list((self.folder / 'backups').glob('*.csv'))
        self.assertTrue(any(path.read_bytes() == self.original for path in backups))

    def test_new_human_label_is_never_overwritten(self):
        self.save_decision(self.generate()[0])
        self.rows[0]['label'] = 'Negative_FUD'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        self.assertEqual(workflow.export_reviewed(self.source, self.decisions), 0)
        self.assertEqual(workflow.read_csv(self.source)[1][0]['label'], 'Negative_FUD')

    def test_rejected_and_stale_decisions_never_export(self):
        row = self.generate()[0]
        self.save_decision(row, 'r')
        self.assertEqual(workflow.export_reviewed(self.source, self.decisions), 0)
        self.save_decision(row)
        self.rows[0]['url'] = 'https://example.org/changed'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        self.assertEqual(workflow.export_reviewed(self.source, self.decisions), 0)

    def test_rerun_preserves_reviews_and_historical_suggestions(self):
        first = self.generate()
        self.save_decision(first[0])
        workflow.export_reviewed(self.source, self.decisions)
        second = self.generate()
        self.assertEqual(len(second), 2)
        self.assertEqual(second[0]['review_status'], 'accepted')
        self.assertEqual(second[0]['reasoning'], first[0]['reasoning'])

    def test_changed_article_cannot_replace_a_human_decision(self):
        first = self.generate()
        self.save_decision(first[0])
        ledger = self.decisions.read_bytes()
        self.rows[0]['headline'] = 'XRP article edited after human review'
        workflow.atomic_csv(self.source, self.fields, self.rows)
        second = self.generate()
        self.assertEqual(second[0]['headline'], first[0]['headline'])
        with patch('builtins.input', side_effect=['q']):
            workflow.review(self.source, self.suggestions, self.decisions, 'Reviewer')
        self.assertEqual(self.decisions.read_bytes(), ledger)
        self.assertEqual(workflow.export_reviewed(self.source, self.decisions), 0)

    def test_bad_confidence_and_duplicate_ids_fail(self):
        proposals = json.loads(self.proposals.read_text())
        proposals[0]['confidence_estimate'] = 1.4
        self.proposals.write_text(json.dumps(proposals))
        with self.assertRaises(ValueError):
            self.generate()
        workflow.atomic_csv(self.source, self.fields, self.rows + [self.rows[0]])
        with self.assertRaises(ValueError):
            workflow.dataset(self.source)

    def test_exclusive_lock(self):
        with workflow.workflow_lock(self.folder):
            with self.assertRaises(RuntimeError):
                with workflow.workflow_lock(self.folder):
                    self.fail('Second process acquired lock')
        self.assertFalse((self.folder / '.ai_labeling.lock').exists())

    def test_inaccessible_article_has_no_verified_claim(self):
        import requests
        with patch('requests.get', side_effect=requests.Timeout('timeout')):
            result = workflow.fetch_one(self.rows[0])
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse(result['article_verified'])
        self.assertEqual(result['excerpt'], '')

    def test_training_filters_ai_and_requires_human_review_evidence(self):
        import pandas as pd
        from train_real_model import human_labeled_rows
        frame = pd.DataFrame([
            dict(label='Real_Adoption', label_origin=origin,
                 human_reviewed_by=reviewer, human_reviewed_at=timestamp)
            for origin, reviewer, timestamp in [('', '', ''), ('ai_suggested', '', ''),
                ('unknown', '', ''), ('human_reviewed_ai', '', ''),
                ('human_reviewed_ai', 'Reviewer', '2026-11-15T00:00:00+00:00')]])
        self.assertEqual(human_labeled_rows(frame).index.tolist(), [0, 4])
        self.assertEqual(len(human_labeled_rows(frame[['label']])), 5)

    def test_current_dataset_cannot_produce_test_accuracy(self):
        import train_real_model
        with patch.object(train_real_model, 'SOURCE', self.source):
            with self.assertRaisesRegex(ValueError, 'Need >=120'):
                train_real_model.load_labeled()


if __name__ == '__main__':
    unittest.main()
