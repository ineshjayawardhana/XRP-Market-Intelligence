"""Gold/Silver, rubric, sampling and Streamlit tests use temporary fake fixtures."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_labeling import DECISION_FIELDS, LABELS, REVIEW, atomic_csv, decide, read_csv
from collect_multisource_news import BASE_FIELDS
from news_label_safety import sha256
from research_context import fetch_additional, parse_public_article
from research_labels import assess, classify_research, clean_excerpt
from research_workflow import FIELDS, build, save_reviews, verification_sample


class RubricTests(unittest.TestCase):
    def test_placeholder_is_not_article_context(self):
        self.assertEqual(clean_excerpt('XRP developer news', 'Read the latest update on XRP developer news.'), '')
        self.assertEqual(clean_excerpt('XRP developer news', 'XRP developer news what the latest source material shows and why it matters for crypto'), '')

    def test_live_product_requires_corroboration(self):
        title = 'Aviva launches tokenized shares on XRP Ledger'
        self.assertEqual(assess(title, 'The fund tokenized shares are available on the ledger.')[0], 'Real_Adoption')
        self.assertEqual(assess(title, '')[0], REVIEW)

    def test_planned_new_chain_is_not_proven_by_old_launch(self):
        title = 'Ripple Launching RLUSD Stablecoin on Layer-2s'
        text = 'Ripple is preparing to launch the stablecoin and plans to begin testing. RLUSD launched last year on another chain.'
        self.assertEqual(assess(title, text)[0], REVIEW)

    def test_webpage_is_not_a_live_financial_product(self):
        self.assertEqual(assess('Ripple Gears Up for RLUSD Rollout With Launch of New Webpage',
                               'Ripple launched a webpage explaining the future product.')[0], REVIEW)

    def test_market_loss_and_liquidity_are_not_security_incidents_or_forecasts(self):
        self.assertEqual(assess('XRP Price Falls Below Support', 'XRP price fell as market support weakened.')[0], 'Neutral_Market_News')
        self.assertEqual(assess('Buy Side Explodes: XRP Liquidity 7x Heavier Than Sells', 'Observed XRP liquidity is concentrated on the buy side.')[0], 'Neutral_Market_News')
        self.assertEqual(assess('Demand for XRP Rebounds to Pre-Lawsuit Levels', 'This is a demand measurement.')[0], 'Neutral_Market_News')

    def test_forecast_and_reported_hack(self):
        self.assertEqual(assess('Analyst Predicts XRP Rally to $20', 'The analyst gives a future XRP price target.')[0], 'Hype_Speculation')
        self.assertEqual(assess('Investor Loses XRP in Hack', 'The report says hackers stole the tokens.')[0], 'Negative_FUD')
        self.assertEqual(assess('Analysts Debunk XRP Hack Rumor', 'No hack occurred according to this report.')[0], REVIEW)

    def test_testnet_reset_and_proposal_not_production_loss_or_deployment(self):
        self.assertEqual(assess('XRPL Testnet and Devnet Reset Wiping Data', 'This is a scheduled testnet reset.')[0], 'Neutral_Market_News')
        self.assertEqual(assess('XRP Ledger Lending Proposal Opens New Design', 'This proposal requires validator approval.')[0], 'Neutral_Market_News')

    def test_conflicting_or_stale_ai_cannot_silently_win(self):
        row = dict(id='a', headline='Analyst Predicts XRP Rally to $20', url='https://fixture.example/a')
        context = dict(row, status='retrieved_unverified', excerpt='An analyst discusses a future XRP price target and potential returns over the next several years.')
        previous = dict(row, predicted_label='Neutral_Market_News', heuristic_confidence='.8', reasoning='Earlier test assessment')
        self.assertEqual(classify_research(row, context, previous)[0], REVIEW)
        previous['headline'] = 'Changed article'
        self.assertEqual(classify_research(row, context, previous)[0], 'Hype_Speculation')

    def test_public_article_requires_matching_heading_and_visible_body(self):
        class Response:
            encoding = 'utf-8'
            headers = {'Content-Type': 'text/html'}
            content = b'''<main><h1>XRP wallet product launch</h1><article><p>
                A new XRP wallet product is now available to application developers. The announcement
                describes how users can create accounts and sign payment transactions with software tools,
                while outlining limitations of the currently available service for its initial rollout.
                </p></article></main>'''
        row = dict(headline='XRP wallet product launch')
        self.assertIn('XRP wallet', parse_public_article(row, Response()))
        with self.assertRaises(ValueError):
            parse_public_article(dict(headline='Completely unrelated banking news'), Response())
        response = Response()
        response.content = response.content.replace(b'<main>', b'<div class="entry-content">').replace(b'</main>', b'</div>')
        self.assertIn('XRP wallet', parse_public_article(row, response))
        response.content += b'<script>{"isAccessibleForFree":false}</script>'
        with self.assertRaisesRegex(ValueError, 'restricted'):
            parse_public_article(row, response)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        titles = ['Existing XRP verified market report', 'Previously skipped XRP item',
                  'Aviva launches tokenized shares on XRP Ledger', 'Analyst Predicts XRP Rally to $20',
                  'XRP ETF Inflows Rise During Trading', 'Investor Loses XRP in Hack',
                  'Existing XRP ledger human decision', 'Ripple Plans to Launch RLUSD Payment Product',
                  'Analyst Predicts XRP Rally to $20 | Publisher', 'Rejected XRP commentary']
        self.rows = [dict(id=str(i), headline=title, source='fixture.example', url=f'https://fixture.example/story/{i}',
                         date='2026-11-15T08:00:00+00:00' if i > 1 else '2020-01-01T12:00:00+00:00',
                         timestamp_type='PROJECT_OBSERVED_AT', label='Neutral_Market_News' if i == 0 else 'SKIP' if i == 1 else '',
                         published_at=f'2020-01-{i+1:02d}T12:00:00+00:00', availability_at='') for i, title in enumerate(titles)]
        self.source = self.folder / 'xrp_real_labeled.csv'
        atomic_csv(self.source, BASE_FIELDS + ['published_at', 'availability_at'], self.rows)
        self.source_hash = sha256(self.source)
        body = ['A human annotation exists.', '', 'The fund tokenized shares are now available on the XRP Ledger.',
                'The analyst discusses a future price target for XRP and says it may rally over the next several years.',
                'XRP ETF inflows and trading volume are measured in this report.', 'The report says hackers stole the XRP tokens.',
                'This record already has a human decision.', 'Ripple plans to launch the RLUSD service, which is not yet available.',
                'The analyst discusses a future price target for XRP and says it may rally over the next several years.', 'Rejected item.']
        contexts = [dict(id=row['id'], headline=row['headline'], url=row['url'], status='retrieved_unverified',
                         excerpt=text, fetched_at='2026-11-15T09:00:00+00:00', context_kind='PUBLISHER_FEED_SUMMARY')
                    for row, text in zip(self.rows, body)]
        (self.folder / 'xrp_article_context.json').write_text(json.dumps(contexts))
        decisions = [decide(dict(self.rows[6], suggested_label='Hype_Speculation'), 'a', 'Original Reviewer'),
                     decide(dict(self.rows[9], suggested_label='Neutral_Market_News'), 'r', 'Original Reviewer')]
        self.ledger = self.folder / 'xrp_human_reviews.csv'
        atomic_csv(self.ledger, DECISION_FIELDS, decisions)
        self.original_ledger = self.ledger.read_bytes()
        self.report = build(self.folder)

    def all_rows(self):
        return read_csv(self.folder / 'research' / 'all_classifications.csv')[1]

    def choice(self, identifier, action='accept', label=''):
        row = next(row for row in self.all_rows() if row['id'] == identifier)
        return dict(id=identifier, action=action, human_label=label, prediction_fingerprint=row['prediction_fingerprint'])

    def test_all_records_processed_and_source_annotations_unchanged(self):
        self.assertEqual(len(self.all_rows()), 10)
        self.assertEqual(sha256(self.source), self.source_hash)
        self.assertEqual(self.ledger.read_bytes(), self.original_ledger)
        self.assertEqual(self.report['gold_records'], 2)
        self.assertEqual(self.report['human_skip_count'], 1)
        self.assertEqual(self.report['human_rejection_count'], 1)

    def test_silver_and_gold_are_disjoint_and_deduplicated(self):
        output = self.folder / 'research'
        silver, gold = read_csv(output / 'silver_dataset.csv')[1], read_csv(output / 'gold_dataset.csv')[1]
        self.assertFalse({row['id'] for row in silver} & {row['id'] for row in gold})
        self.assertTrue(all(row['label_status'] == 'AI_PROVISIONAL' for row in silver))
        self.assertTrue(all(row['label_status'] == 'HUMAN_VERIFIED' for row in gold))
        self.assertEqual(len([row for row in silver if row['label'] == 'Hype_Speculation']), 1)
        self.assertNotIn(REVIEW, {row['label'] for row in silver + gold})

    def test_explicit_multi_record_review_promotes_only_selected_humans(self):
        self.assertEqual(save_reviews(self.folder, [self.choice('2'), self.choice('4')], 'Reviewer'), 2)
        build(self.folder)
        gold = read_csv(self.folder / 'research' / 'gold_dataset.csv')[1]
        self.assertEqual({row['id'] for row in gold}, {'0', '2', '4', '6'})
        self.assertEqual(sha256(self.source), self.source_hash)
        self.assertTrue(any(row['id'] == '2' and row['human_reviewed_by'] == 'Reviewer' for row in gold))

    def test_review_reject_defer_and_correct(self):
        choices = [self.choice('2', 'reject'), self.choice('3', 'defer'), self.choice('7', 'correct', 'Neutral_Market_News')]
        self.assertEqual(save_reviews(self.folder, choices, 'Reviewer'), 2)
        report = build(self.folder)
        self.assertEqual(report['gold_records'], 3)
        self.assertEqual(report['human_rejection_count'], 2)

    def test_stale_duplicate_or_invalid_batch_never_partially_saves(self):
        for choices in ([self.choice('2'), self.choice('7')], [self.choice('2'), self.choice('2')],
                        [dict(self.choice('2'), prediction_fingerprint='stale')]):
            with self.assertRaises(ValueError):
                save_reviews(self.folder, choices, 'Reviewer')
            self.assertEqual(self.ledger.read_bytes(), self.original_ledger)

    def test_changed_source_timestamp_invalidates_review_snapshot(self):
        choice = self.choice('2')
        fields, rows = read_csv(self.source)
        rows[2]['date'] = '2026-11-15T09:00:00+00:00'
        atomic_csv(self.source, fields, rows)
        with self.assertRaises(ValueError):
            save_reviews(self.folder, [choice], 'Reviewer')
        self.assertEqual(self.ledger.read_bytes(), self.original_ledger)

    def test_context_failures_cache_and_only_extraction_failures_retry(self):
        output = self.folder / 'research'
        cache = output / 'additional_context.json'
        row = self.rows[2]
        cache.write_text(json.dumps([dict(id=row['id'], headline=row['headline'], url=row['url'],
            status='unavailable', excerpt='', detail='No substantive matching public article text')]))
        with patch('research_context.PublicClient.fetch', side_effect=RuntimeError('HTTP 403 blocked')) as fetch:
            result = fetch_additional(self.folder, [row], {}, limit=1)
            self.assertEqual(fetch.call_count, 0)
            self.assertEqual(result['retrieved'], 0)
            fetch_additional(self.folder, [row], {}, limit=1, retry_unavailable=True)
            self.assertEqual(fetch.call_count, 1)
            fetch_additional(self.folder, [row], {}, limit=1, retry_unavailable=True)
            self.assertEqual(fetch.call_count, 1)
        self.assertTrue(list((output / 'backups').glob('additional_context_*.json')))

    def test_protected_human_skip_and_rejection_cannot_be_overwritten(self):
        for identifier in ('0', '1', '6', '9'):
            with self.assertRaises(ValueError):
                save_reviews(self.folder, [self.choice(identifier, 'correct', 'Real_Adoption')], 'Reviewer')
        self.assertEqual(self.ledger.read_bytes(), self.original_ledger)

    def test_no_name_no_approval_and_no_ground_truth_from_blank_actions(self):
        with self.assertRaises(ValueError):
            save_reviews(self.folder, [self.choice('2')], '')
        self.assertEqual(save_reviews(self.folder, [self.choice('2', '')], 'Reviewer'), 0)
        self.assertEqual(self.ledger.read_bytes(), self.original_ledger)

    def test_first_available_and_publication_sampling_are_separate(self):
        row = next(row for row in self.all_rows() if row['id'] == '2')
        self.assertEqual(row['date'], self.rows[2]['date'])
        self.assertEqual(row['availability_at'], self.rows[2]['date'])
        self.assertEqual(row['sampling_time'], self.rows[2]['published_at'])
        self.assertNotEqual(row['date'], row['sampling_time'])

    def test_gold_only_evaluation_and_readiness_blocks_both_models(self):
        self.assertFalse(self.report['tfidf']['can_begin_supervised_training'])
        self.assertFalse(self.report['distilbert']['can_begin_supervised_training'])
        for name in ('train_human.csv', 'validation_human.csv', 'test_human.csv'):
            rows = read_csv(self.folder / 'research' / 'training_prepared' / name)[1]
            self.assertTrue(all(row['label_status'] == 'HUMAN_VERIFIED' for row in rows))
            self.assertTrue({row['id'] for row in rows}.issubset({'0', '6'}))

    def test_sample_reports_shortages_without_human_promotion(self):
        plan = self.report['verification_plan']
        self.assertGreater(plan['Real_Adoption']['remaining_if_all_confirmed'], 0)
        sample = read_csv(self.folder / 'research' / 'verification_sample.csv')[1]
        self.assertTrue(all(row['label_status'] == 'AI_PROVISIONAL' for row in sample))
        self.assertEqual(self.report['gold_records'], 2)

    def test_streamlit_load_and_unselected_save_are_read_only(self):
        from streamlit.testing.v1 import AppTest
        app_path = Path(__file__).resolve().parents[1] / 'review_research_app.py'
        with patch.dict(os.environ, {'XRP_RESEARCH_DATA_DIR': str(self.folder)}):
            app = AppTest.from_file(str(app_path)).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.button(key='save_reviews').click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
        self.assertEqual(self.ledger.read_bytes(), self.original_ledger)

    def test_streamlit_explicit_batch_correction_saves_multiple_records(self):
        from streamlit.testing.v1 import AppTest
        app_path = Path(__file__).resolve().parents[1] / 'review_research_app.py'
        with patch.dict(os.environ, {'XRP_RESEARCH_DATA_DIR': str(self.folder)}):
            app = AppTest.from_file(str(app_path)).run(timeout=30)
            app.checkbox[0].check()
            app.text_input(key='reviewer_name').input('UI Reviewer')
            next(widget for widget in app.selectbox if widget.label == 'Default decision for selected rows').select('correct')
            next(widget for widget in app.selectbox if widget.label == 'Default corrected category').select('Neutral_Market_News')
            app.button(key='save_reviews').click()
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            next(widget for widget in app.selectbox if widget.label == 'Review set').select('Gold (read-only)').run(timeout=30)
            self.assertEqual(len(app.exception), 0)
        gold = read_csv(self.folder / 'research' / 'gold_dataset.csv')[1]
        self.assertGreater(len(gold), 3)
        self.assertEqual(sha256(self.source), self.source_hash)
        self.assertEqual(next(row for row in gold if row['id'] == '6')['label'], 'Hype_Speculation')


class SamplingTests(unittest.TestCase):
    def test_quartiles_sources_and_confidence_are_represented(self):
        rows = []
        for label in LABELS:
            for index in range(40):
                row = {key: '' for key in FIELDS}
                row.update(id=f'{label}_{index}', label_status='AI_PROVISIONAL', canonical_record='True',
                           eligible_content='True', duplicate_group=f'{label}_{index}', predicted_label=label,
                           sampling_time=f'2020-{index // 28 + 1:02d}-{index % 28 + 1:02d}T00:00:00+00:00',
                           publisher_group=f'publisher{index % 3}', confidence_band='high' if index % 2 else 'medium')
                rows.append(row)
        sample, plan = verification_sample(rows, [], target=25)
        self.assertEqual(len(sample), 100)
        for label in LABELS:
            subset = [row for row in sample if row['predicted_label'] == label]
            self.assertEqual(len(subset), 25)
            self.assertEqual(len({row['publication_time_bin'] for row in subset}), 4)
            self.assertEqual(len({row['publisher_group'] for row in subset}), 3)
            self.assertEqual(len({row['confidence_band'] for row in subset}), 2)
            self.assertEqual(plan[label]['remaining_if_all_confirmed'], 0)


if __name__ == '__main__':
    unittest.main()
