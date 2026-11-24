"""Readable scientific comparison artifacts; no model or dashboard mutation."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from news_label_safety import atomic_json
from trend_metrics import OUTCOMES, QUALITY_MESSAGES

METRICS = ['sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece']


def table(rows, columns):
    result = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        result.append('| ' + ' | '.join(f'{row.get(key):.4f}' if isinstance(row.get(key), float) else str(row.get(key, '')) for key in columns) + ' |')
    return '\n'.join(result)


def metric_row(name, values):
    return dict(model=name, **{key: values.get(key) for key in METRICS})


def write_report(report, root):
    folder = Path(root) / 'reports' / 'trend_investigation' / report['protocol_sha256'][:16]
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / 'comparison.json', report)
    old = report['old_model_diagnosis']
    old_rows = [metric_row(name, value) for name, value in old['recorded_candidate_holdout'].items()]
    old_rows += [metric_row(name, value) for name, value in old['recorded_baselines'].items()]
    outer_rows = [metric_row(name, candidate['metrics']) for name, candidate in report['candidates'].items()]
    pd.DataFrame(outer_rows).to_csv(folder / 'nested_comparison.csv', index=False)
    pd.DataFrame(old_rows).to_csv(folder / 'old_test_diagnosis.csv', index=False)
    class_rows = []
    for stage, models in [('old_exposed_test', old['recorded_candidate_holdout']),
                           ('outer_nested_validation', {name: value['metrics'] for name, value in report['candidates'].items()})]:
        for name, values in models.items():
            for label in OUTCOMES:
                class_rows.append(dict(stage=stage, model=name, outcome=label, **values['classification_report'][label]))
    lines = ['# XRP trend model investigation', '',
        'Production dashboard, model pointers, news labels and trained production models are unchanged. All new candidates are research artifacts.', '',
        '## Diagnosis of the already-examined 2026 test', '',
        'These historical test results motivated hypotheses; they are not an untouched evaluation for the new experiments.', '',
        table(old_rows, ['model'] + METRICS), '',
        '### Existing partition distributions', '', '```json',
        __import__('json').dumps(old['partition_class_distributions'], indent=2), '```', '',
        '### Train, calibration and test behavior', '']
    gap_rows = []
    for name, model in old['model_partition_metrics'].items():
        for partition in ('train', 'calibration', 'test'):
            gap_rows.append(dict(model=name, partition=partition, **{key: model[partition][key] for key in ['sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1']}))
    lines += [table(gap_rows, ['model', 'partition', 'sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1']), '']
    for name, values in old['recorded_candidate_holdout'].items():
        lines += ['### ' + name + ' old-test confusion and per-class performance', '',
            table([dict(actual=label, **dict(zip(OUTCOMES, row))) for label, row in zip(OUTCOMES, values['confusion_matrix'])], ['actual'] + OUTCOMES), '',
            table([dict(outcome=label, **values['classification_report'][label]) for label in OUTCOMES], ['outcome', 'precision', 'recall', 'f1-score', 'support']), '']
    lines += ['- ' + conclusion for conclusion in old['conclusions']]
    lines += ['', '### Uncertainty and feature diagnostics', '', '```json',
        __import__('json').dumps(dict(uncertainty=old['uncertainty'], features=old['features']), indent=2), '```', '',
        'The original regime/month tables, per-class reports and baselines are in comparison.json. They are descriptive, post-hoc and subject to small-sample/multiple-comparison uncertainty.', '',
        '## Frozen experimental design', '',
        'Protocol SHA256: `' + report['protocol_sha256'] + '`.', '',
        'The new reserve is May–October 2024, collected only after the plan was frozen. Earlier data from January 2022 onward supplies development. All feature and target timestamps are purged at the 24-hour boundary. All supervised origins are 25 hours apart and both feature recipes use identical origins.', '',
        'Inner chronological folds select hyperparameters; outer chronological folds compare tuning procedures. The final family/feature group and parameters were frozen before accessing reserve targets. HistGradientBoosting uses fixed iteration counts and early_stopping=False, with no random validation split.', '',
        'The previously saved production models were trained on later data, so they cannot be applied to the earlier 2024 reserve as leakage-free benchmarks. LR/RF algorithm comparisons here are fresh earlier-data fits, not reuse of later-trained weights.', '',
        'The final 45-origin calibration block is excluded from all family/feature selection and hyperparameter tuning. Each outer fold also keeps its own calibration block out of inner tuning. Both feature recipes use common eligible observations after the longest warm-up.', '',
        '### Source quality', '', '```json', __import__('json').dumps(dict(raw_quality=report.get('data_quality'),
             development_handling=report['data'].get('development_gap_handling')), indent=2), '```', '',
        'The first strict collector stopped before fitting on an absent hour; a second validation stopped on an adjacent partial hourly candle. Both unopened protocols were retained with explicit amendment reasons before the revised method ran. No candle was invented and no scored reserve was reused.', '',
        '## Nested development comparison', '', table(outer_rows, ['model'] + METRICS), '',
        '**Selected procedure:** ' + report['selected_group'] + '. Parameters: `' + str(report['selected_params']) + '`.', '',
        'The winning outer score is used for selection and can be optimistic. Only the subsequent reserved block is independent of this family selection.', '']
    fold_rows = []
    for name, candidate in report['candidates'].items():
        for fold in candidate['folds']:
            fold_rows.append(dict(model=name, fold=fold['fold'], train_end=fold['train_end'], calibration_end=fold['calibration_end'],
                validation_start=fold['test_start'], validation_end=fold['test_end'],
                train_macro_f1=fold['train_metrics']['macro_f1'], validation_macro_f1=fold['metrics']['macro_f1'],
                majority_macro_f1=fold['baselines']['majority']['macro_f1'], technical_macro_f1=fold['baselines']['technical_rule']['macro_f1'],
                accuracy=fold['metrics']['accuracy'], balanced_accuracy=fold['metrics']['balanced_accuracy']))
    pd.DataFrame(fold_rows).to_csv(folder / 'fold_comparison.csv', index=False)
    lines += [table(fold_rows, ['model', 'fold', 'train_macro_f1', 'validation_macro_f1', 'majority_macro_f1', 'technical_macro_f1']), '']
    if report['final_reserve_scored']:
        final = report['final_metrics']
        final_rows = [metric_row(report['selected_group'], final)]
        final_rows += [metric_row(name, values) for name, values in report['final_baselines'].items()]
        pd.DataFrame(final_rows).to_csv(folder / 'reserved_test_comparison.csv', index=False)
        lines += ['## Once-scored newly collected historical reserve', '', table(final_rows, ['model'] + METRICS), '',
            '**This is an older, newly collected historical reserve, not prospective evidence for today’s market.** The original 2026 test was not relabeled as untouched. Its accuracy and this reserve accuracy refer to different periods and must not be treated as a direct before/after improvement.', '',
            'Final test period (UTC): ' + report['final_split']['test_start'] + ' to ' + report['final_split']['test_end'] + '.', '',
            'Observed outcome counts: `' + str(final['outcome_distribution']) + '`.', '',
            '### Confusion matrix (rows actual, columns predicted)', '',
            table([dict(actual=label, **dict(zip(OUTCOMES, values))) for label, values in zip(OUTCOMES, final['confusion_matrix'])], ['actual'] + OUTCOMES), '',
            '### Per-class precision, recall and F1', '',
            table([dict(outcome=label, **final['classification_report'][label]) for label in OUTCOMES], ['outcome', 'precision', 'recall', 'f1-score', 'support']), '',
            '### Calibration and stability', '',
            'Temperature scaling is fitted on an earlier calibration partition. Low calibration error alone cannot establish useful discrimination. Multiclass Brier is summed over three outcomes (range 0–2). Majority and technical-rule vectors are deterministic one-hot outputs; compare probability quality primarily against the training-prior distribution.', '',
            '```json', __import__('json').dumps(dict(quality=report['quality'], consistency=report['consistency'],
                paired_block_intervals=report['paired_block_intervals']), indent=2), '```', '',
            'Paired intervals use circular moving blocks of seven non-overlapping origins and 1,000 seeded resamples. These approximate historical uncertainty under limited assumptions; they are not future-event confidence or financial guarantees.', '']
        for failure in report['quality']['failures']:
            lines.append('- ' + QUALITY_MESSAGES.get(failure, failure))
        lines += ['', '### Market conditions', '', 'Volatility cutoffs are learned on training observations only. Month/regime slices below are post-hoc descriptive checks; groups with fewer than 30 examples are especially unstable.', '']
        subgroup_rows = []
        for grouping in ('trend_regime', 'volatility_regime', 'combined_regime', 'month'):
            for value in report['final_regime_results'][grouping]:
                subgroup_rows.append(dict(grouping=grouping, value=value[grouping], small_sample=value['small_sample'], **{key: value['metrics'][key] for key in METRICS}))
        pd.DataFrame(subgroup_rows).to_csv(folder / 'reserved_regime_metrics.csv', index=False)
        lines += [table(subgroup_rows, ['grouping', 'value', 'sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'small_sample']), '',
            'Production decision: **' + report['promotion_decision'] + '**', '']
        for label in OUTCOMES:
            class_rows.append(dict(stage='new_historical_reserve', model=report['selected_group'], outcome=label, **final['classification_report'][label]))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        matrix = np.array(final['confusion_matrix'])
        axes[0].imshow(matrix, cmap='Blues')
        axes[0].set(xticks=range(3), yticks=range(3), xticklabels=OUTCOMES, yticklabels=OUTCOMES,
            xlabel='Predicted', ylabel='Actual', title=f'New historical reserve · n={final["sample_size"]}')
        for row in range(3):
            for column in range(3):
                axes[0].text(column, row, str(matrix[row, column]), ha='center', va='center', color='white' if matrix[row, column] > matrix.max() * .6 else 'black')
        bins = final['reliability_bins']
        axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration reference')
        axes[1].plot([value['mean_probability'] for value in bins], [value['observed_accuracy'] for value in bins], 'o-', label='Observed held-out bins')
        for value in bins:
            location = (value['mean_probability'], value['observed_accuracy'])
            offset = (-25, -16) if value['observed_accuracy'] > .9 else (-25, 10)
            axes[1].annotate('n=' + str(value['count']), location, xytext=offset, textcoords='offset points', fontsize=8)
        axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean top-class probability', ylabel='Observed correct fraction', title='Historical calibration diagnostic')
        axes[1].legend(fontsize=8)
        axes[1].text(.02, .96, 'Sparse bins are unstable', transform=axes[1].transAxes, va='top', fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / 'reserved_diagnostics.png', dpi=160)
        plt.close(fig)
        lines += ['![Confusion matrix and calibration diagnostic](reserved_diagnostics.png)', '']
    else:
        lines += ['## Final evaluation', '', 'Development-only run: reserved target labels were not accessed and no final test metric is claimed.', '']
    pd.DataFrame(class_rows).to_csv(folder / 'class_reports.csv', index=False)
    lines += ['## Limitations and preservation', ''] + ['- ' + value for value in report['limitations']]
    lines += ['', 'Protected production files checked: ' + str(report['preservation']['files_checked']) + '; changed: ' + str(len(report['preservation']['changed'])) + '.', '',
        'No dashboard change, probability release, trading, leverage or automatic promotion was implemented.', '',
        'This is for education only, not financial advice. Crypto is high risk.', '']
    (folder / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return {key: str((folder / filename).resolve()) for key, filename in [('markdown', 'comparison.md'), ('json', 'comparison.json'), ('csv', 'nested_comparison.csv')]}
