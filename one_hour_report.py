"""Derived readable 1h report with dependence-aware descriptive uncertainty."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from one_hour_trend import ROOT, verified_model
from news_label_safety import atomic_json
from trend_market_data import read_candles


def make_report(root=ROOT):
    root = Path(root)
    _, report, quality = verified_model(root)
    pointer = json.loads((root / 'artifacts/trend_1h_current.json').read_text())
    oos = pd.read_csv(root / pointer['files']['oos']['path'], float_precision='round_trip')
    times = pd.to_datetime(oos.available_at, utc=True)
    dates = pd.DatetimeIndex(times)
    intervals = {}
    for block in (24, 168):
        valid = np.array([i for i in range(len(times) - block + 1)
                          if np.all(np.diff(dates[i:i + block].asi8) == pd.Timedelta(hours=1).value)])
        if not len(valid):
            intervals[str(block)] = dict(reason='No contiguous blocks of this length.')
            continue
        rng = np.random.default_rng(42)
        starts = rng.choice(valid, size=(1000, int(np.ceil(len(oos) / block))))
        indices = (starts[:, :, None] + np.arange(block)).reshape(1000, -1)[:, :len(oos)]
        outcomes = np.array(['UP', 'DOWN', 'NEUTRAL'])
        actual = oos.true_outcome.to_numpy()
        predicted = outcomes[oos[['p_UP', 'p_DOWN', 'p_NEUTRAL']].values.argmax(axis=1)]
        current = predicted == actual
        intervals[str(block)] = {}
        for baseline in ('majority', 'technical_rule'):
            base = outcomes[oos[[baseline + '_' + label for label in outcomes]].values.argmax(axis=1)]
            difference = current.astype(float) - (base == actual).astype(float)
            sampled = difference[indices].mean(axis=1)
            intervals[str(block)][baseline] = dict(accuracy_difference=float(difference.mean()), lower=float(np.quantile(sampled, .025)), upper=float(np.quantile(sampled, .975)))
    raw = read_candles(root / pointer['files']['market']['path'])
    missing = pd.date_range(raw.open_time.min(), raw.open_time.max(), freq='h').difference(pd.DatetimeIndex(raw.open_time))
    partial = raw[(raw.close_time - raw.open_time) != pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)]
    folder = root / 'reports' / 'one_hour'
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / 'report.json', dict(training_report=report, numerical_quality=quality, paired_block_accuracy_intervals=intervals,
        source_quality=dict(rows=len(raw), missing_hours=len(missing), partial_rows=len(partial), duplicate_opens=int(raw.open_time.duplicated().sum())),
        uncertainty_note='Exploratory pointwise paired moving-block intervals, 1000 replicates. Blocks do not bridge source gaps; long-range/nonstationary dependence is not fully captured.'))
    comparison = []
    for name, value in report['candidates'].items():
        comparison.append(dict(model=name, scope='Development validation', **{key: value['validation'][key] for key in ('sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece')}))
    comparison.append(dict(model=report['selected_model'], scope='Final test', **{key: report['test'][key] for key in ('sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece')}))
    for name, values in report['baselines'].items():
        comparison.append(dict(model=name, scope='Same final test', **{key: values[key] for key in ('sample_size', 'accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece')}))
    pd.DataFrame(comparison).to_csv(folder / 'comparison.csv', index=False)
    lines = ['# Next 1 Hour — Research Outlook: evaluation', '',
        f'Selected family: **{report["selected_model"]}**, version `{report["version"]}`. Selection used validation only; the final test did not select a different family.', '',
        f'New target: next completed hourly close/current completed hourly close minus one. Final threshold is ±**{report["threshold_pct"]:.6f}%**, learned as the one-third quantile of absolute classifier-training next-hour returns. No 24-hour labels or fitted weights were reused.', '',
        '| Model | Scope | n | Accuracy | Balanced accuracy | Macro-F1 | Brier | Log loss | ECE |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        lines.append('| ' + ' | '.join([row['model'], row['scope'], str(row['sample_size'])] + [f'{row[key]:.4f}' for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'log_loss', 'calibration_ece')]) + ' |')
    lines += ['', '## Final per-class results', '', '| Outcome | Precision | Recall | F1 | n |', '|---|---:|---:|---:|---:|']
    for label in ('UP', 'DOWN', 'NEUTRAL'):
        row = report['test']['classification_report'][label]
        lines.append(f'| {label} | {row["precision"]:.4f} | {row["recall"]:.4f} | {row["f1-score"]:.4f} | {int(row["support"])} |')
    lines += ['', 'Confusion matrix, rows actual and columns predicted, ordered UP/DOWN/NEUTRAL:', '',
        '```json', json.dumps(report['test']['confusion_matrix']), '```', '',
        '## Chronology and thresholds', '', '```json', json.dumps(report['split'], indent=2), '```', '',
        'Each validation fold learns a fresh threshold using its earlier classifier-training data only. The final calibration block is excluded from model-family selection. Earlier labels must mature strictly before the next partition starts, leaving a one-hour purge. Within each partition, hourly return increments do not overlap, although adjacent outcomes share a price endpoint and market regimes remain dependent.', '',
        'The final 2021 historical block was fixed before collection and lies before the earliest existing 2022–2026 research cache. It is new historical evidence for this method, not a prospective track record. The old 24-hour model was trained on later data and was not applied to this earlier test.', '',
        '## Calibration and reliability', '',
        'A scalar temperature was fitted on the separate calibration block, with model weights fixed. Original and calibrated test scoring, reliability bins and all validation thresholds are in report.json. The final stored model is exactly the model tested; it was not refitted on the test or all data.', '',
        'Numerical historical policy passed: **' + str(quality['passed']) + '**. This is not proof of statistical/economic reliability. In particular, NEUTRAL recall is low, market periods differ and the test is old.', '',
        '**Current dashboard percentages remain hidden** because recent out-of-sample market evidence within 180 days is also required. Data freshness alone cannot make a 2021 evaluation current. The dashboard can show an experimental class only with the exact latest completed hourly candle.', '',
        'Paired accuracy-difference intervals versus baselines (historical, approximate, not future confidence):', '', '```json', json.dumps(intervals, indent=2), '```', '',
        '## Data quality and reproducibility', '',
        f'{len(raw)} raw historical public Binance records, {len(missing)} absent hourly timestamps and {len(partial)} non-standard/partial hourly bars. Duplicate opens: {raw.open_time.duplicated().sum()}. Source prices were never interpolated. Features/targets restart after missing or excluded bars, with 168-hour warm-up; this is why eligible calibration/test counts are smaller than calendar-hour counts.', '',
        'CSV verification uses explicit ISO8601 timestamp parsing and round-trip floating-point parsing. This preserves mixed fractional-second timestamps and exact tree inputs. The reader repairs did not retrain the model or change frozen labels/scores. Original training code was backed up before verification repairs.', '',
        'Repeat training commands return the completed frozen report rather than scoring this reserve again. A new design needs a separate new untouched study. The 24-hour models, reports, news labels, News Intelligence and Market Conditions Research are preserved.', '',
        'This is for education only, not financial advice. Crypto is high risk.']
    (folder / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return folder


if __name__ == '__main__':
    print(make_report())
