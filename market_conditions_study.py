"""Descriptive causal regime/outcome study, never a model validation update."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_label_safety import atomic_json, sha256
from trend_diagnosis import wilson_interval
from trend_experiment_features import segmented_experiment_dataset
from trend_market_data import read_candles, save_frame
from trend_metrics import OUTCOMES
from trend_news import _load as load_timed_news, audit_news

ROOT = Path(__file__).resolve().parent
REFERENCE_END = pd.Timestamp('2023-01-01T00:00:00Z')
ANCHOR = pd.Timestamp('2022-01-01T00:59:59.999Z')
DISCLAIMER = 'Historical outcome frequencies are descriptions of this dataset, not validated future probabilities.'
LIMITATIONS = [DISCLAIMER,
    '25-hour origins avoid overlapping 24-hour outcomes; persistent market regimes and shared feature lookbacks still create dependence.',
    'Intervals are pointwise descriptive estimates, not simultaneous significance tests. Many regimes/years are compared.',
    'Moving-block intervals assume local temporal structure is informative; seven-origin blocks do not capture every long regime or structural break.',
    'Wilson intervals assume independent Bernoulli observations and can understate uncertainty under dependence.',
    'All data has already been available for earlier research. This is not a new untouched forecasting evaluation.',
    'Regime thresholds are fixed explanatory conventions, not optimized profitable signals. No causal news effect is established.',
    'Exchange data vintages, publication accuracy, transaction costs, slippage and execution latency are not independently verified.',
]


def active_reserves(root):
    path = Path(root) / 'data' / 'trend_experiments' / 'reserve_registry.json'
    if not path.exists():
        return []
    entries = json.loads(path.read_text(encoding='utf-8'))['entries']
    result = []
    for entry in entries:
        if entry['status'] not in ('reserved', 'opened', 'evaluated', 'superseded_unopened'):
            raise ValueError('Unrecognized ML reserve status; refusing to expose research outcomes.')
        if entry['status'] in ('reserved', 'opened'):
            start, end = pd.Timestamp(entry['start']), pd.Timestamp(entry['end'])
            if start.tzinfo is None or end.tzinfo is None or start >= end:
                raise ValueError('Invalid ML reserve boundaries.')
            result.append(dict(start=start, end=end, protocol=entry['protocol_sha256']))
    return result


def outside_reserves(starts, ends, reserves):
    keep = np.ones(len(starts), dtype=bool)
    for window in reserves:
        keep &= ~((starts < window['end']) & (ends >= window['start'])).to_numpy()
    return keep


def regime_rows(candles):
    complete = (candles.close_time - candles.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1))
    frame = segmented_experiment_dataset(candles[complete].copy())
    quality = dict(frame.attrs)
    quality['incomplete_candles_excluded'] = int((~complete).sum())
    parts = []
    for _, segment in frame.groupby('contiguous_segment', sort=False):
        segment = segment.copy()
        # Previous 90 days of hourly volatility; minimum 30 days. Never full-sample fitting.
        segment['high_volatility_threshold'] = segment.volatility_24h.shift(1).rolling(2160, min_periods=720).quantile(.75)
        segment['regime'] = np.select([segment.ema_gap > .001, segment.ema_gap < -.001],
                                      ['Bullish', 'Bearish'], default='Sideways')
        segment['high_volatility'] = segment.volatility_24h > segment.high_volatility_threshold
        parts.append(segment)
    result = pd.concat(parts, ignore_index=True).dropna(subset=['high_volatility_threshold'])
    offsets = (result.close_time - ANCHOR) / pd.Timedelta(hours=1)
    keep = np.isclose(offsets, offsets.round()) & offsets.ge(0) & offsets.round().astype('int64').mod(25).eq(0)
    result = result[keep].reset_index(drop=True)
    result['technical_outcome'] = result.technical.map({'Bullish': 'UP', 'Bearish': 'DOWN', 'Neutral': 'NEUTRAL'})
    result.attrs.update(quality)
    return result


def bootstrap_indices(times, repeats=1500, block=7, seed=42):
    """Contiguous moving blocks do not bridge missing source/warm-up periods."""
    dates = pd.DatetimeIndex(times)
    n = len(dates)
    if n == 0:
        return np.empty((repeats, 0), dtype=int)
    block = min(block, n)
    valid = [start for start in range(n - block + 1)
             if block == 1 or np.all(np.diff(dates[start:start + block].asi8) == pd.Timedelta(hours=25).value)]
    if not valid:
        raise ValueError('No contiguous moving blocks are available; refusing an independent-row fallback.')
    rng = np.random.default_rng(seed)
    starts = rng.choice(valid, size=(repeats, int(np.ceil(n / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(repeats, -1)[:, :n]


def describe_group(frame, mask, indices, majority):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    sampled_mask = mask[indices]
    denominators = sampled_mask.sum(axis=1)
    valid = denominators > 0
    outcomes = {}
    for label in OUTCOMES:
        positive = frame.target.eq(label).to_numpy()
        count = int((mask & positive).sum())
        values = (sampled_mask & positive[indices]).sum(axis=1)[valid] / denominators[valid]
        outcomes[label] = dict(count=count, historical_frequency=count / n if n else None,
            nominal_wilson_95=wilson_interval(count, n) if n else None,
            block_95=dict(lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975))) if len(values) else None)
    baselines = {}
    for name, predicted in [('Reference-majority', np.full(len(frame), majority)), ('Technical rule', frame.technical_outcome.to_numpy())]:
        correct = predicted == frame.target.to_numpy()
        scores = (sampled_mask & correct[indices]).sum(axis=1)[valid] / denominators[valid]
        baselines[name] = dict(matches=int((mask & correct).sum()), samples=n,
            historical_match_rate=float(correct[mask].mean()) if n else None,
            block_95=dict(lower=float(np.quantile(scores, .025)), upper=float(np.quantile(scores, .975))) if len(scores) else None)
    return dict(samples=n, small_sample=n < 30, outcomes=outcomes, baselines=baselines,
                bootstrap_valid_replicates=int(valid.sum()))


def news_associations(root, candles, study_rows, reserves=()):
    """News entry is the first completed candle AT/AFTER feature availability."""
    audit = audit_news(root)
    try:
        events, _, _, _ = load_timed_news(root)
    except (OSError, ValueError, KeyError) as exc:
        return dict(estimable=False, reason=str(exc), timing_audit=audit)
    complete = candles[(candles.close_time - candles.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1))].copy()
    complete = complete.sort_values('close_time')
    indexed = complete.set_index('close_time')
    hourly_features = segmented_experiment_dataset(complete).set_index('close_time')
    accepted, excluded = [], dict(no_mature_gap_free_outcome=0, overlapping_event_window=0, protected_ml_reserve=0)
    previous_end = None
    for event in sorted(events, key=lambda value: value['feature_available_at']):
        available = pd.Timestamp(event['feature_available_at'])
        candidates = complete[complete.close_time >= available]
        if candidates.empty:
            excluded['no_mature_gap_free_outcome'] += 1
            continue
        entry = candidates.close_time.iloc[0]
        end = entry + pd.Timedelta(hours=24)
        if any(entry < window['end'] and end >= window['start'] for window in reserves):
            excluded['protected_ml_reserve'] += 1
            continue
        window = indexed.loc[entry:end]
        if (entry - available > pd.Timedelta(hours=1) or len(window) != 25 or end not in indexed.index
                or not window.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()):
            excluded['no_mature_gap_free_outcome'] += 1
            continue
        if previous_end is not None and entry < previous_end:
            excluded['overlapping_event_window'] += 1
            continue
        change = float(indexed.loc[end, 'close'] / indexed.loc[entry, 'close'] - 1)
        accepted.append(dict(id=event['id'], role=event['role'], label=event['label'], available_at=available.isoformat(),
            entry_at=entry.isoformat(), target_end=end.isoformat(), return_24h=change,
            outcome='UP' if change >= .01 else 'DOWN' if change <= -.01 else 'NEUTRAL'))
        previous_end = end
    # Controls are earlier, same trend-regime observations with no reused/overlapping windows.
    paired, used = [], set()
    for event in accepted:
        entry = pd.Timestamp(event['entry_at'])
        if entry not in hourly_features.index:
            continue
        gap = hourly_features.loc[entry, 'ema_gap']
        regime = 'Bullish' if gap > .001 else 'Bearish' if gap < -.001 else 'Sideways'
        candidates = study_rows[(study_rows.target_end < entry) & (study_rows.close_time >= entry - pd.Timedelta(days=90))
                                & study_rows.regime.eq(regime)]
        for _, control in candidates.iloc[::-1].iterrows():
            key = control.close_time.isoformat()
            overlaps = any(control.close_time < pd.Timestamp(value['target_end']) and control.target_end > pd.Timestamp(value['entry_at']) for value in accepted)
            used_overlap = any(control.close_time < end and control.target_end > start for start, end in used)
            if overlaps or used_overlap:
                continue
            used.add((control.close_time, control.target_end))
            paired.append(dict(event_id=event['id'], role=event['role'], label=event['label'],
                event_return=event['return_24h'], control_return=float(control.future_return_24h),
                control_at=key, event_at=event['entry_at']))
            break
    days = len({value['entry_at'][:10] for value in accepted})
    enough = len(paired) >= 30 and days >= 10
    return dict(estimable=enough, trustworthy_timed_records=len(events), matured_nonoverlapping_events=len(accepted),
        matched_prior_controls=len(paired), distinct_event_days=days, excluded=excluded, timing_audit=audit,
        reason='Exploratory association only; no causal effect or future probability.' if enough else
               'Insufficient trustworthy, matured, non-overlapping news events and matched controls (need at least 30 pairs over 10 days).',
        event_counts={label: sum(value['label'] == label for value in accepted) for label in audit.get('raw_class_distribution', {}).get('gold', {})},
        mean_event_minus_control_return=float(np.mean([value['event_return'] - value['control_return'] for value in paired])) if enough else None,
        events=accepted, pairs=paired,
        limitation='Gold and Silver remain separate in event provenance. Timing eligibility is not factual verification. Matching is observational and cannot remove confounding or selection bias.')


def build_study(candles, root=ROOT, repeats=1500):
    rows = regime_rows(candles)
    reserves = active_reserves(root)
    keep = outside_reserves(rows.close_time, rows.target_end, reserves)
    excluded_count = int((~keep).sum())
    rows = rows[keep].copy()
    reference = rows[rows.target_end < REFERENCE_END]
    evaluation = rows[rows.close_time >= REFERENCE_END].reset_index(drop=True)
    if len(reference) < 100 or len(evaluation) < 100:
        raise ValueError('Need at least 100 reference and 100 later non-overlapping observations.')
    majority = reference.target.value_counts().reindex(OUTCOMES, fill_value=0).idxmax()
    indices = bootstrap_indices(evaluation.close_time, repeats=repeats)
    groups = {}
    for name in ('Bullish', 'Bearish', 'Sideways', 'High volatility'):
        mask = evaluation.high_volatility.to_numpy() if name == 'High volatility' else evaluation.regime.eq(name).to_numpy()
        groups[name] = describe_group(evaluation, mask, indices, majority)
    annual = []
    for year in sorted(evaluation.close_time.dt.year.unique()):
        for name in groups:
            mask = evaluation.close_time.dt.year.eq(year).to_numpy()
            mask &= evaluation.high_volatility.to_numpy() if name == 'High volatility' else evaluation.regime.eq(name).to_numpy()
            annual.append(dict(year=int(year), regime=name, **describe_group(evaluation, mask, indices, majority)))
    report = dict(schema_version=1, created_at=pd.Timestamp.now(tz='UTC').isoformat(), study_type='descriptive_historical_outcomes',
        disclaimer=DISCLAIMER, model_validation_status_changed=False,
        method=dict(ema_gap_threshold=.001, high_volatility='vol24 above 75th percentile of preceding 2160 hourly volatility values, shifted one hour; minimum720',
            horizon_hours=24, origin_stride_hours=25, anchor=ANCHOR.isoformat(), reference_end_exclusive=REFERENCE_END.isoformat(),
            bootstrap_replicates=repeats, block_origins=7, seed=42, interval_level=.95,
            regimes_overlap='Bullish/Bearish/Sideways are exclusive. High volatility overlaps any of them; its count must not be added to their total.'),
        ml_separation=dict(active_reserve_windows=len(reserves), excluded_observations=excluded_count,
            policy='Exclude unfinished reserved/opened ML intervals from both reference/evaluation counts and news-event outputs. Previously evaluated tests remain descriptive history, never untouched evidence.'),
        data=dict(hourly_rows=len(candles), start=candles.close_time.min().isoformat(), end=candles.close_time.max().isoformat(),
            source='Existing public Binance XRPUSDT 1h cache', quality=rows.attrs),
        reference=dict(samples=len(reference), start=reference.close_time.min().isoformat(), end=reference.close_time.max().isoformat(),
            class_counts={label: int(reference.target.eq(label).sum()) for label in OUTCOMES}, majority=majority),
        evaluation=dict(samples=len(evaluation), start=evaluation.close_time.min().isoformat(), end=evaluation.close_time.max().isoformat(),
            class_counts={label: int(evaluation.target.eq(label).sum()) for label in OUTCOMES}),
        groups=groups, annual=annual, overall=describe_group(evaluation, np.ones(len(evaluation), dtype=bool), indices, majority),
        news=news_associations(root, candles, rows, reserves), limitations=LIMITATIONS)
    return report, evaluation


def run_study(root=ROOT):
    root = Path(root)
    path = root / 'data' / 'trend_experiments' / 'cache' / 'xrp_hourly.csv'
    if not path.exists():
        path = root / 'data' / 'trend' / 'xrp_hourly.csv'
    raw = read_candles(path)
    report, rows = build_study(raw, root)
    report['data'].update(file=str(path.relative_to(root)), sha256=sha256(path))
    folder = root / 'reports' / 'market_conditions'
    folder.mkdir(parents=True, exist_ok=True)
    save_frame(folder / 'observations.csv', rows)
    report['observations_sha256'] = sha256(folder / 'observations.csv')
    atomic_json(folder / 'report.json', report)
    lines = ['# Market Conditions Research', '', DISCLAIMER, '',
        f'Evaluation: {report["evaluation"]["start"]} to {report["evaluation"]["end"]}, n={len(rows)}.', '',
        '| Regime | n | UP count/frequency | DOWN count/frequency | NEUTRAL count/frequency |', '|---|---:|---:|---:|---:|']
    for name, group in report['groups'].items():
        cells = [f'{group["outcomes"][label]["count"]} / {group["outcomes"][label]["historical_frequency"]:.1%}' for label in OUTCOMES]
        lines.append('| ' + ' | '.join([name, str(group['samples'])] + cells) + ' |')
    lines += ['', 'Block and nominal Wilson intervals, year slices and same-origin baselines are in report.json and the dashboard.', '',
        '| Regime / outcome | Historical frequency | Moving-block 95% interval |', '|---|---:|---:|']
    for name, group in report['groups'].items():
        for label, outcome in group['outcomes'].items():
            interval = outcome['block_95']
            lines.append(f'| {name} / {label} | {outcome["historical_frequency"]:.1%} | {interval["lower"]:.1%}–{interval["upper"]:.1%} |')
    lines += ['', '| Regime | Reference-majority matches | Technical-rule matches |', '|---|---:|---:|']
    for name, group in report['groups'].items():
        base = group['baselines']
        lines.append(f'| {name} | {base["Reference-majority"]["historical_match_rate"]:.1%} | {base["Technical rule"]["historical_match_rate"]:.1%} |')
    lines += ['', 'News: ' + report['news']['reason'], ''] + ['- ' + item for item in LIMITATIONS]
    (folder / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    from audit_market_conditions import generate_audit
    generate_audit(root)
    return report


if __name__ == '__main__':
    result = run_study()
    print(json.dumps(dict(evaluation=result['evaluation'], groups=result['groups'], news={key: value for key, value in result['news'].items() if key not in ('events', 'pairs', 'timing_audit')}), indent=2))
