"""Reproducible read-only input audit; writes only derived audit reports."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_conditions_study import (ANCHOR, REFERENCE_END, ROOT, active_reserves,
    bootstrap_indices, describe_group, regime_rows)
from news_label_safety import atomic_json, sha256
from trend_experiment_features import segmented_experiment_dataset
from trend_market_data import read_candles


def candle_inventory(path):
    frame = read_candles(path)
    complete = (frame.close_time - frame.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1))
    expected = pd.date_range(frame.open_time.min(), frame.open_time.max(), freq='h')
    state_path = Path(path).parent / 'collection_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    return frame, dict(file=str(path), collected_at=state.get('collected_at'), rows=len(frame), unique_hour_timestamps=frame.open_time.nunique(),
        complete_hourly_candles=int(complete.sum()), incomplete_rows=int((~complete).sum()),
        earliest_open=frame.open_time.min().isoformat(), latest_open=frame.open_time.max().isoformat(),
        earliest_close=frame.close_time.min().isoformat(), latest_close=frame.close_time.max().isoformat(),
        duplicate_open_timestamps=int(frame.open_time.duplicated().sum()), duplicate_close_timestamps=int(frame.close_time.duplicated().sum()),
        missing_hour_timestamps=[date.isoformat() for date in expected.difference(pd.DatetimeIndex(frame.open_time))], sha256=sha256(path))


def window_overlap(observations, starts, ends):
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    mask = np.zeros(len(observations), bool)
    matched = 0
    for start, end in zip(starts, ends):
        hit = (observations.close_time < end) & (observations.target_end > start)
        mask |= hit.to_numpy()
        matched += bool(hit.any())
    return dict(descriptive_observations_overlapping=int(mask.sum()), ml_intervals_overlapping=int(matched),
                exact_origin_matches=int(observations.close_time.isin(starts).sum()))


def test_exposure(root, observations):
    root = Path(root)
    result = []
    contract_path = root / 'artifacts' / 'trend_24h_validation.json'
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        path = root / 'artifacts' / contract['oos_predictions_file']
        oos = pd.read_csv(path)
        starts = pd.to_datetime(oos.available_at, utc=True)
        result.append(dict(name='Production trend model historical test', status='already_evaluated',
            start=starts.min().isoformat(), end=(starts.max() + pd.Timedelta(hours=24)).isoformat(),
            ml_samples=len(oos), **window_overlap(observations, starts, starts + pd.Timedelta(hours=24))))
    registry = root / 'data' / 'trend_experiments' / 'reserve_registry.json'
    if registry.exists():
        for entry in json.loads(registry.read_text())['entries']:
            if entry['status'] != 'evaluated':
                continue
            path = (root / entry['result_path']).parent / 'final_predictions.csv'
            data = pd.read_csv(path)
            starts, ends = pd.to_datetime(data.available_at, utc=True), pd.to_datetime(data.target_end, utc=True)
            result.append(dict(name='Additional historical experiment reserve', status='already_evaluated',
                start=starts.min().isoformat(), end=ends.max().isoformat(), ml_samples=len(data),
                **window_overlap(observations, starts, ends)))
    return result


def generate_audit(root=ROOT):
    root = Path(root)
    folder = root / 'reports' / 'market_conditions'
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    primary = root / 'data' / 'trend' / 'xrp_hourly.csv'
    extended = root / report['data']['file']
    old, old_info = candle_inventory(primary)
    raw, raw_info = candle_inventory(extended)
    observations = pd.read_csv(folder / 'observations.csv')
    for column in ('close_time', 'target_end'):
        observations[column] = pd.to_datetime(observations[column], utc=True)
    reconstructed = regime_rows(raw)
    complete = raw[(raw.close_time - raw.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1))]
    hourly = segmented_experiment_dataset(complete)
    eligible_volatility = len(hourly) - sum(min(720, len(group)) for _, group in hourly.groupby('contiguous_segment'))
    reserves = active_reserves(root)
    from market_conditions_study import outside_reserves
    keep = outside_reserves(reconstructed.close_time, reconstructed.target_end, reserves)
    expected = reconstructed[keep & reconstructed.close_time.ge(REFERENCE_END)].reset_index(drop=True)
    price_map = raw.set_index('close_time').close
    actual_returns = observations.target_end.map(price_map).to_numpy() / observations.close_time.map(price_map).to_numpy() - 1
    expected_labels = np.select([actual_returns >= .01, actual_returns <= -.01], ['UP', 'DOWN'], default='NEUTRAL')
    spacing = observations.close_time.diff().dropna() / pd.Timedelta(hours=1)
    overlaps = int((observations.close_time.iloc[1:].reset_index(drop=True) < observations.target_end.iloc[:-1].reset_index(drop=True)).sum())
    complete_dates = pd.DatetimeIndex(complete.close_time)
    gap_free = all(pd.date_range(row.close_time, row.target_end, freq='h').isin(complete_dates).all()
                   for row in observations[['close_time', 'target_end']].itertuples())
    comparisons = old.merge(raw, on='open_time', suffixes=('_old', '_new'))
    different = np.zeros(len(comparisons), bool)
    for column in ('open', 'high', 'low', 'close', 'volume', 'quote_volume'):
        different |= ~np.isclose(comparisons[column + '_old'], comparisons[column + '_new'], rtol=0, atol=1e-10)
    different |= comparisons.close_time_old.ne(comparisons.close_time_new).to_numpy()
    # Independently recompute each selected volatility threshold from strictly prior values.
    threshold_ok = True
    for _, segment in hourly.groupby('contiguous_segment', sort=False):
        positions = {date: index for index, date in enumerate(segment.close_time)}
        values = segment.volatility_24h.to_numpy()
        selected = observations[observations.close_time.isin(positions)]
        for _, row in selected.iterrows():
            position = positions[row.close_time]
            threshold_ok &= position >= 720 and bool(np.isclose(np.quantile(values[max(0, position - 2160):position], .75), row.high_volatility_threshold))
    checks = dict(no_raw_timestamp_duplicates=raw_info['duplicate_open_timestamps'] == raw_info['duplicate_close_timestamps'] == 0,
        no_duplicate_observations=not observations.close_time.duplicated().any(),
        ordered_origins=observations.close_time.is_monotonic_increasing,
        exact_24_hour_horizons=bool((observations.target_end - observations.close_time).eq(pd.Timedelta(hours=24)).all()),
        every_outcome_has_25_complete_hourly_closes=bool(gap_free),
        disjoint_outcome_intervals=overlaps == 0, minimum_25_hour_spacing=bool((spacing >= 25).all()),
        reconstructed_origins_match=list(expected.close_time) == list(observations.close_time),
        reconstructed_regimes_match=list(expected.regime) == list(observations.regime),
        reconstructed_high_volatility_match=list(expected.high_volatility) == list(observations.high_volatility),
        independently_recomputed_thresholds_match=bool(threshold_ok),
        independently_recomputed_outcomes_match=bool(np.array_equal(expected_labels, observations.target)),
        stored_returns_match=bool(np.allclose(actual_returns, observations.future_return_24h)),
        original_extended_overlap_agrees=not bool(different.any()),
        source_hash_matches_report=sha256(extended) == report['data']['sha256'],
        observations_hash_matches_report=sha256(folder / 'observations.csv') == report['observations_sha256'])
    if not all(checks.values()):
        raise ValueError('Research audit failed: ' + ', '.join(key for key, value in checks.items() if not value))
    sensitivity = {}
    for block in (3, 7, 14):
        indices = bootstrap_indices(observations.close_time, block=block)
        sensitivity[str(block)] = {name: describe_group(observations,
            observations.high_volatility.to_numpy() if name == 'High volatility' else observations.regime.eq(name).to_numpy(),
            indices, report['reference']['majority']) for name in report['groups']}
    union = pd.Index(old.open_time).union(pd.Index(raw.open_time))
    model_contract = root / 'artifacts' / 'trend_24h_validation.json'
    model_evaluated_at = json.loads(model_contract.read_text()).get('evaluated_at') if model_contract.exists() else None
    audit = dict(audited_at=pd.Timestamp.now(tz='UTC').isoformat(), passed=True, checks=checks,
        original_cache=old_info, study_cache=raw_info, unique_union_hour_timestamps=len(union),
        additional_unique_hours=len(pd.Index(raw.open_time).difference(pd.Index(old.open_time))),
        shared_hours=len(comparisons), differing_shared_rows=int(different.sum()),
        model_evaluated_at=model_evaluated_at,
        extended_collection_after_model_evaluation=bool(pd.Timestamp(raw_info['collected_at']) > pd.Timestamp(model_evaluated_at)) if raw_info['collected_at'] and model_evaluated_at else None,
        waterfall=dict(raw_rows=len(raw), complete_rows=len(complete), feature_and_known_target_rows=len(hourly),
            volatility_ready_hourly_rows=eligible_volatility, fixed_grid_origins=len(reconstructed),
            reference_origins=int((reconstructed.target_end < REFERENCE_END).sum()),
            reference_boundary_purged=int(((reconstructed.close_time < REFERENCE_END) & (reconstructed.target_end >= REFERENCE_END)).sum()),
            active_reserve_excluded=int((~keep).sum()), descriptive_origins=len(observations)),
        interval_audit=dict(overlaps=overlaps, minimum_spacing_hours=float(spacing.min()), maximum_spacing_hours=float(spacing.max()),
            duplicate_observation_rows=int(observations.duplicated().sum()),
            note='Non-overlap concerns future return windows, not statistical independence of market regimes.'),
        ml_test_exposure=test_exposure(root, observations), active_ml_reserves=len(reserves),
        interval_sensitivity=sensitivity, frequencies=report['groups'],
        uncertainty_assessment='Counts and denominators are verified. Seven-origin blocks approximate short-range dependence; 3/7/14-origin sensitivity is provided. Neither these intervals nor nominal Wilson intervals fully account for nonstationary, long-range dependence. They are pointwise historical uncertainty, never forecast confidence.',
        test_separation='The descriptive study includes previously evaluated ML test history. Those periods are already exposed and cannot be reused as untouched evidence. Active unfinished ML reserves are omitted from reference, evaluation and news outputs.',
        predictive_probabilities='Remain hidden; no validation artifact is modified.', report_sha256=sha256(folder / 'report.json'))
    atomic_json(folder / 'audit.json', audit)
    lines = ['# Market Conditions Research audit', '', 'All timestamps below are UTC. No new candles or model fits were produced during this audit.', '',
        '| Cache | Rows | Earliest open | Latest close |', '|---|---:|---|---|',
        f'| Original model cache | {len(old)} | {old_info["earliest_open"]} | {old_info["latest_close"]} |',
        f'| Extended study cache | {len(raw)} | {raw_info["earliest_open"]} | {raw_info["latest_close"]} |', '',
        f'Unique union: {len(union)} hour timestamps, not the sum of the caches. Later collection added {audit["additional_unique_hours"]} unique hours; {len(comparisons)} shared records agree.', '',
        '## Exact timestamp boundaries', '']
    for name, value in [('Original', old_info), ('Extended', raw_info)]:
        lines += [f'- {name}: first open `{value["earliest_open"]}`, first close `{value["earliest_close"]}`, last open `{value["latest_open"]}`, last close `{value["latest_close"]}`.']
    lines += ['', '## Selection and interval checks', '', '```json', json.dumps(audit['waterfall'], indent=2), '```', '',
        'The fixed grid is 25 hours from 2022-01-01T00:59:59.999Z. Complete contiguous candles, feature warm-up, prior-only volatility threshold warm-up and exact known future closes are required. Reference data is before 2023; one boundary origin is purged. No incomplete or gap-crossing outcome is filled.', '',
        f'Outcome intervals last exactly 24 hours; minimum origin spacing is {spacing.min():.0f} hours. Pairwise adjacent overlaps: {overlaps}. Duplicate raw timestamps: 0; duplicate observation timestamps: 0. The longest spacing is {spacing.max():.0f} hours because gap recovery and warm-up exclude origins.', '',
        f'Collection metadata: original cache `{old_info["collected_at"]}`; extended cache `{raw_info["collected_at"]}`; previous model evaluation `{model_evaluated_at}`. Extended collection after model evaluation: {audit["extended_collection_after_model_evaluation"]}. No new collection occurred during this audit.', '',
        '## Actual historical outcomes', '', '| Condition | n | UP | DOWN | NEUTRAL |', '|---|---:|---:|---:|---:|']
    for name, values in report['groups'].items():
        cells = [f'{values["outcomes"][label]["historical_frequency"]:.1%}' for label in ('UP', 'DOWN', 'NEUTRAL')]
        lines.append('| ' + ' | '.join([name, str(values['samples'])] + cells) + ' |')
    lines += ['', 'High volatility overlaps the three trend groups. Frequencies are historical descriptions, not validated future probabilities.', '',
        '## Causality, uncertainty and ML separation', '',
        'EMA regimes and every selected trailing volatility threshold reproduce from earlier candles only. Future target labels are separately recomputed from the price at exactly t+24h. No data normalization or threshold is fitted on the full future sample.', '',
        audit['uncertainty_assessment'], '', audit['test_separation'], '',
        '```json', json.dumps(audit['ml_test_exposure'], indent=2), '```', '',
        'Correction: source-data lineage and explicit ML-test overlap disclosures are now visible in the dashboard. An unfinished-reserve exclusion guard was added. A silent independent-row fallback for unavailable bootstrap blocks was removed. Actual counts and point frequencies did not change.', '',
        'One partial source bar (2023-03-24 12:00, ending 12:39:48.301) is preserved in raw data but excluded; the 13:00 source hour is absent. Feature/target windows restart afterward. News timing is insufficient for an event-association estimate.', '',
        'All independent audit checks passed. Automated test results and protected-file hashes are reported with the task delivery. Models and validation status are unchanged.']
    (folder / 'audit.md').write_text('\n'.join(lines), encoding='utf-8')
    return audit


if __name__ == '__main__':
    result = generate_audit()
    print(json.dumps({key: result[key] for key in ('passed', 'waterfall', 'interval_audit', 'ml_test_exposure')}, indent=2))
