"""Isolated, predeclared nested chronological research; never changes production.

The 2026 holdout is already exposed. A new 2024 historical reserve is scored once
after selection using only pre-May2024 data, and is not prospective evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from experiment_market_data import collect_experiment_candles

from news_label_safety import atomic_json, sha256
from trend_diagnosis import diagnose_existing
from trend_experiment_features import (EXPANDED_FEATURES, EXPERIMENT_FEATURE_VERSION, FEATURE_GROUPS,
    causal_regimes, segmented_experiment_dataset)
from trend_market_data import collect_candles, read_candles, save_frame
from trend_metrics import OUTCOMES, metrics, quality_checks
from trend_pipeline import (FEATURES, ROOT, STRIDE, baseline_probabilities, fit_temperature,
    partitions, predict_bundle, probability_matrix, split_metadata)

GRID = {
 'LogisticRegression': [{'C': .1}, {'C': .5}, {'C': 2.}],
 'RandomForest': [{'max_depth': 6, 'min_samples_leaf': 12}, {'max_depth': 4, 'min_samples_leaf': 24}],
 'HistGradientBoosting': [{'max_leaf_nodes': 7, 'learning_rate': .05, 'l2_regularization': 1.},
                         {'max_leaf_nodes': 15, 'learning_rate': .05, 'l2_regularization': 5.}],
 'ExtraTrees': [{'max_depth': 6, 'min_samples_leaf': 12}, {'max_depth': 10, 'min_samples_leaf': 24}],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def protocol_definition():
    sources = [Path(__file__), Path(__file__).with_name('trend_experiment_features.py'), Path(__file__).with_name('experiment_market_data.py'),
               Path(__file__).with_name('trend_pipeline.py'), Path(__file__).with_name('trend_metrics.py')]
    return dict(schema_version=1, purpose='isolated nested chronological model investigation',
        training_start='2022-01-01T00:00:00Z', grid_anchor='2022-01-01T00:59:59.999Z',
        reserve_start='2024-05-01T00:00:00Z', reserve_end_exclusive='2024-11-01T00:00:00Z',
        holdout_type='newly_collected_historical_reserve_not_prospective', stride_hours=25, horizon_hours=24,
        outer_folds=3, outer_test_origins=60, inner_folds=2, inner_test_origins=45, calibration_origins=45,
        final_calibration_held_out_from_selection=True, outer_calibration_held_out_from_inner_tuning=True,
        gap_policy='Exclude non-standard/incomplete hourly candles; no filled candles; reset features and exclude warm-up and target windows at each contiguous-segment boundary.',
        selection='highest pooled outer macro-F1, then lowest Brier; final hyperparameters from inner development only',
        grids=GRID, feature_sets={'Base': FEATURES, 'Expanded': EXPANDED_FEATURES},
        feature_version=EXPERIMENT_FEATURE_VERSION, feature_groups=FEATURE_GROUPS,
        fixed_settings=dict(tree_estimators=200, hgb_iterations=150, hgb_early_stopping=False, random_state=42),
        promotion_policy='no automatic production replacement; existing quality policy plus consistent development gains and paired uncertainty checks',
        code_sha256={path.name: sha256(path) for path in sources})


def check_unseen_reserve(root, protocol):
    """Fail if any existing production market snapshot intersects the reserve."""
    paths = list((Path(root) / 'artifacts' / 'trend_runs').glob('*/market_snapshot.csv'))
    cached = Path(root) / 'data' / 'trend' / 'xrp_hourly.csv'
    if cached.exists():
        paths.append(cached)
    start, end = pd.Timestamp(protocol['reserve_start']), pd.Timestamp(protocol['reserve_end_exclusive'])
    evidence = []
    for path in paths:
        dates = pd.to_datetime(pd.read_csv(path, usecols=['close_time']).close_time, utc=True)
        if dates.between(start, end, inclusive='left').any():
            raise ValueError('Historical reserve was already in a production dataset: ' + str(path))
        evidence.append(dict(file=str(path.relative_to(root)), start=dates.min().isoformat(), end=dates.max().isoformat(), sha256=sha256(path)))
    return evidence


def reserve_protocol(folder, protocol, development_only=False):
    """Persist the plan before downloading reserved data or seeing its targets."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identifier = digest(protocol)
    path = folder / 'reserve_registry.json'
    registry = json.loads(path.read_text()) if path.exists() else {'entries': []}
    for entry in registry['entries']:
        overlap = (pd.Timestamp(protocol['reserve_start']) < pd.Timestamp(entry['end']) and
                   pd.Timestamp(entry['start']) < pd.Timestamp(protocol['reserve_end_exclusive']))
        if overlap and entry['protocol_sha256'] != identifier and entry.get('status') != 'superseded_unopened' and not development_only:
            raise ValueError('This reserve is already committed to another protocol. Use development-only research or new untouched dates.')
        if entry['protocol_sha256'] == identifier and not development_only:
            return identifier, entry
    protocol_path = folder / 'protocols' / (identifier + '.json')
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['protocol'] != protocol:
            raise ValueError('The immutable protocol file changed.')
    else:
        atomic_json(protocol_path, dict(protocol=protocol,
            protocol_sha256=identifier, frozen_at=pd.Timestamp.now(tz='UTC').isoformat(), development_only=development_only))
    entry = dict(protocol_sha256=identifier, start=protocol['reserve_start'], end=protocol['reserve_end_exclusive'],
                 status='reserved', reserved_at=pd.Timestamp.now(tz='UTC').isoformat())
    if not development_only:
        registry['entries'].append(entry)
        atomic_json(path, registry)
    return identifier, entry


def update_reserve(folder, identifier, **values):
    path = Path(folder) / 'reserve_registry.json'
    registry = json.loads(path.read_text())
    for entry in registry['entries']:
        if entry['protocol_sha256'] == identifier:
            entry.update(values)
    atomic_json(path, registry)


def supersede_unopened_protocol(folder, identifier, reason):
    """Explicit data-quality amendment is allowed only before any fitted model."""
    folder = Path(folder)
    path = folder / 'reserve_registry.json'
    registry = json.loads(path.read_text())
    old = next((row for row in registry['entries'] if row['protocol_sha256'] == identifier), None)
    run = folder.parents[1] / 'artifacts' / 'trend_experiments' / identifier[:16]
    evidence = [path.name for path in run.rglob('*') if path.is_file() and path.name not in ('protocol.json', 'production_manifest.json')]
    if not reason.strip() or old is None or old.get('status') != 'reserved' or evidence:
        raise ValueError('Only an unfitted, unopened data-quality protocol may be superseded.')
    old.update(status='superseded_unopened', reason=reason, superseded_at=pd.Timestamp.now(tz='UTC').isoformat())
    atomic_json(path, registry)


def aligned_origins(candles, protocol, end_exclusive):
    # Exclude later market data BEFORE feature/target construction.
    eligible = candles[candles.close_time < pd.Timestamp(end_exclusive)].copy()
    complete = (eligible.close_time - eligible.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1))
    excluded = [dict(open_time=row.open_time.isoformat(), close_time=row.close_time.isoformat())
                for row in eligible.loc[~complete, ['open_time', 'close_time']].itertuples()]
    frame = segmented_experiment_dataset(eligible[complete].copy())
    frame.attrs['excluded_nonstandard_hourly_candles'] = excluded
    anchor = pd.Timestamp(protocol['grid_anchor'])
    hours = (frame.close_time - anchor) / pd.Timedelta(hours=1)
    offsets = hours.round().astype('int64')
    valid = np.isclose(hours, offsets) & offsets.ge(0) & offsets.mod(STRIDE).eq(0)
    frame = frame[valid & (frame.target_end < pd.Timestamp(end_exclusive))].copy()
    if len(frame) < 200:
        raise ValueError('Insufficient common historical origins after warm-up and exact maturity filtering.')
    return frame.reset_index(drop=True)


def make_estimator(family, params):
    if family == 'LogisticRegression':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=42, **params))
    if family == 'RandomForest':
        return RandomForestClassifier(n_estimators=200, max_features=.8, n_jobs=2, random_state=42, **params)
    if family == 'ExtraTrees':
        return ExtraTreesClassifier(n_estimators=200, max_features=.8, n_jobs=2, random_state=42, **params)
    if family == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=20,
            early_stopping=False, random_state=42, **params)
    raise ValueError('Unknown family: ' + family)


def fit_candidate(family, params, train, calibration, features):
    estimator = make_estimator(family, params)
    estimator.fit(train[features], train.target)
    temperature = fit_temperature(probability_matrix(estimator, calibration[features]), calibration.target)
    return dict(estimator=estimator, temperature=temperature, features=features, family=family, params=params,
                training_prior=[float(train.target.eq(label).mean()) for label in OUTCOMES], sklearn_version=sklearn.__version__)


def rolling_blocks(data, folds, size):
    if len(data) - folds * size < 180:
        raise ValueError('Not enough prior data for nested walk-forward blocks.')
    for number in range(folds):
        start = len(data) - (folds - number) * size
        yield number + 1, data.iloc[start:start + size].copy()


def tune(data, family, features, protocol):
    """Only inner chronological validation labels can choose parameters."""
    trials = []
    for params in protocol['grids'][family]:
        all_y, all_p, records = [], [], []
        for number, validation in rolling_blocks(data, protocol['inner_folds'], protocol['inner_test_origins']):
            train, calibration = partitions(data, validation.close_time.min(), protocol['calibration_origins'])
            bundle = fit_candidate(family, params, train, calibration, features)
            p = predict_bundle(bundle, validation)
            all_y.extend(validation.target)
            all_p.append(p)
            records.append(dict(fold=number, **split_metadata(train, calibration, validation), metrics=metrics(validation.target, p)))
        score = metrics(all_y, np.concatenate(all_p))
        trials.append(dict(params=params, metrics=score, folds=records))
    best = max(trials, key=lambda trial: (trial['metrics']['macro_f1'], -trial['metrics']['brier_score']))
    return best['params'], trials


def score_rows(bundle, frame, train, calibration, fold=None):
    p = predict_bundle(bundle, frame)
    baselines = baseline_probabilities(bundle, frame)
    rows = pd.DataFrame(dict(available_at=frame.close_time.to_numpy(), target_end=frame.target_end.to_numpy(),
        true_outcome=frame.target.to_numpy(), true_return_24h=frame.future_return_24h.to_numpy(),
        prediction_computed_at=pd.Timestamp.now(tz='UTC').isoformat(), evaluation_type='retrospective_replay',
        train_end=train.close_time.max(), calibration_end=calibration.close_time.max(), fold=fold))
    for index, label in enumerate(OUTCOMES):
        rows['p_' + label] = p[:, index]
        for name, value in baselines.items():
            rows[name + '_' + label] = value[:, index]
    return rows


def rows_metrics(rows, prefix='p'):
    return metrics(rows.true_outcome, rows[[prefix + '_' + label for label in OUTCOMES]].values)


def nested_development(data, protocol, progress=None):
    candidates = {}
    for feature_name, features in protocol['feature_sets'].items():
        for family in protocol['grids']:
            name = family + '/' + feature_name
            fold_records, predictions = [], []
            for number, validation in rolling_blocks(data, protocol['outer_folds'], protocol['outer_test_origins']):
                train, calibration = partitions(data, validation.close_time.min(), protocol['calibration_origins'])
                params, trials = tune(train, family, features, protocol)
                bundle = fit_candidate(family, params, train, calibration, features)
                rows = score_rows(bundle, validation, train, calibration, number)
                predictions.append(rows)
                baselines = {key: rows_metrics(rows, key) for key in ('majority', 'training_prior', 'technical_rule')}
                fold_records.append(dict(fold=number, params=params, inner_trials=trials,
                    **split_metadata(train, calibration, validation), metrics=rows_metrics(rows), baselines=baselines,
                    train_metrics=metrics(train.target, predict_bundle(bundle, train)),
                    calibration_metrics=metrics(calibration.target, predict_bundle(bundle, calibration)),
                    uncalibrated_metrics=metrics(validation.target, predict_bundle(bundle, validation, calibrated=False)),
                    temperature=bundle['temperature']))
                if progress:
                    progress(name + ': outer fold ' + str(number) + ' complete')
            pooled = pd.concat(predictions, ignore_index=True)
            candidates[name] = dict(family=family, feature_set=feature_name, metrics=rows_metrics(pooled), folds=fold_records,
                baselines={key: rows_metrics(pooled, key) for key in ('majority', 'training_prior', 'technical_rule')},
                macro_f1_gain_vs_majority_by_fold=[row['metrics']['macro_f1'] - row['baselines']['majority']['macro_f1'] for row in fold_records],
                macro_f1_gain_vs_technical_by_fold=[row['metrics']['macro_f1'] - row['baselines']['technical_rule']['macro_f1'] for row in fold_records],
                predictions=pooled)
    selected = max(candidates, key=lambda name: (candidates[name]['metrics']['macro_f1'], -candidates[name]['metrics']['brier_score']))
    return selected, candidates


def paired_block_intervals(y, first, second, repeats=1000, block=7, seed=42):
    """Approximate paired circular moving-block intervals, not future confidence."""
    y = np.array([OUTCOMES.index(value) for value in y])
    first, second = np.asarray(first), np.asarray(second)
    n, rng = len(y), np.random.default_rng(seed)
    def stats(actual, predicted):
        cm = np.bincount(actual * 3 + predicted, minlength=9).reshape(3, 3)
        tp, support, called = cm.diagonal(), cm.sum(axis=1), cm.sum(axis=0)
        recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
        f1 = np.divide(2 * tp, support + called, out=np.zeros(3), where=(support + called) != 0)
        return np.array([np.mean(actual == predicted), recall.mean(), f1.mean()])
    differences = []
    for _ in range(repeats):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
        differences.append(stats(y[indices], first[indices]) - stats(y[indices], second[indices]))
    values = np.array(differences)
    point = stats(y, first) - stats(y, second)
    return {key: dict(difference=float(point[index]), lower=float(np.quantile(values[:, index], .025)),
                     upper=float(np.quantile(values[:, index], .975))) for index, key in enumerate(('accuracy', 'balanced_accuracy', 'macro_f1'))}


def subgroup_results(rows, frame, train):
    regime = causal_regimes(frame, train)
    merged = rows.merge(regime, left_on='available_at', right_on='close_time')
    merged['month'] = merged.available_at.dt.strftime('%Y-%m')
    result = {}
    for key in ('trend_regime', 'volatility_regime', 'combined_regime', 'month'):
        result[key] = [{key: str(value), 'small_sample': len(group) < 30, 'metrics': rows_metrics(group)}
                       for value, group in merged.groupby(key)]
    result['training_volatility_threshold'] = regime.attrs.get('training_volatility_median')
    result['interpretation'] = 'Post-hoc descriptive subgroups; small samples and multiple comparisons prevent firm regime claims.'
    return result


def verify_preservation(manifest, root):
    changed = [row['file'] for row in manifest if not (Path(root) / row['file']).is_file() or sha256(Path(root) / row['file']) != row['sha256']]
    if changed:
        raise ValueError('Protected production files changed: ' + ', '.join(changed))
    return dict(files_checked=len(manifest), changed=[])


def production_manifest(root):
    root = Path(root)
    paths = [root / filename for filename in ('app.py', 'trend_pipeline.py', 'forecast_validation.py', 'trend_metrics.py',
        'live_data.py', 'dashboard_data.py', 'market_intelligence.py', 'model.py', 'trend_news.py', 'train_trend_model.py')]
    for folder in ('data/trend', 'data/multisource', 'artifacts'):
        paths.extend(path for path in (root / folder).rglob('*') if path.is_file()
                     and 'backups' not in path.parts and 'trend_experiments' not in path.parts)
    paths.extend(path for path in (root / 'data').glob('*') if path.is_file())
    return [dict(file=str(path.relative_to(root)), sha256=sha256(path)) for path in sorted(set(paths)) if path.exists()]


def run_investigation(root=ROOT, offline=False, development_only=False, progress=print):
    root = Path(root)
    folder = root / 'data' / 'trend_experiments'
    protocol = protocol_definition()
    prior_exposure = check_unseen_reserve(root, protocol)
    identifier, entry = reserve_protocol(folder, protocol, development_only)
    run = root / 'artifacts' / 'trend_experiments' / identifier[:16]
    report_path = run / ('development_report.json' if development_only else 'report.json')
    if entry.get('status') == 'opened' and not development_only:
        raise ValueError('Reserve targets were already opened. Do not refit or rescore; inspect the frozen run for recovery.')
    if entry.get('status') == 'evaluated' and not development_only:
        if not report_path.exists():
            raise ValueError('The evaluated reserve report is missing; refusing a new evaluation.')
        report = json.loads(report_path.read_text())
        if sha256(report_path) != entry.get('report_sha256'):
            raise ValueError('Recorded final report changed.')
        verify_preservation(report['production_manifest'], root)
        return report
    if development_only:
        run = run / ('development_' + pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ'))
        report_path = run / 'development_report.json'
    run.mkdir(parents=True, exist_ok=True)
    protected = production_manifest(root)
    atomic_json(run / 'production_manifest.json', protected)
    atomic_json(run / 'protocol.json', dict(protocol=protocol, sha256=identifier,
        reserve_provenance=prior_exposure, frozen_before_collection=True))
    progress('Protocol frozen: ' + identifier[:16] + '. Production files protected: ' + str(len(protected)))
    raw = read_candles(folder / 'cache' / 'xrp_hourly.csv') if offline else collect_experiment_candles(folder / 'cache', days=1825)
    cache_sha256 = sha256(folder / 'cache' / 'xrp_hourly.csv')
    if raw.close_time.min() > pd.Timestamp(protocol['training_start']) - pd.Timedelta(days=7):
        raise ValueError('Insufficient early historical warm-up; no incomplete comparison is manufactured.')
    development = aligned_origins(raw, protocol, protocol['reserve_start'])
    save_frame(run / 'development_dataset.csv', development)
    train, calibration = partitions(development, pd.Timestamp(protocol['reserve_start']), protocol['calibration_origins'])
    with threadpool_limits(limits=2):
        selected, candidates = nested_development(train, protocol, progress)
        chosen = candidates[selected]
        features = protocol['feature_sets'][chosen['feature_set']]
        params, final_trials = tune(train, chosen['family'], features, protocol)
        # Final fit cutoff is fixed by protocol, without looking at final origins/labels.
        bundle = fit_candidate(chosen['family'], params, train, calibration, features)
    bundle.update(protocol_sha256=identifier, experimental_only=True, selected_group=selected)
    model_path = run / ('development_model.joblib' if development_only else 'model.joblib')
    joblib.dump(bundle, model_path)
    selection = dict(group=selected, params=params, features=features, model_sha256=sha256(model_path),
        protocol_sha256=identifier, final_trials=final_trials, training_end=train.close_time.max().isoformat(),
        calibration_end=calibration.close_time.max().isoformat(), frozen_at=pd.Timestamp.now(tz='UTC').isoformat())
    selection_path = run / ('development_selection.json' if development_only else 'selection.json')
    atomic_json(selection_path, selection)
    for name, candidate in candidates.items():
        save_frame(run / ('outer_' + name.replace('/', '_') + '.csv'), candidate.pop('predictions'))
    report = dict(protocol_sha256=identifier, protocol=protocol, selected_group=selected, selected_params=params,
        analysis_started_at=selection['frozen_at'], evaluation_type='retrospective_nested_walk_forward',
        development_samples=len(development), selection_samples=len(train), candidates=candidates, selected_before_final_reserve=True,
        data=dict(raw_cache_sha256=cache_sha256, development_sha256=sha256(run / 'development_dataset.csv'),
                  downloaded_rows=len(raw), start=raw.close_time.min().isoformat(), end=raw.close_time.max().isoformat(),
                  development_gap_handling=development.attrs),
        final_reserve_scored=False, production_manifest=protected, historical_reserve_provenance=prior_exposure,
        old_model_diagnosis=diagnose_existing(root), limitations=[
            'The older 2024 reserve was newly collected and excluded from prior production snapshots; it is not prospective evidence for 2026.',
            'Model-design hypotheses were motivated by already exposed later-period diagnostics; no later candles enter fitting or features.',
            'One historical reserve cannot prove stable performance across market cycles; future untouched periods are needed.',
            'Outer validation selects the family/feature group, so its winning score has selection uncertainty; only final scoring is independent of that selection.',
            'Regime and monthly slices are descriptive post-hoc analyses with small samples.',
            'Historical candle revisions, data latency, costs and slippage are not modeled.',
            'No news features were added because prior timestamp coverage was insufficient.'])
    quality_path = folder / 'cache' / 'data_quality.json'
    report['data_quality'] = json.loads(quality_path.read_text()) if quality_path.exists() else {'source_gaps': 'not available in offline cache metadata'}
    if not development_only:
        if sha256(folder / 'cache' / 'xrp_hourly.csv') != cache_sha256:
            raise ValueError('Market cache changed during selection; reserve remains unopened.')
        save_frame(run / 'market_snapshot.csv', raw[raw.close_time < pd.Timestamp(protocol['reserve_end_exclusive'])])
        update_reserve(folder, identifier, status='opened', selection_spec_sha256=digest({key: selection[key] for key in ('group', 'params', 'features')}),
                       model_sha256=sha256(model_path), raw_cache_sha256=cache_sha256,
                       development_sha256=sha256(run / 'development_dataset.csv'), opened_at=pd.Timestamp.now(tz='UTC').isoformat())
        # First access to reserved target labels occurs AFTER selection/model freeze.
        full = aligned_origins(raw, protocol, protocol['reserve_end_exclusive'])
        final = full[full.close_time >= pd.Timestamp(protocol['reserve_start'])].copy()
        if final.empty or train.target_end.max() >= calibration.close_time.min() or calibration.target_end.max() >= final.close_time.min():
            raise ValueError('Final reserve or purging is invalid.')
        rows = score_rows(bundle, final, train, calibration)
        save_frame(run / 'final_predictions.csv', rows)
        save_frame(run / 'final_dataset.csv', final)
        base = {name: rows_metrics(rows, name) for name in ('majority', 'training_prior', 'technical_rule')}
        final_metrics = rows_metrics(rows)
        chosen_score = chosen['metrics']
        quality = quality_checks(final_metrics, base, chosen_score['macro_f1'] > chosen['baselines']['majority']['macro_f1'])
        paired = {name: paired_block_intervals(rows.true_outcome, rows[['p_' + label for label in OUTCOMES]].values.argmax(axis=1),
            rows[[name + '_' + label for label in OUTCOMES]].values.argmax(axis=1)) for name in ('majority', 'technical_rule')}
        consistent = (sum(value > 0 for value in chosen['macro_f1_gain_vs_majority_by_fold']) >= 2 and
                      sum(value > 0 for value in chosen['macro_f1_gain_vs_technical_by_fold']) >= 2)
        robust = quality['passed'] and consistent and paired['majority']['balanced_accuracy']['lower'] > 0 and paired['technical_rule']['macro_f1']['lower'] > 0
        report.update(final_reserve_scored=True, final_metrics=final_metrics, final_baselines=base,
            final_split=split_metadata(train, calibration, final), selected_training_metrics=metrics(train.target, predict_bundle(bundle, train)),
            selected_calibration_metrics=metrics(calibration.target, predict_bundle(bundle, calibration)),
            final_uncalibrated_metrics=metrics(final.target, predict_bundle(bundle, final, calibrated=False)),
            final_regime_results=subgroup_results(rows, final, train), paired_block_intervals=paired, quality=quality,
            consistency=dict(positive_macro_f1_vs_majority_folds=sum(value > 0 for value in chosen['macro_f1_gain_vs_majority_by_fold']),
                positive_macro_f1_vs_technical_folds=sum(value > 0 for value in chosen['macro_f1_gain_vs_technical_by_fold']),
                minimum_two_of_three_passed=consistent, robust_improvement_demonstrated=bool(robust)),
            production_replaced=False, promotion_decision='No automatic replacement. A current prospective evaluation is still needed.')
        report['artifacts'] = dict(model_sha256=sha256(model_path), selection_sha256=sha256(selection_path),
            market_sha256=sha256(run / 'market_snapshot.csv'), final_predictions_sha256=sha256(run / 'final_predictions.csv'),
            final_dataset_sha256=sha256(run / 'final_dataset.csv'))
    report['evaluated_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    report['preservation'] = verify_preservation(protected, root)
    atomic_json(report_path, report)
    if not development_only:
        update_reserve(folder, identifier, status='evaluated', report_sha256=sha256(report_path), result_path=str(report_path.relative_to(root)))
    return report
