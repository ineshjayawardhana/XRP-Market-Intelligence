"""Read-only, descriptive diagnosis of the already-examined trend model run.

This does not create a new validation claim, select a replacement, or rewrite
production artifacts. All subgroup/feature observations are exploratory.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast_validation import validated_research
from trend_metrics import OUTCOMES, metrics
from trend_pipeline import FEATURES, fit_bundle, predict_bundle

ROOT = Path(__file__).resolve().parent


def class_distribution(values):
    values = pd.Series(values)
    return {label: dict(count=int(values.eq(label).sum()),
                        fraction=float(values.eq(label).mean()) if len(values) else None)
            for label in OUTCOMES}


def wilson_interval(correct, sample_size, z=1.959963984540054):
    """Nominal binomial interval; serial dependence can reduce effective n."""
    if sample_size <= 0 or not 0 <= correct <= sample_size:
        raise ValueError('A positive sample size and valid correct count are required.')
    observed = correct / sample_size
    denominator = 1 + z * z / sample_size
    center = (observed + z * z / (2 * sample_size)) / denominator
    width = z * np.sqrt(observed * (1 - observed) / sample_size + z * z / (4 * sample_size ** 2)) / denominator
    return dict(lower=float(max(0, center - width)), upper=float(min(1, center + width)),
                assumption='Nominal binomial interval; dependent market regimes may widen uncertainty.')


def paired_accuracy_interval(y, model_predictions, baseline_predictions, block_size=7, repeats=2000, seed=42):
    """Moving-block bootstrap of paired accuracy difference, not a significance test."""
    y, predicted, baseline = np.asarray(y), np.asarray(model_predictions), np.asarray(baseline_predictions)
    if not len(y) or y.shape != predicted.shape or y.shape != baseline.shape or repeats < 2 or block_size < 1:
        raise ValueError('Paired nonempty outcomes and valid bootstrap settings are required.')
    difference = (predicted == y).astype(float) - (baseline == y).astype(float)
    size = min(block_size, len(y))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(y) - size + 1, size=(repeats, int(np.ceil(len(y) / size))))
    indices = (starts[:, :, None] + np.arange(size)).reshape(repeats, -1)[:, :len(y)]
    sampled = difference[indices].mean(axis=1)
    return dict(accuracy_difference=float(difference.mean()), lower=float(np.quantile(sampled, .025)),
                upper=float(np.quantile(sampled, .975)), block_origins=size, repeats=repeats,
                limitation='Exploratory paired moving-block percentile interval; block choice and regime dependence remain uncertain.')


def feature_diagnostics(train, test, features=FEATURES):
    """Training-fitted scale/correlation and heldout covariate descriptions only."""
    training = train[features].astype(float)
    testing = test[features].astype(float)
    mean, scale = training.mean(), training.std(ddof=0)
    shifts = ((testing.mean() - mean) / scale.replace(0, np.nan)).abs()
    ratios = testing.std(ddof=0) / scale.replace(0, np.nan)
    rows = []
    for feature in features:
        row = dict(feature=feature, training_mean=float(mean[feature]),
                   training_std=float(scale[feature]), training_unique_values=int(training[feature].nunique()),
                   test_mean=float(testing[feature].mean()),
                   absolute_mean_shift_in_training_std=float(shifts[feature]) if np.isfinite(shifts[feature]) else None,
                   test_to_training_std_ratio=float(ratios[feature]) if np.isfinite(ratios[feature]) else None)
        row['shift_flag'] = bool(row['absolute_mean_shift_in_training_std'] is not None and
                                 row['absolute_mean_shift_in_training_std'] >= .5)
        rows.append(row)
    correlations = training.corr()
    redundant = [dict(first=left, second=right, correlation=float(correlations.loc[left, right]))
                 for index, left in enumerate(features) for right in features[index + 1:]
                 if abs(correlations.loc[left, right]) >= .95]
    relevance = []
    if 'future_return_24h' in train:
        for feature in features:
            correlation = training[feature].corr(train.future_return_24h)
            relevance.append(dict(feature=feature, training_return_correlation=float(correlation) if np.isfinite(correlation) else None))
        relevance.sort(key=lambda row: abs(row['training_return_correlation'] or 0), reverse=True)
    return dict(covariate_shift=rows, mean_shift_flags=[row['feature'] for row in rows if row['shift_flag']],
                constant_training_features=[row['feature'] for row in rows if row['training_std'] == 0],
                highly_correlated_training_pairs=redundant, training_return_correlations=relevance,
                interpretation='Redundancy and marginal correlations are descriptive, not stable feature importance or proof of causality. Shift threshold 0.5 is a diagnostic heuristic.')


def causal_regime_labels(train, test):
    """Classify conditions at the prediction origin, never by its future outcome."""
    low, high = train.volatility_24h.quantile([1 / 3, 2 / 3]).tolist()
    trend_threshold = float(train.ema_gap.abs().quantile(1 / 3))
    volatility = np.select([test.volatility_24h < low, test.volatility_24h > high], ['Low volatility', 'High volatility'], default='Middle volatility')
    trend = np.select([test.ema_gap > trend_threshold, test.ema_gap < -trend_threshold], ['Upward EMA trend', 'Downward EMA trend'], default='Small EMA gap')
    return dict(volatility=np.asarray(volatility), ema_trend=np.asarray(trend)), dict(
        volatility_low=float(low), volatility_high=float(high), ema_gap_absolute_threshold=trend_threshold,
        threshold_fit='Training observations only; fixed before describing heldout conditions.')


def _metric_slice(y, probabilities):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        measured = metrics(y, probabilities)
    measured['all_outcomes_present'] = all(measured['outcome_distribution'][label] > 0 for label in OUTCOMES)
    measured['small_subgroup'] = len(y) < 30
    return measured


def regime_analysis(train, test, probabilities):
    labels, thresholds = causal_regime_labels(train, test)
    training_labels, _ = causal_regime_labels(train, train)
    baselines = {}
    if 'target' in train:
        prior = np.array([train.target.eq(label).mean() for label in OUTCOMES])
        majority = np.zeros_like(probabilities)
        majority[:, prior.argmax()] = 1.
        baselines['training_majority'] = majority
        baselines['training_prior'] = np.tile(prior, (len(test), 1))
    if 'technical' in test:
        rule = test.technical.map({'Bullish': 'UP', 'Bearish': 'DOWN', 'Neutral': 'NEUTRAL'})
        baselines['technical_rule'] = np.array([[float(label == value) for label in OUTCOMES] for value in rule])
    def describe(mask):
        return dict(**_metric_slice(test.target.to_numpy()[mask], probabilities[mask]),
                    baselines={name: _metric_slice(test.target.to_numpy()[mask], values[mask]) for name, values in baselines.items()})
    results = {}
    for condition, values in labels.items():
        results[condition] = []
        for value in sorted(set(values)):
            mask = values == value
            results[condition].append(dict(regime=value, **describe(mask)))
    months = test.close_time.dt.strftime('%Y-%m')
    monthly = []
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy()
        monthly.append(dict(month=month, **describe(mask)))
    return dict(training_thresholds=thresholds, by_condition=results, by_utc_month=monthly,
                training_regime_counts={name: {label: int((values == label).sum()) for label in sorted(set(values))} for name, values in training_labels.items()},
                heldout_regime_counts={name: {label: int((values == label).sum()) for label in sorted(set(values))} for name, values in labels.items()},
                limitation='Post-hoc descriptive subgroups. Small samples and missing outcomes make subgroup balanced accuracy unstable; no multiple-comparison adjustment or regime-specific strategy claim.')


def diagnose_existing(root=ROOT, reconstruct_random_forest=True):
    """Inspect frozen artifacts without writing to data, models, dashboard or reports."""
    root = Path(root)
    artifacts = root / 'artifacts'
    verified = validated_research(artifacts)
    if not verified.get('integrity_passed'):
        raise ValueError('Cannot diagnose unverified artifacts: ' + verified['limitation'])
    contract = json.loads((artifacts / 'trend_24h_validation.json').read_text(encoding='utf-8'))
    report = json.loads((artifacts / contract['report_file']).read_text(encoding='utf-8'))
    bundle = joblib.load(artifacts / contract['model_file'])
    data = pd.read_csv(artifacts / contract['dataset_file'])
    data['close_time'] = pd.to_datetime(data.close_time, utc=True)
    data['target_end'] = pd.to_datetime(data.target_end, utc=True)
    split = report['split']
    subsets = {name: data[data.close_time.between(pd.Timestamp(split[name + '_start']), pd.Timestamp(split[name + '_end']))].copy()
               for name in ('train', 'calibration', 'test')}
    train, calibration, test = [subsets[key] for key in ('train', 'calibration', 'test')]
    all_metrics, metadata = {}, {}
    models = {report['selected_model']: bundle}
    if reconstruct_random_forest and 'RandomForest' in report['candidates'] and report['selected_model'] != 'RandomForest':
        rebuilt = fit_bundle('RandomForest', train, calibration, bundle['features'])
        heldout = metrics(test.target, predict_bundle(rebuilt, test))
        expected = report['candidates']['RandomForest']['holdout']
        matched = (heldout['confusion_matrix'] == expected['confusion_matrix'] and
                   all(np.isclose(heldout[key], expected[key], atol=1e-9) for key in ('accuracy', 'macro_f1', 'brier_score', 'log_loss')))
        if matched:
            models['RandomForest'] = rebuilt
        metadata['RandomForest'] = dict(reconstructed_for_diagnosis=True, matches_recorded_holdout=bool(matched),
                                       note='The old run saved only its selected model. This RF is reconstructed from original fixed settings and frozen partitions; it is never saved or promoted.')
    for name, estimator in models.items():
        all_metrics[name] = {key: metrics(frame.target, predict_bundle(estimator, frame)) for key, frame in subsets.items()}
        all_metrics[name]['temperature'] = estimator['temperature']
        all_metrics[name]['test_uncalibrated'] = metrics(test.target, predict_bundle(estimator, test, calibrated=False))
    probabilities = predict_bundle(bundle, test)
    predicted = np.array(OUTCOMES)[probabilities.argmax(axis=1)]
    majority = OUTCOMES[int(np.argmax(bundle['training_prior']))]
    baseline = np.full(len(test), majority)
    correct = int((predicted == test.target.to_numpy()).sum())
    feature_report = feature_diagnostics(train, test, bundle['features'])
    gap = all_metrics[report['selected_model']]['train']['macro_f1'] - all_metrics[report['selected_model']]['test']['macro_f1']
    conclusions = [
        'The previously reported 175-example test has already been examined. It is diagnostic history, not an untouched test for new tuning.',
        'Class prevalence changes across partitions; fixed +/-1% targets can have different distributions in quieter or more volatile periods.',
        f'The selected model has a train-to-test Macro-F1 gap of {gap:.4f}. This is compatible with weak generalization or distribution shift; it does not prove a single cause.',
        'Temperature scaling changes probability sharpness while preserving class ranking; better ECE/log loss cannot repair low UP recall.',
        'Causal features and purged chronological partitions reproduce from frozen artifacts; this audit found no implemented target-overlap leakage, but does not prove absence of every data-vintage or research-design bias.',
        'Many indicators summarize the same price history. Correlated features and weak marginal associations are reasons to test regularized, parsimonious alternatives within development only.',
        'Regime/month results and diagnostic feature shifts are post-hoc descriptions, not evidence for a deployable conditional strategy.',
    ]
    if 'RandomForest' in all_metrics:
        rf_gap = all_metrics['RandomForest']['train']['macro_f1'] - all_metrics['RandomForest']['test']['macro_f1']
        conclusions.append(f'Reconstructed Random Forest train-to-test Macro-F1 gap is {rf_gap:.4f}; a large positive gap is consistent with overfitting, subject to distribution-shift and small-sample limitations.')
    return dict(schema_version=1, diagnostic_scope='Already-examined production-run history; descriptive only',
        inspected_at=pd.Timestamp.now(tz='UTC').isoformat(), model_version=report['model_version'],
        artifact_integrity_passed=True, partition_class_distributions={name: class_distribution(frame.target) for name, frame in subsets.items()},
        model_partition_metrics=all_metrics, reconstruction=metadata,
        recorded_candidate_holdout={name: candidate['holdout'] for name, candidate in report['candidates'].items()},
        recorded_walk_forward={name: candidate['walk_forward'] for name, candidate in report['candidates'].items()},
        recorded_baselines=report['baselines'], features=feature_report,
        regime_performance=regime_analysis(train, test, probabilities),
        uncertainty=dict(nominal_accuracy_wilson_95=wilson_interval(correct, len(test)),
                         paired_majority_accuracy_difference_95=paired_accuracy_interval(test.target, predicted, baseline)),
        original_quality=report['quality'], conclusions=conclusions)


if __name__ == '__main__':
    print(json.dumps(diagnose_existing(), indent=2))
