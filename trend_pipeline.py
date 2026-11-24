"""Causal 24h market-direction research. No news-model fitting or trading.

Historical OOS predictions are retrospective replays computed at the actual run
time. Their simulated information cutoffs are never presented as live predictions.
"""
from __future__ import annotations

import json
import hashlib
import inspect
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from live_data import calculate_indicators
from market_intelligence import candle_quality, freshness
from news_label_safety import atomic_json, sha256
from trend_market_data import collect_candles, read_candles, save_frame
from trend_metrics import OUTCOMES, metrics, quality_checks
from trend_news import audit_news, news_features_asof

ROOT = Path(__file__).resolve().parent
STRIDE = 25  # Rotates through hourly origins, with disjoint 24h target windows.
FEATURES = ['rsi', 'ema20_distance', 'ema50_distance', 'ema_gap', 'macd_relative',
    'macd_signal_relative', 'macd_hist_relative', 'volume_change_1h', 'volume_change_24h',
    'volume_relative_20h', 'return_1h', 'return_6h', 'return_12h', 'return_24h',
    'return_72h', 'volatility_24h', 'volatility_72h', 'range_relative', 'body_relative',
    'hour_sin', 'hour_cos']
LIMITATIONS = [
    'Retrospective chronological replay, not a prospective live track record.',
    'Fixed model families selected on development folds only; the final holdout was not used for selection.',
    '25-hour-spaced origins have disjoint target windows but market regimes can remain dependent.',
    'Historical calibration can drift. Policy checks do not prove future accuracy or profitability.',
    'Targets use hourly close-to-close returns; latency, trading costs and slippage are not modeled.',
    'Historical Binance candles were downloaded later; original contemporaneous data vintages are not available.',
]


def feature_frame(candles, now=None):
    issues = candle_quality(candles, now, minimum=100)
    if issues:
        raise ValueError('Invalid research candles: ' + ' '.join(issues))
    duration = candles.close_time - candles.open_time
    if not duration.eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).all():
        raise ValueError('Candles must be complete one-hour Binance intervals.')
    if ((candles.high < candles[['open', 'close', 'low']].max(axis=1)).any()
            or (candles.low > candles[['open', 'close', 'high']].min(axis=1)).any()):
        raise ValueError('OHLC prices are internally inconsistent.')
    data = calculate_indicators(candles)
    close = data.close
    data['ema20_distance'] = close / data.ema20 - 1
    data['ema50_distance'] = close / data.ema50 - 1
    data['ema_gap'] = (data.ema20 - data.ema50) / close
    data['macd_relative'] = data.macd / close
    data['macd_signal_relative'] = data.macd_signal / close
    data['macd_hist_relative'] = (data.macd - data.macd_signal) / close
    for hours in (1, 6, 12, 24, 72):
        data[f'return_{hours}h'] = close.pct_change(hours, fill_method=None)
    for hours in (1, 24):
        previous = data.volume.shift(hours)
        # Zero-volume denominators are missing evidence, never forward/back-filled.
        data[f'volume_change_{hours}h'] = data.volume / previous.replace(0, np.nan) - 1
    data['volume_relative_20h'] = data.volume / data.volume.shift(1).rolling(20).mean().replace(0, np.nan)
    for hours in (24, 72):
        data[f'volatility_{hours}h'] = data.return_1h.rolling(hours).std()
    data['range_relative'] = (data.high - data.low) / close
    data['body_relative'] = (close - data.open) / data.open
    hour = data.close_time.dt.hour
    data['hour_sin'], data['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    return data


def build_dataset(candles, now=None):
    data = feature_frame(candles, now)
    data['target_end'] = data.close_time + pd.Timedelta(hours=24)
    # Timestamp lookup rather than an unchecked row shift.
    prices = data.set_index('close_time').close
    data['future_close'] = data.target_end.map(prices)
    data['future_return_24h'] = data.future_close / data.close - 1
    data['target'] = np.select([data.future_return_24h >= .01, data.future_return_24h <= -.01],
                               ['UP', 'DOWN'], default='NEUTRAL')
    return data.dropna(subset=FEATURES + ['future_close']).reset_index(drop=True)


def feature_signature():
    source = '\n'.join([inspect.getsource(function) for function in (feature_frame, build_dataset, calculate_indicators)])
    return hashlib.sha256((source + repr(FEATURES) + str(STRIDE)).encode()).hexdigest()


def probability_matrix(estimator, x):
    values = estimator.predict_proba(x)
    return values[:, [list(estimator.classes_).index(label) for label in OUTCOMES]]


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, y):
    indices = np.array([OUTCOMES.index(label) for label in y])
    def objective(log_temperature):
        p = temperature_scale(probabilities, np.exp(log_temperature))
        return float(-np.log(np.clip(p[np.arange(len(y)), indices], 1e-12, 1)).mean())
    result = minimize_scalar(objective, bounds=(-3., 3.), method='bounded')
    if not result.success:
        raise ValueError('Calibration optimization failed.')
    return float(np.exp(result.x))


def partitions(data, test_start, calibration_samples=45):
    """Strict maturity purging at both boundaries, based on actual timestamps."""
    known = data[data.target_end < test_start]
    calibration = known.tail(calibration_samples).copy()
    if len(calibration) < 30:
        raise ValueError('At least 30 non-overlapping calibration observations are required.')
    train = known[known.target_end < calibration.close_time.min()].copy()
    if len(train) < 90 or not set(OUTCOMES).issubset(set(train.target)):
        raise ValueError('Insufficient chronological training observations or outcome classes.')
    return train, calibration


def fit_bundle(family, train, calibration, features):
    estimator = (make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=1500, random_state=42))
        if family == 'LogisticRegression' else RandomForestClassifier(n_estimators=200, max_depth=6,
            min_samples_leaf=12, max_features=.8, n_jobs=2, random_state=42))
    estimator.fit(train[features], train.target)
    temperature = fit_temperature(probability_matrix(estimator, calibration[features]), calibration.target)
    prior = np.array([(train.target == label).mean() for label in OUTCOMES])
    return dict(estimator=estimator, temperature=temperature, features=features, training_prior=prior.tolist(),
                family=family, sklearn_version=sklearn.__version__)


def predict_bundle(bundle, data, calibrated=True):
    p = probability_matrix(bundle['estimator'], data[bundle['features']])
    return temperature_scale(p, bundle['temperature']) if calibrated else p


def baseline_probabilities(bundle, test):
    prior = np.array(bundle['training_prior'])
    majority = np.zeros((len(test), 3))
    majority[:, prior.argmax()] = 1.
    rule = test.technical.map({'Bullish': 'UP', 'Bearish': 'DOWN', 'Neutral': 'NEUTRAL'})
    technical = np.array([[float(label == value) for label in OUTCOMES] for value in rule])
    return dict(majority=majority, training_prior=np.tile(prior, (len(test), 1)), technical_rule=technical)


def split_metadata(train, calibration, test):
    result = dict(purge_hours=24, evaluation_stride_hours=STRIDE)
    for name, frame in [('train', train), ('calibration', calibration), ('test', test)]:
        result.update({name + '_start': frame.close_time.min().isoformat(), name + '_end': frame.close_time.max().isoformat(),
                       name + '_samples': len(frame)})
    result['train_target_end'] = train.target_end.max().isoformat()
    result['calibration_target_end'] = calibration.target_end.max().isoformat()
    return result


def replay_rows(bundle, test, computed_at, train, calibration):
    p = predict_bundle(bundle, test)
    frame = pd.DataFrame(dict(available_at=test.close_time.to_numpy(), target_end=test.target_end.to_numpy(),
        prediction_computed_at=computed_at, evaluation_type='retrospective_replay',
        simulated_train_end=train.close_time.max(), simulated_calibration_end=calibration.close_time.max(),
        true_outcome=test.target.to_numpy(), true_return_24h=test.future_return_24h.to_numpy(),
        technical_outcome=test.technical.map({'Bullish': 'UP', 'Bearish': 'DOWN', 'Neutral': 'NEUTRAL'}).to_numpy()))
    for index, label in enumerate(OUTCOMES):
        frame['p_' + label] = p[:, index]
        frame['prior_' + label] = bundle['training_prior'][index]
    return frame


def train_research(root=ROOT, days=730, offline=False, candles=None):
    root = Path(root)
    raw_path = root / 'data' / 'trend' / 'xrp_hourly.csv'
    raw = candles if candles is not None else read_candles(raw_path) if offline else collect_candles(raw_path.parent, days=days)
    current = pd.Timestamp.now(tz='UTC')
    data = build_dataset(raw, current)
    # Trim the seven-day feature warmup from supervised research origins.
    data = data[data.close_time >= raw.close_time.max() - pd.Timedelta(days=days)].reset_index(drop=True)
    span = (raw.close_time.max() - raw.close_time.min()).total_seconds() / 86400
    if span < 180:
        raise ValueError('At least six months (180 days) of contiguous hourly data are required.')
    sampled = data.iloc[::STRIDE].copy().reset_index(drop=True)
    if len(sampled) < 170:
        raise ValueError('Not enough matured, non-overlapping observations for train/calibration/walk-forward/holdout research.')
    test_count = max(30, int(len(sampled) * .25))
    development, holdout = sampled.iloc[:-test_count].copy(), sampled.iloc[-test_count:].copy()
    # Keep the development origins whose outcomes mature strictly before final holdout.
    development = development[development.target_end < holdout.close_time.min()].copy()
    fold_size = min(30, max(10, (len(development) - 140) // 3))
    if fold_size < 10:
        raise ValueError('Insufficient development history for three purged walk-forward folds.')
    news = audit_news(root, sampled.close_time)
    feature_sets = {'Technical': FEATURES}
    news_inputs = {}
    if news['ready']:
        for relative in ('data/multisource/xrp_real_labeled.csv', 'data/multisource/xrp_human_reviews.csv',
                         'data/multisource/research/gold_dataset.csv', 'data/multisource/research/silver_dataset.csv'):
            path = root / relative
            if path.exists():
                news_inputs[relative] = path.read_bytes()
        extra = news_features_asof(root, sampled.close_time)
        columns = [column for column in extra if column != 'prediction_time']
        sampled = sampled.merge(extra, left_on='close_time', right_on='prediction_time', how='left')
        development = sampled[sampled.close_time.isin(development.close_time)].copy()
        holdout = sampled[sampled.close_time.isin(holdout.close_time)].copy()
        feature_sets['Technical+News'] = FEATURES + columns
    research_started_at = pd.Timestamp.now(tz='UTC').isoformat()
    candidates, definitions = {}, {}
    for feature_name, features in feature_sets.items():
        for family in ('LogisticRegression', 'RandomForest'):
            name = family if feature_name == 'Technical' else family + '+News'
            folds, replays, baseline_parts = [], [], {key: [] for key in ('majority', 'training_prior', 'technical_rule')}
            for fold in range(3):
                start = len(development) - (3 - fold) * fold_size
                test = development.iloc[start:start + fold_size]
                train, calibration = partitions(development, test.close_time.min())
                bundle = fit_bundle(family, train, calibration, features)
                rows = replay_rows(bundle, test, pd.Timestamp.now(tz='UTC').isoformat(), train, calibration)
                rows['fold'] = fold + 1
                replays.append(rows)
                base = baseline_probabilities(bundle, test)
                for key in base:
                    baseline_parts[key].append(base[key])
                folds.append(dict(fold=fold + 1, **split_metadata(train, calibration, test),
                    metrics=metrics(test.target, predict_bundle(bundle, test)),
                    baselines={key: metrics(test.target, value) for key, value in base.items()}))
            pooled = pd.concat(replays, ignore_index=True)
            aggregate = metrics(pooled.true_outcome, pooled[['p_' + label for label in OUTCOMES]].values)
            aggregate['folds'] = folds
            aggregate['baselines'] = {key: metrics(pooled.true_outcome, np.concatenate(value)) for key, value in baseline_parts.items()}
            candidates[name] = dict(walk_forward=aggregate)
            definitions[name] = (family, features, pooled)
    # Frozen selection before accessing holdout scores. No hyperparameter tuning on holdout.
    selected = max(candidates, key=lambda name: (candidates[name]['walk_forward']['macro_f1'], -candidates[name]['walk_forward']['brier_score']))
    final_train, final_calibration = partitions(development, holdout.close_time.min())
    bundles = {}
    for name, (family, features, _) in definitions.items():
        bundle = fit_bundle(family, final_train, final_calibration, features)
        candidates[name]['holdout'] = metrics(holdout.target, predict_bundle(bundle, holdout))
        candidates[name]['holdout_uncalibrated'] = metrics(holdout.target, predict_bundle(bundle, holdout, calibrated=False))
        candidates[name]['temperature'] = bundle['temperature']
        bundles[name] = bundle
    bundle = bundles[selected]
    baseline_p = baseline_probabilities(bundle, holdout)
    baselines = {key: metrics(holdout.target, value) for key, value in baseline_p.items()}
    selected_metrics = candidates[selected]['holdout']
    wf = candidates[selected]['walk_forward']
    quality = quality_checks(selected_metrics, baselines, wf['macro_f1'] > wf['baselines']['majority']['macro_f1'])
    version = 'xrp24h_' + pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    run = root / 'artifacts' / 'trend_runs' / version
    run.mkdir(parents=True, exist_ok=False)
    for relative, content in news_inputs.items():
        if (root / relative).read_bytes() != content:
            raise ValueError('News provenance changed during training; rerun with consistent inputs.')
        copy = run / 'news_snapshot' / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(content)
    bundle.update(model_version=version, selected_model=selected, evaluation_type='retrospective_replay',
                  uses_news='+News' in selected, feature_signature=feature_signature())
    joblib.dump(bundle, run / 'model.joblib')
    save_frame(run / 'market_snapshot.csv', raw)
    save_frame(run / 'hourly_dataset.csv', data)
    save_frame(run / 'supervised_origins.csv', sampled)
    oos = replay_rows(bundle, holdout, pd.Timestamp.now(tz='UTC').isoformat(), final_train, final_calibration)
    save_frame(run / 'holdout_predictions.csv', oos)
    save_frame(run / 'development_predictions.csv', definitions[selected][2])
    split = split_metadata(final_train, final_calibration, holdout)
    evaluated_at = pd.Timestamp.now(tz='UTC').isoformat()
    report = dict(schema_version=2, model_version=version, evaluated_at=evaluated_at,
        research_started_at=research_started_at, feature_signature=feature_signature(),
        evaluation_type='retrospective_replay', selected_model=selected, selected_before_holdout=True,
        features=bundle['features'], data=dict(hourly_candles=len(raw), usable_hourly_rows=len(data),
            daily_samples=len(sampled), nonoverlapping_samples=len(sampled), start=raw.close_time.min().isoformat(),
            end=raw.close_time.max().isoformat(), source='Binance XRPUSDT spot public 1h klines', sha256=sha256(run / 'market_snapshot.csv')),
        split=split, candidates=candidates, baselines=baselines, holdout=selected_metrics,
        quality=quality, news=news, limitations=LIMITATIONS,
        policy=dict(minimum_holdout=100, minimum_per_class=10, balanced_accuracy_floor=.38, maximum_ece=.10,
                    skill_baseline='training-period class prior; majority and existing technical rules on identical holdout origins'),
        software=dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__))
    atomic_json(run / 'report.json', report)
    prefix = 'trend_runs/' + version + '/'
    contract = dict(schema_version=2, pipeline='xrp_chronological_trend_v2', task='XRP_24H_DIRECTION',
        horizon_hours=24, labels=OUTCOMES, model_version=version, model_file=prefix + 'model.joblib',
        model_sha256=sha256(run / 'model.joblib'), oos_predictions_file=prefix + 'holdout_predictions.csv',
        oos_predictions_sha256=sha256(run / 'holdout_predictions.csv'), report_file=prefix + 'report.json',
        report_sha256=sha256(run / 'report.json'), dataset_file=prefix + 'supervised_origins.csv',
        dataset_sha256=sha256(run / 'supervised_origins.csv'), market_file=prefix + 'market_snapshot.csv',
        market_sha256=sha256(run / 'market_snapshot.csv'), evaluation_type='retrospective_replay',
        feature_signature=feature_signature(),
        news_snapshot_files=[dict(file=prefix + 'news_snapshot/' + relative,
                            sha256=hashlib.sha256(content).hexdigest()) for relative, content in news_inputs.items()],
        evaluated_at=evaluated_at, test_samples=len(holdout), train_end=split['train_end'],
        calibration_fit_end=split['calibration_end'], test_start=split['test_start'], test_end=split['test_end'],
        chronological_out_of_sample_passed=True, calibration_passed=quality['checks']['calibration_ece_at_most_0_10'],
        quality_passed=quality['passed'])
    atomic_json(run / 'validation.json', contract)
    atomic_json(root / 'artifacts' / 'trend_24h_report.json', report)
    atomic_json(root / 'artifacts' / 'trend_24h_validation.json', contract)
    return report


def live_forecast(candles, root=ROOT):
    """Inference only. Fresh completed input and verified local artifacts required."""
    from forecast_validation import validated_forecast, validated_research
    result = dict(validated=False, probabilities=None, experimental_label=None,
        message='Predictive probabilities are not yet validated.', limitation='No evaluated trend model is available.',
        model_version=None, feature_as_of=None, generated_at=None, evaluated_at=None,
        sample_size=None, test_start=None, test_end=None, quality_failures=[])
    artifacts = Path(root) / 'artifacts'
    evidence = validated_research(artifacts)
    for key in ('model_version', 'evaluated_at', 'sample_size', 'test_start', 'test_end', 'quality_failures'):
        if key in evidence:
            result[key] = evidence[key]
    if not evidence.get('integrity_passed'):
        result['limitation'] = evidence['limitation']
        return result
    try:
        data = feature_frame(candles)
        latest = data.iloc[-1:]
        instant = latest.close_time.iloc[0]
        result['feature_as_of'] = instant.isoformat()
        if not freshness(instant, 90 * 60)['fresh']:
            raise ValueError('Market inputs are stale; no current model output is shown.')
        with warnings.catch_warnings():
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.simplefilter('error', InconsistentVersionWarning)
            bundle = joblib.load(evidence['model_path'])
        if bundle.get('model_version') != evidence['model_version'] or bundle.get('sklearn_version') != sklearn.__version__:
            raise ValueError('Trend model version or software environment changed; reevaluation required.')
        if bundle.get('uses_news'):
            extra = news_features_asof(root, [instant])
            latest = latest.merge(extra, left_on='close_time', right_on='prediction_time')
        if latest[bundle['features']].isna().any().any():
            raise ValueError('Latest causal feature row is incomplete.')
        p = predict_bundle(bundle, latest)[0]
        generated_at = pd.Timestamp.now(tz='UTC').isoformat()
        result.update(experimental_label=OUTCOMES[int(p.argmax())], generated_at=generated_at)
        forecast = dict(model_sha256=evidence['model_sha256'], horizon_hours=24,
            generated_at=generated_at, feature_as_of=instant.isoformat(), probabilities=dict(zip(OUTCOMES, p.tolist())))
        gate = validated_forecast(artifacts, instant, forecast=forecast)
        result.update(validated=gate['validated'], probabilities=gate.get('probabilities'),
                      limitation=gate['limitation'], message=gate['message'])
        return result
    except Exception as exc:
        result['limitation'] = str(exc)
        return result
