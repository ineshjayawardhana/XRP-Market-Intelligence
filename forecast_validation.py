"""Fail-closed contracts for evaluated 24h direction probability artifacts.

No probabilities are inferred from news-model confidence or heuristic votes.
Trusted local validation artifacts, reproducible OOS/calibration checks, matching
model hashes and a fresh forecast are all required. V2 supports explicit
retrospective research replays; v1 contracts remain backwards compatible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_label_safety import sha256

OUTCOMES = ['UP', 'DOWN', 'NEUTRAL']


def validated_forecast(artifacts, market_time=None, now=None, forecast=None):
    artifacts = Path(artifacts)
    current = pd.to_datetime(now, utc=True) if now is not None else pd.Timestamp.now(tz='UTC')
    result = dict(validated=False, message='Predictive probabilities are not yet validated.',
                  sample_size=None, evaluated_at=None, test_start=None, test_end=None,
                  probabilities=None, limitation='No evaluated 24-hour direction model and calibration evidence are available.')
    contract_path = artifacts / 'trend_24h_validation.json'
    if not contract_path.exists():
        return result
    try:
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        if not isinstance(contract, dict):
            raise ValueError('Validation contract must be a structured object.')
        if contract.get('schema_version') == 2:
            return _forecast_v2(artifacts, market_time, current, forecast)
        result.update(sample_size=contract.get('test_samples'), evaluated_at=contract.get('evaluated_at'),
                      test_start=contract.get('test_start'), test_end=contract.get('test_end'))
        if (contract.get('task') != 'XRP_24H_DIRECTION' or contract.get('horizon_hours') != 24
                or contract.get('labels') != OUTCOMES or contract.get('chronological_out_of_sample_passed') is not True
                or contract.get('calibration_passed') is not True):
            raise ValueError('Wrong prediction task or missing chronological/calibration approval.')
        def local_file(name):
            path = artifacts / name
            if path.resolve().parent != artifacts.resolve() or not path.is_file():
                raise ValueError('Validation must reference existing local artifact files.')
            return path
        model = local_file(contract['model_file'])
        predictions = local_file(contract['oos_predictions_file'])
        if sha256(model) != contract['model_sha256'] or sha256(predictions) != contract['oos_predictions_sha256']:
            raise ValueError('Model or evaluation artifact hash changed.')
        dates = {key: pd.to_datetime(contract[key], utc=True, errors='raise') for key in
                 ['train_end', 'calibration_fit_end', 'test_start', 'test_end', 'evaluated_at']}
        if (any(pd.isna(value) for value in dates.values()) or
                not dates['train_end'] < dates['calibration_fit_end'] or
                not dates['calibration_fit_end'] + pd.Timedelta(hours=24) <= dates['test_start'] or
                not dates['test_start'] < dates['test_end'] or
                not dates['test_end'] + pd.Timedelta(hours=24) <= dates['evaluated_at'] <= current):
            raise ValueError('Chronological periods, embargo or completed outcome horizon are invalid.')
        test = pd.read_csv(predictions)
        if len(test) < 100 or len(test) != contract['test_samples']:
            raise ValueError('At least 100 chronological holdout examples are required by the dashboard validation policy.')
        required = {'available_at', 'prediction_generated_at', 'true_outcome', 'p_UP', 'p_DOWN', 'p_NEUTRAL'}
        if not required.issubset(test.columns):
            raise ValueError('Out-of-sample prediction evidence is incomplete.')
        times = pd.to_datetime(test.available_at, utc=True, errors='coerce')
        generated = pd.to_datetime(test.prediction_generated_at, utc=True, errors='coerce')
        if (times.isna().any() or generated.isna().any() or times.duplicated().any()
                or not times.is_monotonic_increasing or not times.between(dates['test_start'], dates['test_end']).all()
                or not (generated <= times).all() or test.true_outcome.isin(OUTCOMES).eq(False).any()
                or any(test.true_outcome.value_counts().get(label, 0) < 10 for label in OUTCOMES)):
            raise ValueError('Holdout timing, class coverage or outcome labels are invalid.')
        p = test[['p_UP', 'p_DOWN', 'p_NEUTRAL']].to_numpy(dtype=float)
        if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=.001)):
            raise ValueError('Holdout probability vectors are invalid.')
        targets = np.array([OUTCOMES.index(label) for label in test.true_outcome])
        correct = p.argmax(axis=1) == targets
        confidence = p.max(axis=1)
        ece = 0.0
        for index in range(10):
            mask = (confidence >= index / 10) & (confidence <= (index + 1) / 10 if index == 9 else confidence < (index + 1) / 10)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        actual = np.eye(3)[targets]
        brier = float(np.square(p - actual).sum(axis=1).mean())
        baseline = float(np.square(actual.mean(axis=0)[None, :] - actual).sum(axis=1).mean())
        if ece > .10 or brier >= baseline:
            raise ValueError('Recomputed calibration error or Brier-skill check did not pass.')
        forecast = json.loads((artifacts / 'trend_24h_forecast.json').read_text(encoding='utf-8'))
        instant = pd.to_datetime(forecast['generated_at'], utc=True, errors='raise')
        feature_time = pd.to_datetime(forecast['feature_as_of'], utc=True, errors='raise')
        latest_market = pd.to_datetime(market_time, utc=True, errors='coerce')
        if (forecast.get('model_sha256') != contract['model_sha256'] or forecast.get('horizon_hours') != 24
                or pd.isna(instant) or pd.isna(feature_time) or pd.isna(latest_market)
                or not current - pd.Timedelta(minutes=30) <= instant <= current
                or not current - pd.Timedelta(minutes=90) <= latest_market <= current
                or feature_time != latest_market or not feature_time <= instant):
            raise ValueError('Latest forecast is stale, mismatched or based on unavailable market inputs.')
        values = np.array([forecast['probabilities'][label] for label in OUTCOMES], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1, atol=.001):
            raise ValueError('Latest forecast probabilities are invalid.')
        result.update(validated=True, message='Validated chronological research forecast',
                      probabilities=dict(zip(OUTCOMES, values.tolist())),
                      limitation='Historical calibration can change across market regimes; this is not financial advice.',
                      calibration_ece=ece, brier_score=brier, brier_baseline=baseline,
                      forecast_generated_at=instant.isoformat())
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        result['limitation'] = str(exc)
    return result


def validated_research(artifacts, now=None):
    """Reproduce v2 artifact metrics; allow an experimental class after integrity checks.

    Training provenance is explicit local research provenance, not a prospective
    track record. Quality failures never expose probabilities to the main view.
    """
    from trend_metrics import QUALITY_MESSAGES, metrics, quality_checks
    artifacts = Path(artifacts)
    current = pd.to_datetime(now, utc=True) if now is not None else pd.Timestamp.now(tz='UTC')
    result = dict(integrity_passed=False, validated=False, quality_failures=[],
                  limitation='No evaluated trend research artifacts are available.')
    try:
        contract = json.loads((artifacts / 'trend_24h_validation.json').read_text(encoding='utf-8'))
        if (contract.get('schema_version') != 2 or contract.get('pipeline') != 'xrp_chronological_trend_v2'
                or contract.get('task') != 'XRP_24H_DIRECTION' or contract.get('horizon_hours') != 24
                or contract.get('labels') != OUTCOMES or contract.get('evaluation_type') != 'retrospective_replay'):
            raise ValueError('An explicit v2 chronological replay contract is required.')
        def checked_file(key, hash_key):
            path = (artifacts / contract[key]).resolve()
            if not path.is_relative_to(artifacts.resolve()) or not path.is_file() or sha256(path) != contract[hash_key]:
                raise ValueError('Missing or modified research artifact: ' + key)
            return path
        paths = {name: checked_file(key, hash_key) for name, key, hash_key in [
            ('model', 'model_file', 'model_sha256'), ('oos', 'oos_predictions_file', 'oos_predictions_sha256'),
            ('report', 'report_file', 'report_sha256'), ('dataset', 'dataset_file', 'dataset_sha256'),
            ('market', 'market_file', 'market_sha256')]}
        report = json.loads(paths['report'].read_text(encoding='utf-8'))
        if (report.get('schema_version') != 2 or report.get('model_version') != contract['model_version']
                or report.get('selected_before_holdout') is not True):
            raise ValueError('Model selection provenance does not match the evaluated artifact.')
        split = report['split']
        dates = {key: pd.to_datetime(split[key], utc=True, errors='raise') for key in
                 ['train_start', 'train_end', 'calibration_start', 'calibration_end', 'test_start', 'test_end']}
        evaluated = pd.to_datetime(report['evaluated_at'], utc=True, errors='raise')
        if (any(pd.isna(value) for value in dates.values()) or pd.isna(evaluated)
                or not dates['train_start'] <= dates['train_end']
                or not dates['train_end'] + pd.Timedelta(hours=24) < dates['calibration_start']
                or not dates['calibration_start'] <= dates['calibration_end']
                or not dates['calibration_end'] + pd.Timedelta(hours=24) < dates['test_start']
                or not dates['test_start'] < dates['test_end']
                or not dates['test_end'] + pd.Timedelta(hours=24) <= evaluated <= current):
            raise ValueError('Research split periods or target-maturity purging are invalid.')
        dataset, oos = pd.read_csv(paths['dataset']), pd.read_csv(paths['oos'])
        for frame in (dataset, oos):
            key = 'close_time' if frame is dataset else 'available_at'
            frame[key] = pd.to_datetime(frame[key], utc=True, errors='raise')
            frame['target_end'] = pd.to_datetime(frame.target_end, utc=True, errors='raise')
            times = frame[key]
            if (times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing
                    or (times.diff().dropna() < pd.Timedelta(hours=24)).any()
                    or not frame.target_end.eq(times + pd.Timedelta(hours=24)).all()):
                raise ValueError('Invalid or overlapping 24h research origins/targets.')
        computed = pd.to_datetime(oos.prediction_computed_at, utc=True, errors='raise')
        if (not oos.evaluation_type.eq('retrospective_replay').all() or computed.isna().any()
                or not (computed <= evaluated).all() or not (computed >= oos.target_end).all()
                or not oos.available_at.between(dates['test_start'], dates['test_end']).all()
                or oos.available_at.min() != dates['test_start'] or oos.available_at.max() != dates['test_end']):
            raise ValueError('Historical replay timing is inconsistent; no backdated live predictions are allowed.')
        # Reconstruct targets and causal features directly from the frozen market snapshot.
        from trend_market_data import read_candles
        from trend_pipeline import FEATURES, feature_frame, feature_signature, predict_bundle
        import joblib
        import sklearn
        import warnings
        market = read_candles(paths['market'])
        derived = feature_frame(market, current).set_index('close_time')
        if (market.close_time.max() - market.close_time.min()) < pd.Timedelta(days=180):
            raise ValueError('Less than six months of historical market evidence.')
        origin = derived.reindex(dataset.close_time)
        future = derived.close.reindex(pd.DatetimeIndex(dataset.target_end)).to_numpy()
        returns = future / origin.close.to_numpy() - 1
        targets = np.select([returns >= .01, returns <= -.01], ['UP', 'DOWN'], default='NEUTRAL')
        if (not np.isfinite(returns).all() or not np.array_equal(targets, dataset.target.to_numpy())
                or not np.allclose(returns, dataset.future_return_24h.to_numpy())
                or not np.allclose(dataset[FEATURES].to_numpy(), origin[FEATURES].to_numpy(), atol=1e-10)):
            raise ValueError('Saved features or outcomes do not match the frozen causal market data.')
        frames = {name: dataset[dataset.close_time.between(dates[name + '_start'], dates[name + '_end'])].copy()
                  for name in ('train', 'calibration', 'test')}
        if any(len(frame) != split[name + '_samples'] for name, frame in frames.items()):
            raise ValueError('Research partition counts changed.')
        if (not np.array_equal(frames['test'].close_time.to_numpy(), oos.available_at.to_numpy())
                or not np.array_equal(frames['test'].target.to_numpy(), oos.true_outcome.to_numpy())
                or not np.allclose(frames['test'].future_return_24h.to_numpy(), oos.true_return_24h.to_numpy())
                or not pd.to_datetime(oos.simulated_train_end, utc=True).eq(dates['train_end']).all()
                or not pd.to_datetime(oos.simulated_calibration_end, utc=True).eq(dates['calibration_end']).all()
                or len(oos) != contract['test_samples']):
            raise ValueError('Holdout outcome evidence is inconsistent.')
        with warnings.catch_warnings():
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.simplefilter('error', InconsistentVersionWarning)
            bundle = joblib.load(paths['model'])
        if (bundle['model_version'] != contract['model_version'] or bundle['features'] != report['features']
                or bundle.get('feature_signature') != feature_signature()
                or contract.get('feature_signature') != feature_signature()
                or report.get('feature_signature') != feature_signature()
                or bundle['sklearn_version'] != sklearn.__version__
                or any(feature not in FEATURES and not feature.startswith(('gold_', 'silver_')) for feature in bundle['features'])):
            raise ValueError('Trend bundle version, software or feature policy mismatch.')
        if bundle.get('uses_news'):
            snapshots = contract.get('news_snapshot_files', [])
            if len(snapshots) < 3:
                raise ValueError('News-enhanced evidence requires frozen provenance snapshots.')
            snapshot_root = paths['model'].parent / 'news_snapshot'
            for entry in snapshots:
                path = (artifacts / entry['file']).resolve()
                if not path.is_relative_to(snapshot_root.resolve()) or sha256(path) != entry['sha256']:
                    raise ValueError('Historical news snapshot is missing or modified.')
            from trend_news import news_features_asof
            frozen = news_features_asof(snapshot_root, dataset.close_time)
            columns = [key for key in frozen if key != 'prediction_time']
            if not np.array_equal(frozen[columns].to_numpy(), dataset[columns].to_numpy()):
                raise ValueError('Historical news counts do not match frozen point-in-time evidence.')
        p = oos[['p_' + label for label in OUTCOMES]].to_numpy(dtype=float)
        if not np.allclose(p, predict_bundle(bundle, frames['test']), atol=1e-9):
            raise ValueError('Saved OOS scores do not reproduce from the exact saved model.')
        prior = np.array([(frames['train'].target == label).mean() for label in OUTCOMES])
        if not np.allclose(prior, bundle['training_prior']):
            raise ValueError('Class-prior baseline does not match the training partition.')
        majority = np.zeros_like(p)
        majority[:, prior.argmax()] = 1
        rule = derived.reindex(oos.available_at).technical.map({'Bullish': 'UP', 'Bearish': 'DOWN', 'Neutral': 'NEUTRAL'})
        technical = np.array([[float(label == value) for label in OUTCOMES] for value in rule])
        baselines = {key: metrics(oos.true_outcome, value) for key, value in dict(majority=majority,
            training_prior=np.tile(prior, (len(oos), 1)), technical_rule=technical).items()}
        measured = metrics(oos.true_outcome, p)
        for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'brier_score', 'calibration_ece', 'log_loss'):
            if not np.isclose(measured[key], report['holdout'][key]):
                raise ValueError('Recorded metrics do not reproduce.')
        wf = report['candidates'][report['selected_model']]['walk_forward']
        quality = quality_checks(measured, baselines, wf['macro_f1'] > wf['baselines']['majority']['macro_f1'])
        reasons = [QUALITY_MESSAGES.get(key, key) for key in quality['failures']]
        result.update(integrity_passed=True, validated=quality['passed'], quality_failures=reasons,
            limitation=' '.join(reasons) if reasons else 'Retrospective policy checks passed; future reliability is not guaranteed.',
            model_version=contract['model_version'], model_path=str(paths['model']), model_sha256=contract['model_sha256'],
            sample_size=len(oos), evaluated_at=evaluated.isoformat(), test_start=dates['test_start'].isoformat(),
            test_end=dates['test_end'].isoformat(), metrics=measured, baselines=baselines)
    except Exception as exc:
        result['limitation'] = str(exc)
    return result


def _forecast_v2(artifacts, market_time, current, forecast=None):
    evidence = validated_research(artifacts, current)
    result = dict(validated=False, message='Predictive probabilities are not yet validated.', probabilities=None,
        limitation=evidence['limitation'], sample_size=evidence.get('sample_size'), evaluated_at=evidence.get('evaluated_at'),
        test_start=evidence.get('test_start'), test_end=evidence.get('test_end'))
    if not evidence['validated']:
        return result
    try:
        if forecast is None:
            forecast = json.loads((Path(artifacts) / 'trend_24h_forecast.json').read_text(encoding='utf-8'))
        generated = pd.to_datetime(forecast['generated_at'], utc=True, errors='raise')
        feature_time = pd.to_datetime(forecast['feature_as_of'], utc=True, errors='raise')
        latest = pd.to_datetime(market_time, utc=True, errors='raise')
        if (forecast['model_sha256'] != evidence['model_sha256'] or forecast['horizon_hours'] != 24
                or not current - pd.Timedelta(minutes=30) <= generated <= current
                or not current - pd.Timedelta(minutes=90) <= latest <= generated or feature_time != latest):
            raise ValueError('Current forecast is stale or does not match evaluated model and completed market input.')
        p = np.array([forecast['probabilities'][label] for label in OUTCOMES], dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1):
            raise ValueError('Current probability vector is invalid.')
        result.update(validated=True, probabilities=dict(zip(OUTCOMES, p.tolist())),
            message='Chronologically evaluated research forecast', limitation=evidence['limitation'])
    except Exception as exc:
        result['limitation'] = str(exc)
    return result
