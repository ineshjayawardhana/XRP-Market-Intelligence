"""Separate 1-hour research, frozen historical reserve, never 24h labels."""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from live_data import BINANCE, USER_AGENT
from news_label_safety import atomic_json, sha256
from trend_market_data import COLUMNS, read_candles, save_frame
from trend_metrics import OUTCOMES, metrics, quality_checks, QUALITY_MESSAGES
from trend_pipeline import FEATURES, feature_frame, fit_temperature, predict_bundle, probability_matrix

ROOT = Path(__file__).resolve().parent
TEST_START = pd.Timestamp('2021-01-01T00:00Z')
TEST_END = pd.Timestamp('2021-11-01T00:00Z')
FAMILIES = ['LogisticRegression', 'RandomForest', 'HistGradientBoosting', 'ExtraTrees']


def definition():
    return dict(task='XRP_1H_DIRECTION', horizon_hours=1, labels=OUTCOMES, start='2018-06-01T00:00Z',
        training_start='2018-07-01T00:00Z', test_start=TEST_START.isoformat(), test_end_exclusive=TEST_END.isoformat(),
        threshold='symmetric +/- one-third quantile of absolute next-hour returns from classifier-training rows only',
        folds=3, validation_days_per_fold=60, calibration_days=30, embargo='earlier target_end strictly before next feature origin',
        features=FEATURES, families=FAMILIES, selection='pooled validation macro-F1, tie-break Brier; never test metrics',
        model_settings=dict(logistic_C=.5, tree_estimators=200, tree_max_depth=6, tree_min_leaf=64,
            hgb_iterations=100, hgb_leaves=7, hgb_learning_rate=.05, hgb_l2=5., early_stopping=False, seed=42),
        code_sha256=sha256(Path(__file__)), feature_source_sha256=sha256(Path(__file__).with_name('trend_pipeline.py')),
        scope='newly collected retrospective 2021 historical reserve, not prospective current-market validation')


def collect(folder):
    """Permitted public historical range, checkpoints, real clock for cooldowns."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, state_path = folder / 'xrp_hourly.csv', folder / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    until = pd.to_datetime(state.get('cooldown_until'), utc=True, errors='coerce')
    if pd.notna(until) and until > pd.Timestamp.now(tz='UTC'):
        raise RuntimeError('Binance cooldown until ' + until.isoformat())
    frame = read_candles(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    cursor = frame.open_time.max() + pd.Timedelta(hours=1) if not frame.empty else pd.Timestamp('2018-06-01T00:00Z')
    calls = 0
    with requests.Session() as session:
        while cursor < TEST_END:
            response = session.get(BINANCE + '/api/v3/klines', params=dict(symbol='XRPUSDT', interval='1h',
                startTime=int(cursor.timestamp() * 1000), endTime=int(TEST_END.timestamp() * 1000) - 1, limit=1000),
                headers={'User-Agent': USER_AGENT}, timeout=25)
            calls += 1
            if response.status_code in (418, 429):
                try:
                    seconds = max(60, int(response.headers.get('Retry-After', 3600)))
                except ValueError:
                    seconds = 3600
                until = pd.Timestamp.now(tz='UTC') + pd.Timedelta(seconds=seconds)
                atomic_json(state_path, dict(cooldown_until=until.isoformat(), error='HTTP ' + str(response.status_code)))
                raise RuntimeError('Public API rate limit; no retry before ' + until.isoformat())
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                raise RuntimeError('No historical candles returned; no prices invented.')
            batch = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'trades', 'taker_base', 'taker_quote', 'unused'])
            for key in ('open_time', 'close_time'):
                batch[key] = pd.to_datetime(batch[key], unit='ms', utc=True)
            for key in ('open', 'high', 'low', 'close', 'volume', 'quote_volume'):
                batch[key] = pd.to_numeric(batch[key], errors='raise')
            batch = batch[(batch.open_time >= cursor) & (batch.close_time < TEST_END)]
            if batch.empty:
                raise RuntimeError('Non-advancing public candle response.')
            cursor = batch.open_time.max() + pd.Timedelta(hours=1)
            frame = pd.concat([frame, batch[COLUMNS]], ignore_index=True) if not frame.empty else batch[COLUMNS].copy()
            frame = frame.sort_values('open_time').drop_duplicates('open_time', keep='first').reset_index(drop=True)
            save_frame(path, frame)
            atomic_json(state_path, dict(rows=len(frame), collected_at=pd.Timestamp.now(tz='UTC').isoformat(),
                source=BINANCE, requests_this_run=calls, requested_end=TEST_END.isoformat()))
            if calls >= 100:
                raise RuntimeError('Request budget reached; rerun to resume.')
            if cursor < TEST_END:
                time.sleep(.25)
    return frame


def hourly_features(candles):
    """Targets are new exact next-close returns; old target columns are ignored."""
    if candles.close_time.duplicated().any() or candles.open_time.duplicated().any() or not candles.close_time.is_monotonic_increasing:
        raise ValueError('Duplicate or unordered candles.')
    complete = (candles.close_time - candles.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1))
    data = candles[complete].copy().reset_index(drop=True)
    segments = data.close_time.diff().ne(pd.Timedelta(hours=1)).cumsum()
    parts = []
    for _, segment in data.groupby(segments, sort=False):
        if len(segment) < 170:
            continue
        frame = feature_frame(segment.reset_index(drop=True))
        frame['target_end'] = frame.close_time + pd.Timedelta(hours=1)
        frame['next_close'] = frame.target_end.map(frame.set_index('close_time').close)
        frame['actual_return_1h'] = frame.next_close / frame.close - 1
        parts.append(frame.iloc[168:].dropna(subset=FEATURES + ['next_close']))
    if not parts:
        raise ValueError('No complete contiguous feature/next-hour outcome history.')
    return pd.concat(parts, ignore_index=True)


def labels(returns, threshold):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('A positive finite training-derived threshold is required.')
    return np.select([returns >= threshold, returns <= -threshold], ['UP', 'DOWN'], default='NEUTRAL')


def learn_threshold(train):
    threshold = float(train.actual_return_1h.abs().quantile(1 / 3))
    if not np.isfinite(threshold) or threshold <= 1e-8:
        raise ValueError('Training returns cannot support the predefined threshold rule.')
    return threshold


def split(data, boundary, calibration_days=30):
    eligible = data[data.target_end < boundary]
    cal_start = boundary - pd.Timedelta(days=calibration_days)
    calibration = eligible[eligible.close_time >= cal_start].copy()
    train = eligible[eligible.target_end < cal_start].copy()
    if len(train) < 1000 or len(calibration) < 100:
        raise ValueError('Insufficient strictly earlier training/calibration data.')
    return train, calibration


def fit(family, train, calibration):
    threshold = learn_threshold(train)
    y = labels(train.actual_return_1h, threshold)
    if set(y) != set(OUTCOMES):
        raise ValueError('Training must cover all three outcomes.')
    if family == 'LogisticRegression':
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=1500, random_state=42))
    elif family in ('RandomForest', 'ExtraTrees'):
        cls = RandomForestClassifier if family == 'RandomForest' else ExtraTreesClassifier
        estimator = cls(n_estimators=200, max_depth=6, min_samples_leaf=64, max_features=.8, n_jobs=2, random_state=42)
    elif family == 'HistGradientBoosting':
        estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=64,
            learning_rate=.05, l2_regularization=5., early_stopping=False, random_state=42)
    else:
        raise ValueError('Unsupported family.')
    estimator.fit(train[FEATURES], y)
    temperature = fit_temperature(probability_matrix(estimator, calibration[FEATURES]), labels(calibration.actual_return_1h, threshold))
    return dict(estimator=estimator, threshold=threshold, temperature=temperature, family=family, features=FEATURES,
        training_prior=[float(np.mean(y == label)) for label in OUTCOMES], sklearn_version=sklearn.__version__, horizon_hours=1)


def baselines(bundle, frame):
    prior = np.asarray(bundle['training_prior'])
    majority = np.zeros((len(frame), 3))
    majority[:, prior.argmax()] = 1
    technical = frame.technical.map({'Bullish': 'UP', 'Bearish': 'DOWN', 'Neutral': 'NEUTRAL'})
    return dict(majority=majority, training_prior=np.tile(prior, (len(frame), 1)),
        technical_rule=np.array([[float(value == label) for label in OUTCOMES] for value in technical]))


def replay(bundle, frame):
    p = predict_bundle(bundle, frame)
    result = pd.DataFrame(dict(available_at=frame.close_time.to_numpy(), target_end=frame.target_end.to_numpy(),
        actual_return_1h=frame.actual_return_1h.to_numpy(), true_outcome=labels(frame.actual_return_1h, bundle['threshold']),
        prediction_computed_at=pd.Timestamp.now(tz='UTC').isoformat(), evaluation_type='retrospective_replay'))
    for index, label in enumerate(OUTCOMES):
        result['p_' + label] = p[:, index]
        for name, probabilities in baselines(bundle, frame).items():
            result[name + '_' + label] = probabilities[:, index]
    return result


def measure(rows, prefix='p'):
    return metrics(rows.true_outcome, rows[[prefix + '_' + label for label in OUTCOMES]].values)


def period(frame):
    return dict(start=frame.close_time.min().isoformat(), end=frame.close_time.max().isoformat(), samples=len(frame),
                last_target_end=frame.target_end.max().isoformat())


def feature_signature():
    source = ''.join(inspect.getsource(fn) for fn in (hourly_features, labels, learn_threshold, feature_frame))
    return hashlib.sha256((source + repr(FEATURES)).encode()).hexdigest()


def train(root=ROOT, offline=False, progress=print):
    root = Path(root)
    folder = root / 'data' / 'trend_1h'
    folder.mkdir(parents=True, exist_ok=True)
    plan = definition()
    protocol_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    lock = folder / 'reserve.json'
    if lock.exists():
        state = json.loads(lock.read_text())
        if state['status'] == 'evaluated':
            # Always return the original frozen result, never train/rescore this
            # consumed reserve. Reader/inference repairs do not relabel that run.
            path = root / state['report_file']
            if sha256(path) != state['report_sha256']:
                raise ValueError('Saved evaluated report changed.')
            return json.loads(path.read_text())
        if state['protocol_hash'] != protocol_hash:
            raise ValueError('Frozen 1h method changed; do not reuse its test. New untouched dates are required.')
        if state['status'] == 'opened':
            raise ValueError('Final labels were already opened; no refit/rescore allowed.')
    else:
        # Neither old cache may contain the new reserve before freezing the plan.
        for path in (root / 'data/trend/xrp_hourly.csv', root / 'data/trend_experiments/cache/xrp_hourly.csv'):
            if path.exists():
                times = read_candles(path).close_time
                if times.between(TEST_START, TEST_END, inclusive='left').any():
                    raise ValueError('The proposed 1h reserve was already in an older dataset.')
        atomic_json(lock, dict(status='reserved', protocol_hash=protocol_hash, plan=plan, frozen_at=pd.Timestamp.now(tz='UTC').isoformat()))
    progress('Frozen 1h protocol: ' + protocol_hash[:16])
    raw_path = folder / 'cache/xrp_hourly.csv'
    raw = read_candles(raw_path) if offline else collect(raw_path.parent)
    # Final reserve data is not used to construct development targets.
    development = hourly_features(raw[raw.close_time < TEST_START].copy())
    development = development[development.close_time >= pd.Timestamp(plan['training_start'])].copy()
    final_train, final_cal = split(development, TEST_START)
    candidates, pooled = {}, {}
    with threadpool_limits(limits=2):
        for family in FAMILIES:
            folds, rows = [], []
            for number in range(3):
                end = final_train.close_time.max() + pd.Timedelta(milliseconds=1) - pd.Timedelta(days=(2 - number) * 60)
                start = end - pd.Timedelta(days=60)
                validation = final_train[(final_train.close_time >= start) & (final_train.close_time < end)]
                earlier, calibration = split(final_train, start)
                bundle = fit(family, earlier, calibration)
                scored = replay(bundle, validation)
                rows.append(scored)
                folds.append(dict(fold=number + 1, threshold=bundle['threshold'], train=period(earlier), calibration=period(calibration),
                    validation=period(validation), metrics=measure(scored),
                    baselines={name: measure(scored, name) for name in baselines(bundle, validation)}))
                progress(f'{family}: validation fold {number + 1}/3 complete')
            combined = pd.concat(rows, ignore_index=True)
            candidates[family] = dict(validation=measure(combined), folds=folds)
            pooled[family] = combined
        selected = max(candidates, key=lambda name: (candidates[name]['validation']['macro_f1'], -candidates[name]['validation']['brier_score']))
        bundle = fit(selected, final_train, final_cal)
    version = 'xrp1h_' + pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    run = root / 'artifacts' / 'trend_1h_runs' / version
    run.mkdir(parents=True, exist_ok=False)
    bundle.update(version=version, feature_signature=feature_signature(), training_end=final_train.close_time.max().isoformat(),
                  calibration_end=final_cal.close_time.max().isoformat())
    joblib.dump(bundle, run / 'model.joblib')
    save_frame(run / 'development.csv', development)
    save_frame(run / 'market_snapshot.csv', raw)
    for family in FAMILIES:
        save_frame(run / ('validation_' + family + '.csv'), pooled[family])
    selection = dict(selected=selected, threshold=bundle['threshold'], model_sha256=sha256(run / 'model.joblib'),
        frozen_at=pd.Timestamp.now(tz='UTC').isoformat(), protocol_hash=protocol_hash)
    atomic_json(run / 'selection.json', selection)
    atomic_json(lock, dict(status='opened', protocol_hash=protocol_hash, plan=plan, selection=selection))
    full = hourly_features(raw[raw.close_time < TEST_END].copy())
    test = full[full.close_time >= TEST_START].copy()
    with threadpool_limits(limits=2):
        oos = replay(bundle, test)
        test_metrics = measure(oos)
        base = {name: measure(oos, name) for name in baselines(bundle, test)}
        uncalibrated = metrics(labels(test.actual_return_1h, bundle['threshold']), predict_bundle(bundle, test, calibrated=False))
    if test.close_time.duplicated().any() or (test.close_time.iloc[1:].reset_index(drop=True) < test.target_end.iloc[:-1].reset_index(drop=True)).any():
        raise ValueError('Invalid overlapping or duplicated final outcomes.')
    # Existing numerical policy reused, with explicit new 1h identity and extra stability evidence.
    wf = pooled[selected]
    historical_quality = quality_checks(test_metrics, base, measure(wf)['macro_f1'] > measure(wf, 'majority')['macro_f1'])
    save_frame(run / 'test_dataset.csv', test)
    save_frame(run / 'oos.csv', oos)
    report = dict(task='XRP_1H_DIRECTION', horizon_hours=1, version=version, evaluated_at=pd.Timestamp.now(tz='UTC').isoformat(),
        selected_model=selected, selected_before_test=True, threshold=bundle['threshold'], threshold_pct=100 * bundle['threshold'],
        threshold_rule=plan['threshold'], candidates=candidates, test=test_metrics, baselines=base,
        uncalibrated_test=uncalibrated, historical_quality=historical_quality,
        split=dict(train=period(final_train), calibration=period(final_cal), test=period(test)), plan=plan,
        data=dict(rows=len(raw), earliest=raw.close_time.min().isoformat(), latest=raw.close_time.max().isoformat(), source=BINANCE),
        limitations=['Newly collected older historical replay, not prospective current-market validation.',
            'The next completed hourly candle is the target; this is not 60 minutes from an intrahour dashboard refresh.',
            'Adjacent hourly outcomes share endpoints but do not overlap return increments. Serial dependence and regime shifts remain.',
            'Thresholds differ between validation folds because each uses its own earlier training data; the final threshold is frozen.',
            'No 24-hour labels, old fitted weights or news labels enter this new target.',
            'No all-data refit follows the test. Frozen model age and historical-market shifts limit present-day inference.'])
    atomic_json(run / 'report.json', report)
    pointer = dict(task=report['task'], horizon_hours=1, version=version, feature_signature=feature_signature(),
        files={name: dict(path=str((run / filename).relative_to(root)), sha256=sha256(run / filename)) for name, filename in
               [('model', 'model.joblib'), ('report', 'report.json'), ('test', 'test_dataset.csv'), ('oos', 'oos.csv'),
                ('market', 'market_snapshot.csv'), ('development', 'development.csv')]})
    atomic_json(root / 'artifacts' / 'trend_1h_current.json', pointer)
    atomic_json(lock, dict(status='evaluated', protocol_hash=protocol_hash, plan=plan,
        report_file=str((run / 'report.json').relative_to(root)), report_sha256=sha256(run / 'report.json')))
    return report


def verified_model(root=ROOT):
    """Independent task/hash/threshold/score checks, without updating artifacts."""
    root = Path(root)
    pointer = json.loads((root / 'artifacts/trend_1h_current.json').read_text())
    if pointer['task'] != 'XRP_1H_DIRECTION' or pointer['horizon_hours'] != 1 or pointer['feature_signature'] != feature_signature():
        raise ValueError('Wrong task, horizon or changed feature/target implementation.')
    paths = {}
    for name, entry in pointer['files'].items():
        path = (root / entry['path']).resolve()
        if not path.is_relative_to((root / 'artifacts/trend_1h_runs').resolve()) or sha256(path) != entry['sha256']:
            raise ValueError('Missing or modified 1-hour research artifact.')
        paths[name] = path
    bundle, report = joblib.load(paths['model']), json.loads(paths['report'].read_text())
    if bundle['horizon_hours'] != 1 or bundle['sklearn_version'] != sklearn.__version__ or bundle['version'] != pointer['version']:
        raise ValueError('Model or software version mismatch.')
    if report.get('task') != 'XRP_1H_DIRECTION' or report.get('horizon_hours') != 1 or report.get('selected_before_test') is not True:
        raise ValueError('Missing 1-hour evaluation/selection provenance.')
    periods = report['split']
    if (not pd.Timestamp(periods['train']['last_target_end']) < pd.Timestamp(periods['calibration']['start'])
            or not pd.Timestamp(periods['calibration']['last_target_end']) < pd.Timestamp(periods['test']['start'])
            or not pd.Timestamp(periods['test']['last_target_end']) <= pd.Timestamp(report['evaluated_at']) <= pd.Timestamp.now(tz='UTC')):
        raise ValueError('Chronology, target maturity or calibration purging failed.')
    development = pd.read_csv(paths['development'], float_precision='round_trip')
    train_mask = pd.to_datetime(development.close_time, utc=True).between(pd.Timestamp(report['split']['train']['start']), pd.Timestamp(report['split']['train']['end']))
    threshold = learn_threshold(development[train_mask])
    if not np.isclose(threshold, bundle['threshold']) or not np.isclose(threshold, report['threshold']):
        raise ValueError('Threshold is inconsistent with classifier-training data.')
    test, oos = pd.read_csv(paths['test'], float_precision='round_trip'), pd.read_csv(paths['oos'], float_precision='round_trip')
    times = pd.to_datetime(test.close_time, utc=True)
    ends = pd.to_datetime(test.target_end, utc=True)
    if (times.duplicated().any() or not times.is_monotonic_increasing or not (ends - times).eq(pd.Timedelta(hours=1)).all()
            or (times.iloc[1:].reset_index(drop=True) < ends.iloc[:-1].reset_index(drop=True)).any()):
        raise ValueError('One-hour holdout timing/overlap checks failed.')
    raw = read_candles(paths['market']).set_index('close_time')
    actual = raw.close.reindex(ends).to_numpy() / raw.close.reindex(times).to_numpy() - 1
    if not np.allclose(actual, test.actual_return_1h) or not np.array_equal(labels(actual, threshold), oos.true_outcome):
        raise ValueError('Targets do not reproduce from exact next-hour market closes.')
    training = development[train_mask]
    training_times = pd.to_datetime(training.close_time, utc=True)
    training_ends = pd.to_datetime(training.target_end, utc=True)
    training_returns = raw.close.reindex(training_ends).to_numpy() / raw.close.reindex(training_times).to_numpy() - 1
    if not np.allclose(training_returns, training.actual_return_1h):
        raise ValueError('Threshold-training returns do not match the market snapshot.')
    reconstructed = hourly_features(read_candles(paths['market'])).set_index('close_time').reindex(times)
    if not np.allclose(reconstructed[FEATURES], test[FEATURES], atol=1e-10):
        raise ValueError('Saved 1-hour features do not reproduce from causal market history.')
    if not np.array_equal(times.to_numpy(), pd.to_datetime(oos.available_at, utc=True).to_numpy()):
        raise ValueError('OOS scores and feature origins do not align.')
    with threadpool_limits(limits=2):
        probabilities = predict_bundle(bundle, test)
    if not np.allclose(probabilities, oos[['p_' + label for label in OUTCOMES]], atol=1e-9):
        raise ValueError('Saved model cannot reproduce its OOS scores.')
    measured = metrics(oos.true_outcome, probabilities)
    base = {name: metrics(oos.true_outcome, p) for name, p in baselines(bundle, test).items()}
    quality = quality_checks(measured, base, report['historical_quality']['checks']['development_walk_forward_skill'])
    return bundle, report, quality


def live_forecast(candles, root=ROOT):
    result = dict(validated=False, probabilities=None, experimental_label=None, threshold=None,
        message='Predictive probabilities are not yet validated.', limitation='No evaluated 1-hour model is available.',
        model_version=None, feature_as_of=None, generated_at=None, evaluated_at=None, sample_size=None,
        test_start=None, test_end=None, quality_failures=[])
    try:
        bundle, report, quality = verified_model(root)
        result.update(model_version=bundle['version'], threshold=bundle['threshold'], evaluated_at=report['evaluated_at'],
            sample_size=report['test']['sample_size'], test_start=report['split']['test']['start'], test_end=report['split']['test']['end'],
            metrics=report['test'], selected_model=report['selected_model'])
        frame = feature_frame(candles)
        latest = frame.iloc[-1:]
        now = pd.Timestamp.now(tz='UTC')
        instant = latest.close_time.iloc[0]
        result['feature_as_of'] = instant.isoformat()
        if instant != now.floor('h') - pd.Timedelta(milliseconds=1) or latest[FEATURES].isna().any().any():
            raise ValueError('Latest completed hourly input is stale or incomplete; current signal withheld.')
        with threadpool_limits(limits=2):
            p = predict_bundle(bundle, latest)[0]
        result.update(experimental_label=OUTCOMES[p.argmax()], generated_at=now.isoformat())
        reasons = [QUALITY_MESSAGES.get(key, key) for key in quality['failures']]
        if pd.Timestamp(report['split']['test']['last_target_end']) < now - pd.Timedelta(days=180):
            reasons.append('Out-of-sample market period is older than 180 days; current-regime probability reliability is unvalidated.')
        result.update(validated=not reasons, quality_failures=reasons,
            limitation=' '.join(reasons) if reasons else 'Historical research checks passed; future performance is not guaranteed.')
        if not reasons:
            result['probabilities'] = dict(zip(OUTCOMES, p.tolist()))
    except Exception as exc:
        result['limitation'] = str(exc)
    return result
