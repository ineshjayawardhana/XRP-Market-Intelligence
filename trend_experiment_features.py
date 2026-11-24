"""Causal feature experiments, isolated from the deployed feature signature.

These are candidate covariates, not evidence of predictive skill. The caller
must choose/tune them on chronological development data and preserve its final
holdout. Importing this module never reads or writes a dataset or model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_pipeline import FEATURES, feature_frame

EXPERIMENT_FEATURE_VERSION = 'causal_expanded_v1'
FEATURE_GROUPS = {
    'base': list(FEATURES),
    'lagged_returns': [f'return_1h_lag_{hours}h' for hours in (1, 3, 6, 24)],
    'additional_momentum': [f'return_{hours}h' for hours in (3, 48, 168)],
    'rsi_momentum': [f'rsi_change_{hours}h' for hours in (1, 6, 24)],
    'ema_crossover_momentum': [f'ema_gap_change_{hours}h' for hours in (1, 6, 24)],
    'macd_momentum': [f'macd_hist_change_{hours}h' for hours in (1, 6, 24)],
    'relative_volume': [f'volume_relative_{hours}h' for hours in (24, 168)],
    'additional_volatility': ['volatility_6h', 'volatility_168h', 'volatility_ratio_24_168h'],
    'historical_range': [f'distance_{side}_{hours}h' for hours in (24, 168) for side in ('high', 'low')],
    'scaled_momentum': ['return_acceleration_1_6h', 'risk_adjusted_return_24h'],
}
EXTRA_FEATURES = [name for group, names in FEATURE_GROUPS.items() if group != 'base' for name in names]
EXPANDED_FEATURES = list(FEATURES) + EXTRA_FEATURES


def experiment_features(candles, now=None):
    """Return base+candidate features from completed candles only.

    Return/indicator changes use their historical values at the stated lag.
    Volume and high/low reference windows use preceding completed candles,
    excluding the current candle. There is no centering, interpolation or fill.
    Undefined ratios remain missing and are later excluded from supervised rows.
    """
    data = feature_frame(candles, now=now)
    close = data.close
    for hours in (1, 3, 6, 24):
        data[f'return_1h_lag_{hours}h'] = data.return_1h.shift(hours)
    for hours in (3, 48, 168):
        data[f'return_{hours}h'] = close.pct_change(hours, fill_method=None)
    for hours in (1, 6, 24):
        # RSI changes are scaled to fractions of its 0--100 range.
        data[f'rsi_change_{hours}h'] = data.rsi.diff(hours) / 100.
        data[f'ema_gap_change_{hours}h'] = data.ema_gap.diff(hours)
        data[f'macd_hist_change_{hours}h'] = data.macd_hist_relative.diff(hours)
    for hours in (24, 168):
        volume_reference = data.volume.shift(1).rolling(hours, min_periods=hours).mean()
        data[f'volume_relative_{hours}h'] = data.volume / volume_reference.replace(0, np.nan)
        high = data.high.shift(1).rolling(hours, min_periods=hours).max()
        low = data.low.shift(1).rolling(hours, min_periods=hours).min()
        data[f'distance_high_{hours}h'] = close / high - 1.
        data[f'distance_low_{hours}h'] = close / low - 1.
    for hours in (6, 168):
        data[f'volatility_{hours}h'] = data.return_1h.rolling(hours, min_periods=hours).std()
    data['volatility_ratio_24_168h'] = data.volatility_24h / data.volatility_168h.replace(0, np.nan)
    data['return_acceleration_1_6h'] = data.return_1h - data.return_6h / 6.
    data['risk_adjusted_return_24h'] = data.return_24h / (data.volatility_24h * np.sqrt(24)).replace(0, np.nan)
    data[EXPANDED_FEATURES] = data[EXPANDED_FEATURES].replace([np.inf, -np.inf], np.nan)
    return data


def experiment_dataset(candles, now=None):
    """Build exact-timestamp 24h outcomes; exclude unknown horizons and warm-up."""
    data = experiment_features(candles, now=now)
    data['target_end'] = data.close_time + pd.Timedelta(hours=24)
    data['future_close'] = data.target_end.map(data.set_index('close_time').close)
    data['future_return_24h'] = data.future_close / data.close - 1.
    data['target'] = np.select([data.future_return_24h >= .01, data.future_return_24h <= -.01],
                               ['UP', 'DOWN'], default='NEUTRAL')
    return data.dropna(subset=EXPANDED_FEATURES + ['future_close']).reset_index(drop=True)


def segmented_experiment_dataset(candles, now=None):
    """Build safe independent datasets around missing source hours.

    Strict ``experiment_dataset`` retains its gap rejection. This opt-in
    research adapter rejects malformed input globally, then restarts every
    indicator/rolling window at each gap. No feature or 24h target crosses a
    missing hour; at least168 warm-up hours and24 known target hours must lie
    inside the same segment. No candle or label is interpolated.
    """
    required = {'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(candles):
        raise ValueError('Segmented research input is missing required OHLCV fields.')
    data = candles.copy(deep=True).reset_index(drop=True)
    for key in ('open_time', 'close_time'):
        if not isinstance(data[key].dtype, pd.DatetimeTZDtype):
            raise ValueError('Research candle timestamps must be timezone-aware datetime columns.')
        data[key] = data[key].dt.tz_convert('UTC')
        if data[key].isna().any():
            raise ValueError('Research candle timestamps are missing.')
    close_times = data.close_time
    if close_times.duplicated().any() or not close_times.is_monotonic_increasing:
        raise ValueError('Research candles are duplicated or unordered.')
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if current.tzinfo is None:
        raise ValueError('Research evaluation time must be timezone-aware.')
    if not data.empty and close_times.max() > current:
        raise ValueError('A forming or future research candle is present.')
    if (not (data.close_time - data.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).all()
            or not data.open_time.eq(data.open_time.dt.floor('h')).all()):
        raise ValueError('Research candles must be aligned complete one-hour Binance intervals.')
    numeric = data[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    if (not np.isfinite(numeric.to_numpy()).all() or (numeric[['open', 'high', 'low', 'close']] <= 0).any().any()
            or (numeric.volume < 0).any()
            or (numeric.high < numeric[['open', 'close', 'low']].max(axis=1)).any()
            or (numeric.low > numeric[['open', 'close', 'high']].min(axis=1)).any()):
        raise ValueError('Research OHLCV values are invalid or inconsistent.')
    data[numeric.columns] = numeric
    delta = close_times.diff()
    boundaries = delta.ne(pd.Timedelta(hours=1))
    segment_ids = boundaries.cumsum()
    gaps = []
    for index in data.index[boundaries & delta.notna()]:
        gaps.append(dict(before=close_times.iloc[index - 1].isoformat(), after=close_times.iloc[index].isoformat(),
                         missing_hours=int(delta.iloc[index] / pd.Timedelta(hours=1)) - 1))
    parts, segments = [], []
    for identifier, segment in data.groupby(segment_ids, sort=False):
        output = experiment_dataset(segment.reset_index(drop=True), now=now) if len(segment) >= 200 else None
        count = len(output) if output is not None else 0
        segments.append(dict(segment=int(identifier), start=segment.close_time.min().isoformat(),
                             end=segment.close_time.max().isoformat(), candles=len(segment), usable_rows=count))
        if count:
            output['contiguous_segment'] = int(identifier)
            parts.append(output)
    if parts:
        result = pd.concat(parts, ignore_index=True)
    else:
        columns = list(dict.fromkeys(list(data.columns) + EXPANDED_FEATURES + ['ema20', 'ema50', 'macd',
            'macd_signal', 'technical', 'volume_spike', 'overbought', 'target_end', 'future_close',
            'future_return_24h', 'target', 'contiguous_segment']))
        result = pd.DataFrame(columns=columns)
        for key in ('open_time', 'close_time', 'target_end'):
            result[key] = pd.Series(dtype='datetime64[ns, UTC]')
    result.attrs['gap_policy'] = 'restart_all_features_and_targets_per_contiguous_segment'
    result.attrs['gaps'] = gaps
    result.attrs['missing_hours'] = sum(gap['missing_hours'] for gap in gaps)
    result.attrs['segments'] = segments
    result.attrs['warmup_hours'] = 168
    result.attrs['target_horizon_hours'] = 24
    result.attrs['excluded_rows'] = len(data) - len(result)
    return result


def causal_regimes(frame, training_frame):
    """Descriptive evaluation strata using only past indicators and train cutoffs.

    The volatility split is fitted to training observations only. These regime
    names are post-hoc diagnostic groups, not extra model targets or proof of
    skill. The caller must keep training earlier than its evaluation block.
    """
    required = {'close_time', 'ema_gap', 'volatility_24h'}
    if not required.issubset(frame) or not required.issubset(training_frame):
        raise ValueError('Regime inputs need close_time, ema_gap and volatility_24h.')
    values = pd.to_numeric(training_frame.volatility_24h, errors='coerce')
    values = values[np.isfinite(values)]
    if values.empty:
        raise ValueError('Training volatility observations are unavailable.')
    threshold = float(values.median())
    gap = pd.to_numeric(frame.ema_gap, errors='coerce')
    volatility = pd.to_numeric(frame.volatility_24h, errors='coerce')
    trend = np.select([gap > .001, gap < -.001, gap.notna()],
                      ['Uptrend', 'Downtrend', 'Sideways'], default='Unavailable')
    vol = np.select([volatility > threshold, volatility.notna()],
                    ['High volatility', 'Lower volatility'], default='Unavailable')
    result = pd.DataFrame({'close_time': frame.close_time, 'trend_regime': trend,
                           'volatility_regime': vol}, index=frame.index)
    result['combined_regime'] = result.trend_regime + ' / ' + result.volatility_regime
    result.attrs['training_volatility_median'] = threshold
    result.attrs['trend_gap_threshold'] = .001
    result.attrs['interpretation'] = 'Descriptive strata; no subgroup was used to select a model.'
    return result
