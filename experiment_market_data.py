"""Experimental collection retains reported gaps instead of inventing candles."""
import json
import time
from pathlib import Path

import pandas as pd
import requests

from live_data import BINANCE, USER_AGENT
from news_label_safety import atomic_json
from trend_market_data import COLUMNS, read_candles, save_frame


def collect_experiment_candles(folder, days=1825):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, state_path = folder / 'xrp_hourly.csv', folder / 'collection_state.json'
    current = pd.Timestamp.now(tz='UTC')
    end = current.floor('h')
    start = end - pd.Timedelta(days=days + 7)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    until = pd.to_datetime(state.get('cooldown_until'), utc=True, errors='coerce')
    if pd.notna(until) and until > current:
        raise RuntimeError('Binance cooldown active until ' + until.isoformat())
    frame = read_candles(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    cursor = frame.open_time.max() + pd.Timedelta(hours=1) if not frame.empty else start
    if not frame.empty and frame.open_time.min() > start:
        raise ValueError('Existing experimental cache starts too late; preserve it and use an explicitly new collection folder.')
    if not frame.empty and (frame.open_time.duplicated().any() or not frame.open_time.is_monotonic_increasing):
        raise ValueError('Experimental raw cache is duplicated or unordered.')
    count = 0
    with requests.Session() as session:
        while cursor < end:
            response = session.get(BINANCE + '/api/v3/klines', params={'symbol': 'XRPUSDT', 'interval': '1h',
                'startTime': int(cursor.timestamp() * 1000), 'endTime': int(end.timestamp() * 1000) - 1, 'limit': 1000},
                headers={'User-Agent': USER_AGENT}, timeout=25)
            count += 1
            if response.status_code in (418, 429):
                try:
                    seconds = max(60, int(response.headers.get('Retry-After', '3600')))
                except ValueError:
                    seconds = 3600
                until = pd.Timestamp.now(tz='UTC') + pd.Timedelta(seconds=seconds)
                atomic_json(state_path, dict(cooldown_until=until.isoformat(), error='HTTP ' + str(response.status_code)))
                raise RuntimeError('Binance rate limit; stopped until ' + until.isoformat())
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, list) or not raw:
                raise RuntimeError('No further historical candles returned; no data generated.')
            batch = pd.DataFrame(raw, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'trades', 'taker_base', 'taker_quote', 'unused'])
            for key in ('open_time', 'close_time'):
                batch[key] = pd.to_datetime(batch[key], unit='ms', utc=True)
            for key in ('open', 'high', 'low', 'close', 'volume', 'quote_volume'):
                batch[key] = pd.to_numeric(batch[key], errors='raise')
            batch = batch[(batch.open_time >= cursor) & (batch.close_time < end)]
            if batch.empty:
                raise RuntimeError('Non-advancing historical response; collection stopped.')
            cursor = batch.open_time.max() + pd.Timedelta(hours=1)
            frame = pd.concat([frame, batch[COLUMNS]], ignore_index=True) if not frame.empty else batch[COLUMNS].copy()
            frame = frame.sort_values('open_time').drop_duplicates('open_time', keep='first').reset_index(drop=True)
            save_frame(path, frame)
            atomic_json(state_path, dict(source=BINANCE, symbol='XRPUSDT', interval='1h', rows=len(frame),
                collected_at=pd.Timestamp.now(tz='UTC').isoformat(), requested_start=start.isoformat(),
                requested_end=end.isoformat(), requests_this_run=count, gap_policy='retain missing hours; never fill prices'))
            if count >= 150:
                raise RuntimeError('Bounded request budget reached; rerun to resume.')
            if cursor < end:
                time.sleep(.25)
    expected = pd.date_range(frame.open_time.min(), frame.open_time.max(), freq='h')
    missing = expected.difference(pd.DatetimeIndex(frame.open_time))
    atomic_json(folder / 'data_quality.json', dict(rows=len(frame), missing_hours=[value.isoformat() for value in missing],
        missing_hour_count=len(missing), source=BINANCE, detail='Expected hours absent from cached public kline responses. No price interpolation or substitute candle.',
        handling='Separate contiguous feature/target segments with fresh warm-up after every gap.'))
    return frame
