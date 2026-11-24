"""Incremental, ordinary public Binance hourly-candle research collection."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

from live_data import BINANCE, USER_AGENT
from news_label_safety import atomic_json

COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume']


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.csv.tmp')
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_candles(path):
    frame = pd.read_csv(path, float_precision='round_trip')
    for key in ('open_time', 'close_time'):
        frame[key] = pd.to_datetime(frame[key], utc=True, errors='raise', format='ISO8601')
    return frame


def collect_candles(folder, days=730, now=None, session=None, pause=.25):
    """Checkpoint each page; never retry through 429/418 or fill price gaps."""
    if not 180 <= days <= 1825:
        raise ValueError('Request 180 to 1825 days of research history.')
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if current.tzinfo is None:
        raise ValueError('Collection time must be timezone-aware.')
    current = current.tz_convert('UTC')
    end = current.floor('h')
    start = end - pd.Timedelta(days=days + 7)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, state_path = folder / 'xrp_hourly.csv', folder / 'collection_state.json'
    import json
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    cooldown = pd.to_datetime(state.get('cooldown_until'), utc=True, errors='coerce')
    if pd.notna(cooldown) and cooldown > current:
        raise RuntimeError('Binance cooldown is active until ' + cooldown.isoformat())
    frame = read_candles(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    if not frame.empty and (frame.open_time.duplicated().any() or frame.close_time.isna().any()):
        raise ValueError('Existing candle cache has duplicated or malformed timestamps.')
    expected = pd.date_range(start, end - pd.Timedelta(hours=1), freq='h')
    existing = pd.DatetimeIndex(frame.open_time) if not frame.empty else pd.DatetimeIndex([], tz='UTC')
    missing = expected.difference(existing)
    if path.exists() and len(missing):
        backups = folder / 'backups'
        backups.mkdir(exist_ok=True)
        shutil.copy2(path, backups / ('xrp_hourly_' + pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ') + '.csv'))
    client = session or requests.Session()
    requests_made = 0
    try:
        while len(missing):
            first = missing[0]
            # Bounded contiguous request. Existing pages are retained on failure.
            last = min(first + pd.Timedelta(hours=1000), end)
            if os.getenv('XRP_DASHBOARD_OFFLINE') == '1':
                raise RuntimeError('Offline mode: historical collection disabled.')
            response = client.get(BINANCE + '/api/v3/klines', params={
                'symbol': 'XRPUSDT', 'interval': '1h', 'startTime': int(first.timestamp() * 1000),
                'endTime': int(last.timestamp() * 1000) - 1, 'limit': 1000},
                headers={'User-Agent': USER_AGENT}, timeout=25)
            requests_made += 1
            if response.status_code in (418, 429):
                try:
                    seconds = max(60, int(response.headers.get('Retry-After', 3600)))
                except ValueError:
                    seconds = 3600
                until = pd.Timestamp.now(tz='UTC') + pd.Timedelta(seconds=seconds)
                atomic_json(state_path, dict(cooldown_until=until.isoformat(), error='HTTP ' + str(response.status_code)))
                raise RuntimeError(f'Binance HTTP {response.status_code}; no retry before {until.isoformat()}')
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, list) or not raw:
                raise RuntimeError('Binance did not return the requested historical candles; no prices invented.')
            batch = pd.DataFrame(raw, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'unused'])
            for key in ('open_time', 'close_time'):
                batch[key] = pd.to_datetime(batch[key], unit='ms', utc=True)
            for key in ('open', 'high', 'low', 'close', 'volume', 'quote_volume'):
                batch[key] = pd.to_numeric(batch[key], errors='raise')
            batch = batch[(batch.open_time >= start) & (batch.close_time < end)]
            if batch.empty or not (batch.open_time == first).any():
                raise RuntimeError('Historical candle gap or non-advancing pagination; collection stopped safely.')
            frame = pd.concat([frame, batch[COLUMNS]], ignore_index=True) if not frame.empty else batch[COLUMNS].copy()
            frame = frame.sort_values('open_time').drop_duplicates('open_time', keep='first').reset_index(drop=True)
            save_frame(path, frame)
            missing = expected.difference(pd.DatetimeIndex(frame.open_time))
            atomic_json(state_path, dict(source=BINANCE, symbol='XRPUSDT', interval='1h',
                collected_at=pd.Timestamp.now(tz='UTC').isoformat(), requested_start=start.isoformat(),
                requested_end=end.isoformat(), rows=len(frame), remaining_hours=len(missing), requests_this_run=requests_made))
            if requests_made > 150:
                raise RuntimeError('Bounded request budget reached; rerun to resume.')
            if len(missing) and pause:
                time.sleep(pause)
    finally:
        if session is None:
            client.close()
    return frame[(frame.open_time >= start) & (frame.close_time < end)].reset_index(drop=True)
