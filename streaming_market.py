"""Public Binance market data only. No credentials, orders or research-file writes.

The background worker owns the socket. Streamlit reads locked snapshots; it never
waits for a network request. Chart bars and completed hourly model bars are separate.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import threading
import time

import pandas as pd
import requests

INTERVALS = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
REST = 'https://data-api.binance.vision/api/v3'
STREAM = ('wss://data-stream.binance.vision/stream?streams=' +
          '/'.join(['xrpusdt@trade'] + ['xrpusdt@kline_' + i for i in INTERVALS]))


def current_time():
    return pd.Timestamp.now(tz='UTC')


def stamp(ms):
    return pd.to_datetime(int(ms), unit='ms', utc=True)


def age_ok(value, seconds, now):
    if value is None:
        return False
    age = (now - pd.Timestamp(value)).total_seconds()
    return 0 <= age <= seconds


def valid_bar(bar, interval):
    duration = pd.Timedelta(seconds=INTERVALS[interval]) - pd.Timedelta(milliseconds=1)
    values = [float(bar[k]) for k in ('open', 'high', 'low', 'close', 'volume', 'quote_volume')]
    return (bar['close_time'] - bar['open_time'] == duration
            and bar['open_time'].value % pd.Timedelta(seconds=INTERVALS[interval]).value == 0
            and all(math.isfinite(v) for v in values) and min(values[:4]) > 0
            and min(values[4:]) >= 0 and values[2] <= min(values[0], values[3])
            and values[1] >= max(values[0], values[3]))


class MarketStream:
    def __init__(self, connector=None, http=None, clock=current_time):
        self.connector, self.http, self.clock = connector, http or requests.Session(), clock
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = self.socket = None
        self.connected = False
        self.trade = None
        self.bars = {i: {} for i in INTERVALS}
        self.last_message_at = None
        self.last_error = self.rest_error = None
        self.reconnects = 0
        self.cooldown_until = 0.
        self.last_access = time.monotonic()
        self.last_rest = self.last_repair = -float('inf')

    def start(self):
        if os.getenv('XRP_DASHBOARD_OFFLINE') == '1':
            return self
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.stop_event.clear()
                self.thread = threading.Thread(target=self._run, name='xrp-public-stream', daemon=True)
                self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        if self.socket:
            self.socket.close()

    def _put_bar(self, interval, bar):
        if not valid_bar(bar, interval):
            raise ValueError('Invalid or partial-duration candle received.')
        old = self.bars[interval].get(bar['open_time'])
        # Late REST/stream messages must not roll back a newer bar or reopen a close.
        if old and (old['updated_at'] > bar['updated_at'] or (old['closed'] and not bar['closed'])):
            return
        self.bars[interval][bar['open_time']] = bar
        if len(self.bars[interval]) > 1000:
            del self.bars[interval][min(self.bars[interval])]

    def _put_trade(self, price, trade_at, received_at, source):
        if not math.isfinite(price) or price <= 0 or trade_at > received_at + pd.Timedelta(seconds=5):
            raise ValueError('Invalid price or future trade timestamp.')
        if self.trade is None or trade_at >= self.trade['trade_at']:
            self.trade = dict(price=price, trade_at=trade_at, received_at=received_at, source=source)

    def ingest(self, message):
        """Validate exchange events. Exchange times never become local receipt times."""
        payload = json.loads(message) if isinstance(message, (str, bytes)) else message
        data = payload.get('data', payload)
        if data.get('e') == 'serverShutdown':
            raise ConnectionError('Exchange requested socket reconnection.')
        if data.get('s') != 'XRPUSDT':
            return
        now, event = self.clock(), stamp(data['E'])
        if event > now + pd.Timedelta(seconds=5):
            raise ValueError('Exchange event timestamp is in the future.')
        with self.lock:
            if data['e'] == 'trade':
                self._put_trade(float(data['p']), stamp(data['T']), now, 'WebSocket trade')
            elif data['e'] == 'kline':
                k = data['k']
                interval = k['i']
                if interval not in INTERVALS:
                    return
                if type(k['x']) is not bool:
                    raise ValueError('Candle closed flag is missing or invalid.')
                bar = dict(open_time=stamp(k['t']), close_time=stamp(k['T']),
                    **{name: float(k[key]) for name, key in [('open', 'o'), ('high', 'h'),
                    ('low', 'l'), ('close', 'c'), ('volume', 'v'), ('quote_volume', 'q')]},
                    closed=k['x'], updated_at=event, received_at=now, source='WebSocket kline')
                if bar['open_time'] > now or (bar['closed'] and bar['close_time'] > now):
                    raise ValueError('Future candle cannot be complete.')
                self._put_bar(interval, bar)
            else:
                return
            self.last_message_at = now

    def _get(self, endpoint, **params):
        if time.monotonic() < self.cooldown_until:
            raise RuntimeError('REST rate-limit cooldown is active.')
        response = self.http.get(REST + endpoint, params=params, timeout=6)
        if response.status_code in (429, 418):
            try:
                retry = max(60., float(response.headers.get('Retry-After', 60)))
            except (ValueError, TypeError):
                retry = 60.
            self.cooldown_until = time.monotonic() + retry
        response.raise_for_status()
        return response.json()

    def repair(self, intervals=tuple(INTERVALS), trade=True):
        """Bootstrap/backfill bounded native bars. No invented minute aggregation."""
        issues = []
        if trade:
            try:
                rows = self._get('/trades', symbol='XRPUSDT', limit=1)
                with self.lock:
                    self._put_trade(float(rows[-1]['price']), stamp(rows[-1]['time']), self.clock(), 'REST last trade')
            except Exception as exc:
                issues.append('Trade fallback: ' + str(exc))
        for interval in intervals:
            try:
                requested_at = self.clock()
                rows = self._get('/klines', symbol='XRPUSDT', interval=interval, limit=600)
                now = self.clock()
                with self.lock:
                    for row in rows:
                        bar = dict(open_time=stamp(row[0]), close_time=stamp(row[6]),
                            **dict(zip(['open', 'high', 'low', 'close', 'volume', 'quote_volume'],
                                [float(row[n]) for n in (1, 2, 3, 4, 5, 7)])),
                            closed=stamp(row[6]) < requested_at, updated_at=requested_at, received_at=now, source='REST kline')
                        if bar['open_time'] <= now:
                            self._put_bar(interval, bar)
            except Exception as exc:
                issues.append(interval + ' fallback: ' + str(exc))
        with self.lock:
            self.rest_error = '; '.join(issues) or None

    def snapshot(self, interval='1m', touch=True):
        with self.lock:
            if touch:
                self.last_access = time.monotonic()
            now = self.clock()
            trade = copy.deepcopy(self.trade)
            bars = copy.deepcopy(list(self.bars[interval].values()))
            hourly = copy.deepcopy(list(self.bars['1h'].values()))
            connected = self.connected and age_ok(self.last_message_at, 15, now)
            trade_fresh = bool(trade and age_ok(trade['trade_at'], 15 if trade['source'].startswith('WebSocket') else 45, now)
                               and age_ok(trade['received_at'], 45, now))
            latest = max(bars, key=lambda b: b['open_time']) if bars else None
            chart_fresh = bool(latest and age_ok(latest['updated_at'], 45, now)
                               and latest['open_time'] <= now <= latest['close_time'] + pd.Timedelta(seconds=45))
            return dict(trade=trade, candles=pd.DataFrame(bars).sort_values('open_time') if bars else pd.DataFrame(),
                hourly=pd.DataFrame(hourly).sort_values('open_time') if hourly else pd.DataFrame(),
                connected=bool(connected), trade_fresh=trade_fresh, chart_fresh=chart_fresh,
                last_message_at=self.last_message_at, last_error=self.last_error,
                rest_error=self.rest_error, reconnects=self.reconnects, checked_at=now)

    def stale_channels(self):
        """Cheap worker health check: never construct dataframes for every trade."""
        now = self.clock()
        with self.lock:
            trade_ok = bool(self.trade and age_ok(self.trade['trade_at'], 15 if self.trade['source'].startswith('WebSocket') else 45, now)
                            and age_ok(self.trade['received_at'], 45, now))
            stale = []
            for interval, bars in self.bars.items():
                latest = bars[max(bars)] if bars else None
                if not (latest and age_ok(latest['updated_at'], 45, now)
                        and latest['open_time'] <= now <= latest['close_time'] + pd.Timedelta(seconds=45)):
                    stale.append(interval)
            return not trade_ok, stale

    def _run(self):
        from websockets.sync.client import connect
        connector = self.connector or connect
        delay = 1.
        while not self.stop_event.is_set() and time.monotonic() - self.last_access < 300:
            try:
                # Fallback before connection attempts, including a failed initial handshake.
                if time.monotonic() - self.last_rest >= 20:
                    self.repair()
                    self.last_rest = self.last_repair = time.monotonic()
                with connector(STREAM, open_timeout=6, close_timeout=2, ping_interval=20, ping_timeout=20) as socket:
                    self.socket = socket
                    opened_at = self.clock()
                    with self.lock:
                        self.connected = True
                    while not self.stop_event.is_set() and time.monotonic() - self.last_access < 300:
                        try:
                            self.ingest(socket.recv(timeout=1))
                            delay = 1.
                        except TimeoutError:
                            pass
                        if time.monotonic() - self.last_rest >= 20:
                            trade_stale, stale_intervals = self.stale_channels()
                            if trade_stale or stale_intervals:
                                self.repair(stale_intervals, trade=trade_stale)
                            self.last_rest = time.monotonic()
                        elif time.monotonic() - self.last_repair >= 120:
                            # Repair packets missing during interruptions even after streaming resumes.
                            self.repair(trade=False)
                            self.last_repair = time.monotonic()
                        if not age_ok(self.last_message_at or opened_at, 30, self.clock()):
                            raise ConnectionError('No valid exchange events for 30 seconds.')
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                    self.reconnects += 1
            finally:
                with self.lock:
                    self.connected = False
                self.socket = None
            until = time.monotonic() + min(60, delay) + random.uniform(0, .5)
            while not self.stop_event.wait(1) and time.monotonic() < until:
                if time.monotonic() - self.last_rest >= 20:
                    self.repair()
                    self.last_rest = time.monotonic()
            delay = min(60, delay * 2)


def model_candles(base, snapshot, now=None):
    """Only native, completed, exact-hour bars; no chart bar can enter the model."""
    now = current_time() if now is None else pd.Timestamp(now)
    stream = snapshot.get('hourly', pd.DataFrame())
    if not stream.empty:
        stream = stream[stream.closed.eq(True)].copy()
    parts = [part for part in (base, stream) if part is not None and not part.empty]
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame['open_time'] = pd.to_datetime(frame.open_time, utc=True)
    frame['close_time'] = pd.to_datetime(frame.close_time, utc=True)
    valid = ((frame.close_time < now)
        & (frame.close_time - frame.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1))
        & frame.open_time.eq(frame.open_time.dt.floor('h')))
    columns = ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
    result = frame[valid].drop_duplicates('open_time', keep='last').sort_values('open_time')
    return result[columns].reset_index(drop=True)


def quote_context(ticker, snapshot):
    """Replace only the quote clocks/price; rolling 24h stats keep their own clock."""
    result = dict(ticker or {})
    trade = snapshot.get('trade')
    if trade and snapshot['trade_fresh']:
        result.update(price=trade['price'], retrieved_at=trade['received_at'].isoformat(),
            exchange_close_at=trade['trade_at'].isoformat(), trade_at=trade['trade_at'].isoformat())
    return result or None
