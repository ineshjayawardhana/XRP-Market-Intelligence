"""Bounded public hourly evidence for delayed paper-model observations."""
import os
import threading
import time

import pandas as pd
import requests

from streaming_market import REST, stamp, valid_bar


class PublicHourlyEvidence:
    def __init__(self):
        self.lock = threading.Lock()
        self.cooldown_until = 0

    def fetch(self, target):
        if os.getenv('XRP_DASHBOARD_OFFLINE') == '1':
            raise ValueError('Offline mode: public historical requests are disabled.')
        target = pd.Timestamp(target)
        if target.tzinfo is None or target >= pd.Timestamp.now(tz='UTC'):
            raise ValueError('Only completed, timestamped hourly evidence can be requested.')
        start = target - pd.Timedelta(hours=1) + pd.Timedelta(milliseconds=1)
        if start != start.floor('h'):
            raise ValueError('The recorded target is not an exact native hourly close.')
        with self.lock:
            if time.monotonic() < self.cooldown_until:
                raise ValueError('Public REST cooldown is active; observation remains pending.')
            response = requests.get(REST + '/klines', params=dict(symbol='XRPUSDT', interval='1h',
                startTime=int(start.timestamp() * 1000), endTime=int(target.timestamp() * 1000), limit=1), timeout=6)
            if response.status_code in (429, 418):
                try:
                    delay = max(60, float(response.headers.get('Retry-After', 60)))
                except (ValueError, TypeError):
                    delay = 60
                self.cooldown_until = time.monotonic() + delay
            response.raise_for_status()
        rows = []
        for row in response.json():
            bar = dict(open_time=stamp(row[0]), close_time=stamp(row[6]),
                **dict(zip(['open', 'high', 'low', 'close', 'volume', 'quote_volume'],
                    [float(row[n]) for n in (1, 2, 3, 4, 5, 7)])), closed=True)
            if bar['close_time'] == target and valid_bar(bar, '1h'):
                rows.append(bar)
        if len(rows) != 1:
            raise ValueError('The exact completed target candle is unavailable; observation remains pending.')
        return pd.DataFrame(rows)
