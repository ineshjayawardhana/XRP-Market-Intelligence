"""Bounded public connectivity smoke check. Does not save prices or alter research."""
import json
import time

import pandas as pd

from streaming_market import INTERVALS, MarketStream, model_candles


def check():
    feed = MarketStream()
    result = {'rest': {}, 'websocket': {}, 'scope': 'Public market-data connectivity, not model evaluation'}
    try:
        feed.repair()
        snapshot = feed.snapshot()
        result['rest'] = dict(trade_available=snapshot['trade'] is not None,
            trade_fresh=snapshot['trade_fresh'], error=snapshot['rest_error'],
            candles={i: len(feed.snapshot(i)['candles']) for i in INTERVALS})
        # Test stream ingestion itself, without hiding connection errors behind REST.
        from websockets.sync.client import connect
        from streaming_market import STREAM
        events, first_trade = {}, None
        end = time.monotonic() + 20
        with connect(STREAM, open_timeout=6, close_timeout=2, ping_interval=20, ping_timeout=20) as socket:
            while time.monotonic() < end:
                data = json.loads(socket.recv(timeout=5))
                payload = data.get('data', data)
                feed.ingest(data)
                name = payload.get('e')
                if name == 'kline':
                    name += '_' + payload['k']['i']
                events[name] = events.get(name, 0) + 1
                if payload.get('e') == 'trade':
                    first_trade = feed.snapshot()['trade']['trade_at'].isoformat()
                if first_trade and all('kline_' + interval in events for interval in INTERVALS):
                    break
        result['websocket'] = dict(connected=True, events=events, last_trade_at=first_trade)
    except Exception as exc:
        result['websocket'] = dict(connected=False, error=str(exc))
    finally:
        snap = feed.snapshot()
        hourly = model_candles(None, snap)
        now = feed.clock()
        safe = (not hourly.empty and bool((hourly.close_time < now).all())
                and bool((hourly.close_time - hourly.open_time).eq(pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).all()))
        result['model_boundary'] = dict(completed_hourly_rows=len(hourly),
            latest_close=str(hourly.close_time.max()) if not hourly.empty else None,
            forming_bars_excluded=safe)
        result['checked_at'] = str(feed.clock())
        feed.close()
    return result


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
