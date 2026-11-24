"""Local, virtual-only spot ledger. Decimal accounting; atomic SQLite actions.

No network, exchange credentials, orders, model fitting or research-file writes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

LABELS = ('UP', 'DOWN', 'NEUTRAL')
INITIAL = Decimal('100')


def decimal(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('A finite numeric value is required.') from None
    if not result.is_finite():
        raise ValueError('A finite numeric value is required.')
    return result


def instant(value):
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tzinfo is None:
        raise ValueError('A timezone-aware timestamp is required.')
    return date.tz_convert('UTC')


def fresh_quote(quote, now):
    now = instant(now)
    if not isinstance(quote, dict) or quote.get('source') not in ('WebSocket trade', 'REST last trade'):
        raise ValueError('A fresh public Binance trade is required; ticker-only prices cannot fill virtual positions.')
    price = decimal(quote.get('price'))
    trade_at, received = instant(quote.get('trade_at')), instant(quote.get('received_at'))
    limit = 15 if quote['source'] == 'WebSocket trade' else 45
    if price <= 0 or not 0 <= (now - trade_at).total_seconds() <= limit or not 0 <= (now - received).total_seconds() <= 45:
        raise ValueError('Market data is stale, unavailable or in the future. Virtual fills are blocked.')
    return price, trade_at, received


def outcome(change, threshold):
    value, boundary = decimal(change), decimal(threshold)
    if not 0 < boundary < 1:
        raise ValueError('Invalid recorded training threshold.')
    return 'UP' if value >= boundary else 'DOWN' if value <= -boundary else 'NEUTRAL'


def clean_signal(signal, now):
    """Keep an immutable classification snapshot, never prediction probabilities."""
    if not isinstance(signal, dict) or signal.get('signal') not in LABELS:
        return dict(signal=None, reason=str((signal or {}).get('reason', 'Hourly signal unavailable at entry.')) if isinstance(signal, dict) else 'Hourly signal unavailable at entry.')
    try:
        origin, target, generated = [instant(signal[k]) for k in ('origin_at', 'target_at', 'generated_at')]
        if (origin != now.floor('h') - pd.Timedelta(milliseconds=1) or target != origin + pd.Timedelta(hours=1)
                or not origin <= generated <= now or decimal(signal['origin_price']) <= 0 or not signal['version']):
            raise ValueError('Signal clocks or price are invalid.')
        outcome(0, signal['threshold'])
        return {k: signal[k] for k in ('signal', 'origin_at', 'target_at', 'generated_at', 'origin_price', 'version', 'threshold', 'reason')}
    except (ValueError, KeyError, TypeError):
        return dict(signal=None, reason='The hourly signal snapshot was invalid or stale at entry.')


def capture_signal(forecast, candles, now):
    """Whitelist provenance, never retain or expose unreliable probabilities."""
    result = dict(signal=None, reason=str(forecast.get('limitation', 'Signal unavailable.')))
    try:
        origin = instant(forecast['feature_as_of'])
        generated = instant(forecast['generated_at'])
        now = instant(now)
        if origin != now.floor('h') - pd.Timedelta(milliseconds=1) or not origin <= generated <= now:
            raise ValueError('Hourly model signal is stale or future-dated.')
        if forecast['experimental_label'] not in LABELS or not forecast['model_version']:
            raise ValueError('No experimental hourly classification is available.')
        threshold = decimal(forecast['threshold'])
        outcome(0, threshold)
        rows = candles[pd.to_datetime(candles.close_time, utc=True).eq(origin)]
        if len(rows) != 1:
            raise ValueError('Exact model origin candle is unavailable.')
        row = rows.iloc[0]
        if origin - instant(row.open_time) != pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1):
            raise ValueError('Model origin is not a completed native hourly candle.')
        price = decimal(row.close)
        if price <= 0:
            raise ValueError('Invalid model origin price.')
        result = dict(signal=forecast['experimental_label'], version=forecast['model_version'],
            threshold=str(threshold), origin_at=origin.isoformat(), origin_price=str(price),
            target_at=(origin + pd.Timedelta(hours=1)).isoformat(), generated_at=generated.isoformat(),
            reason='Experimental classification; probabilities withheld. ' + str(forecast.get('limitation', '')))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result['reason'] = str(exc)
    return result


class PaperAccount:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                cash TEXT NOT NULL, retired_at TEXT);
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), account_id INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY, account_id INTEGER NOT NULL REFERENCES accounts(id),
                status TEXT NOT NULL CHECK(status IN ('OPEN','CLOSED')), opened_at TEXT NOT NULL,
                entry_trade_at TEXT NOT NULL, entry_received_at TEXT NOT NULL, entry_source TEXT NOT NULL,
                budget TEXT NOT NULL, reference_entry TEXT NOT NULL, entry_price TEXT NOT NULL,
                quantity TEXT NOT NULL, entry_fee TEXT NOT NULL, fee_rate TEXT NOT NULL, slippage_rate TEXT NOT NULL,
                signal_json TEXT NOT NULL, observe_hour INTEGER NOT NULL, due_at TEXT NOT NULL,
                closed_at TEXT, exit_trade_at TEXT, exit_received_at TEXT, exit_source TEXT,
                reference_exit TEXT, exit_price TEXT, exit_fee TEXT, proceeds TEXT, net_pnl TEXT,
                observed_at TEXT, observation_checked_at TEXT, observed_price TEXT, observed_return TEXT,
                model_outcome TEXT, model_return TEXT, model_target_price TEXT, model_checked_at TEXT);
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
            ''')
        with self.transaction() as con:
            if not con.execute('SELECT 1 FROM settings WHERE id=1').fetchone():
                account = con.execute('INSERT INTO accounts(created_at,cash) VALUES (?,?)',
                    (pd.Timestamp.now(tz='UTC').isoformat(), str(INITIAL))).lastrowid
                con.execute('INSERT INTO settings VALUES (1,?)', (account,))

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA foreign_keys=ON')
        con.execute('PRAGMA busy_timeout=10000')
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @contextmanager
    def transaction(self):
        with self.connection() as con:
            con.execute('BEGIN IMMEDIATE')
            yield con

    def account(self, con=None):
        if con is not None:
            return dict(con.execute('SELECT a.* FROM accounts a JOIN settings s ON a.id=s.account_id WHERE s.id=1').fetchone())
        with self.connection() as db:
            return self.account(db)

    def trades(self, all_accounts=False):
        with self.connection() as con:
            query = 'SELECT * FROM trades' + ('' if all_accounts else ' WHERE account_id=?') + ' ORDER BY opened_at,id'
            return [dict(row) for row in con.execute(query, () if all_accounts else (self.account(con)['id'],))]

    def _action(self, con, token, payload):
        if not isinstance(token, str) or not token.strip():
            raise ValueError('An action ID is required.')
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        old = con.execute('SELECT * FROM actions WHERE id=?', (token,)).fetchone()
        if old and old['fingerprint'] != fingerprint:
            raise ValueError('Action ID was already used for a different request.')
        return fingerprint, old['result'] if old else None

    def _account_check(self, con, expected):
        account = self.account(con)
        if account['id'] != expected:
            raise ValueError('This virtual account has been reset in another session. Refresh the page.')
        return account

    def open(self, expected_account, token, budget, fee_rate, slippage_rate, quote, signal, observe_hour=True, now=None):
        now = instant(now if now is not None else pd.Timestamp.now(tz='UTC'))
        budget, fee, slip = [decimal(v) for v in (budget, fee_rate, slippage_rate)]
        if budget <= 0 or not 0 <= fee <= Decimal('.05') or not 0 <= slip <= Decimal('.05'):
            raise ValueError('Positive position budget and fee/slippage between 0% and 5% are required.')
        with self.transaction() as con:
            fingerprint, old = self._action(con, token, dict(kind='open', account=expected_account,
                budget=str(budget), fee=str(fee), slip=str(slip), observe=bool(observe_hour)))
            if old:
                return old
            account = self._account_check(con, expected_account)
            price, traded, received = fresh_quote(quote, now)
            cash = decimal(account['cash'])
            if budget > cash:
                raise ValueError('Insufficient virtual USDT balance, including the entry fee.')
            # Budget includes the entry fee. Never borrow virtual money.
            entry = price * (1 + slip)
            notional = budget / (1 + fee)
            entry_fee, quantity = budget - notional, notional / entry
            signal = clean_signal(signal, now)
            trade_id = uuid.uuid4().hex
            con.execute('''INSERT INTO trades (id,account_id,status,opened_at,entry_trade_at,entry_received_at,
                entry_source,budget,reference_entry,entry_price,quantity,entry_fee,fee_rate,slippage_rate,
                signal_json,observe_hour,due_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (trade_id, expected_account, 'OPEN', now.isoformat(), traded.isoformat(), received.isoformat(),
                quote['source'], str(budget), str(price), str(entry), str(quantity), str(entry_fee), str(fee), str(slip),
                json.dumps(signal, sort_keys=True), int(observe_hour), (now + pd.Timedelta(hours=1)).isoformat()))
            con.execute('UPDATE accounts SET cash=? WHERE id=?', (str(cash - budget), expected_account))
            con.execute('INSERT INTO actions VALUES (?,?,?,?)', (token, fingerprint, trade_id, now.isoformat()))
            return trade_id

    def close(self, expected_account, token, trade_id, quote, now=None):
        now = instant(now if now is not None else pd.Timestamp.now(tz='UTC'))
        with self.transaction() as con:
            fingerprint, old = self._action(con, token, dict(kind='close', account=expected_account, trade=trade_id))
            if old:
                return old
            account = self._account_check(con, expected_account)
            row = con.execute('SELECT * FROM trades WHERE id=? AND account_id=?', (trade_id, expected_account)).fetchone()
            if row is None or row['status'] != 'OPEN':
                raise ValueError('The virtual position is missing or already closed.')
            price, traded, received = fresh_quote(quote, now)
            if now < instant(row['opened_at']) or traded < instant(row['entry_trade_at']):
                raise ValueError('Exit time precedes the position entry.')
            fill = price * (1 - decimal(row['slippage_rate']))
            gross = decimal(row['quantity']) * fill
            fee = gross * decimal(row['fee_rate'])
            proceeds, pnl = gross - fee, gross - fee - decimal(row['budget'])
            con.execute('''UPDATE trades SET status='CLOSED',closed_at=?,exit_trade_at=?,exit_received_at=?,exit_source=?,
                reference_exit=?,exit_price=?,exit_fee=?,proceeds=?,net_pnl=? WHERE id=?''',
                (now.isoformat(), traded.isoformat(), received.isoformat(), quote['source'], str(price), str(fill),
                str(fee), str(proceeds), str(pnl), trade_id))
            con.execute('UPDATE accounts SET cash=? WHERE id=?', (str(decimal(account['cash']) + proceeds), expected_account))
            con.execute('INSERT INTO actions VALUES (?,?,?,?)', (token, fingerprint, trade_id, now.isoformat()))
            return trade_id

    def reset(self, expected_account, token, confirmed=False, now=None):
        if confirmed is not True:
            raise ValueError('Confirm the virtual account reset first.')
        now = instant(now if now is not None else pd.Timestamp.now(tz='UTC'))
        with self.transaction() as con:
            fingerprint, old = self._action(con, token, dict(kind='reset', account=expected_account))
            if old:
                return int(old)
            self._account_check(con, expected_account)
            con.execute('UPDATE accounts SET retired_at=? WHERE id=?', (now.isoformat(), expected_account))
            new_id = con.execute('INSERT INTO accounts(created_at,cash) VALUES (?,?)', (now.isoformat(), str(INITIAL))).lastrowid
            con.execute('UPDATE settings SET account_id=? WHERE id=1', (new_id,))
            con.execute('INSERT INTO actions VALUES (?,?,?,?)', (token, fingerprint, str(new_id), now.isoformat()))
            return new_id

    def evaluate(self, hourly, quote=None, now=None):
        """Append matured observations. No automatic close or retroactive fill.

        Classification uses exact model hourly prices; paper observation uses the
        first fresh trade seen at/after entry+1h, recording any evaluation delay.
        """
        now = instant(now if now is not None else pd.Timestamp.now(tz='UTC'))
        with self.transaction() as con:
            account = self.account(con)['id']
            for row in con.execute('SELECT * FROM trades WHERE account_id=?', (account,)).fetchall():
                due = instant(row['due_at'])
                if now < due:
                    continue
                if row['observe_hour'] and row['observed_at'] is None and quote:
                    try:
                        price, at, _ = fresh_quote(quote, now)
                        if at >= due:
                            change = price / decimal(row['reference_entry']) - 1
                            con.execute('''UPDATE trades SET observed_at=?,observation_checked_at=?,observed_price=?,
                                observed_return=? WHERE id=? AND observed_at IS NULL''',
                                (at.isoformat(), now.isoformat(), str(price), str(change), row['id']))
                    except ValueError:
                        pass
                signal = json.loads(row['signal_json'])
                if row['model_outcome'] is not None or signal.get('signal') not in LABELS or hourly is None or hourly.empty:
                    continue
                target = instant(signal['target_at'])
                if target >= now:
                    continue
                dates = pd.to_datetime(hourly.close_time, utc=True)
                bars = hourly[dates.eq(target)]
                if len(bars) != 1:
                    continue
                bar = bars.iloc[0]
                if (target - instant(bar.open_time) != pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
                        or ('closed' in bars and not bool(bar.closed))):
                    continue
                price = decimal(bar.close)
                if price <= 0:
                    continue
                change = price / decimal(signal['origin_price']) - 1
                con.execute('''UPDATE trades SET model_outcome=?,model_return=?,model_target_price=?,model_checked_at=?
                    WHERE id=? AND model_outcome IS NULL''',
                    (outcome(change, signal['threshold']), str(change), str(price), now.isoformat(), row['id']))


def statistics(account, trades, quote=None, now=None):
    """Closed-trade profitability and unique-hour classification are separate."""
    now = instant(now if now is not None else pd.Timestamp.now(tz='UTC'))
    closed = [t for t in trades if t['status'] == 'CLOSED']
    pnls = [decimal(t['net_pnl']) for t in closed]
    fees = sum((decimal(t['entry_fee']) + decimal(t['exit_fee'] or 0) for t in trades), Decimal(0))
    groups = {}
    for row in trades:
        s = json.loads(row['signal_json'])
        if row['model_outcome'] and s.get('signal'):
            key = (s['version'], s['origin_at'])
            groups.setdefault(key, set()).add((s['signal'], row['model_outcome'], s['threshold']))
    scores = [next(iter(v)) for v in groups.values() if len(v) == 1]
    opened = [t for t in trades if t['status'] == 'OPEN']
    equity = decimal(account['cash']) if not opened else None
    try:
        price, _, _ = fresh_quote(quote, now)
        equity = decimal(account['cash']) + sum((decimal(t['quantity']) * price *
            (1 - decimal(t['slippage_rate'])) * (1 - decimal(t['fee_rate'])) for t in opened), Decimal(0))
    except ValueError:
        pass
    return dict(cash=decimal(account['cash']), equity=equity, realized_pnl=sum(pnls, Decimal(0)),
        total_fees=fees, closed=len(closed), open=len(opened),
        win_rate=sum(p > 0 for p in pnls) / len(pnls) if pnls else None,
        average_gain=sum((p for p in pnls if p > 0), Decimal(0)) / sum(p > 0 for p in pnls) if any(p > 0 for p in pnls) else None,
        average_loss=sum((p for p in pnls if p < 0), Decimal(0)) / sum(p < 0 for p in pnls) if any(p < 0 for p in pnls) else None,
        model_samples=len(scores), model_accuracy=sum(p == y for p, y, _ in scores) / len(scores) if scores else None,
        model_conflicts=sum(len(v) > 1 for v in groups.values()))


def balance_curve(account, trades):
    """Cash ledger, not a fabricated continuously marked equity history."""
    events = []
    for row in trades:
        events.append((instant(row['opened_at']), -decimal(row['budget']), 'Entry ' + row['id'][:8]))
        if row['closed_at']:
            events.append((instant(row['closed_at']), decimal(row['proceeds']), 'Exit ' + row['id'][:8]))
    cash, result = INITIAL, [dict(time=instant(account['created_at']), balance=float(INITIAL), event='Account starts')]
    for date, value, name in sorted(events, key=lambda v: v[0]):
        cash += value
        result.append(dict(time=date, balance=float(cash), event=name))
    return pd.DataFrame(result)
