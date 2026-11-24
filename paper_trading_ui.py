"""Educational virtual spot UI. All ledger writes stay in a separate SQLite DB."""
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_market_ui import STRETCH, stream_snapshot
from market_intelligence import plot_times, slst
from paper_trading import PaperAccount, balance_curve, capture_signal, decimal, fresh_quote, statistics
from streaming_market import model_candles


@st.cache_resource
def evidence_client():
    from paper_market_data import PublicHourlyEvidence
    return PublicHourlyEvidence()


@st.cache_data(ttl=60, show_spinner=False)
def historical_target(target):
    # Cache failed requests too, so a rerun doesn't hammer an unavailable source.
    try:
        return evidence_client().fetch(target), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@st.cache_data(ttl=55, show_spinner=False)
def paper_hourly_archive(root):
    from dashboard_data import market_component
    return market_component('candles', Path(root))['data']


def intent(kind):
    key = 'paper_intent_' + kind
    if key not in st.session_state:
        st.session_state[key] = uuid.uuid4().hex
    return st.session_state[key]


def finish(kind, message):
    st.session_state['paper_intent_' + kind] = uuid.uuid4().hex
    st.session_state['paper_message'] = message
    st.rerun(scope='app')


def settle_observations(ledger, hourly, quote, now):
    ledger.evaluate(hourly, quote, now)
    warnings = []
    targets = set()
    available = set(pd.to_datetime(hourly.close_time, utc=True)) if not hourly.empty else set()
    for trade in ledger.trades():
        signal = json.loads(trade['signal_json'])
        if signal.get('signal') and trade['model_outcome'] is None and pd.Timestamp(trade['due_at']) <= now:
            target = pd.Timestamp(signal['target_at'])
            if target not in available:
                targets.add(target.isoformat())
    # Only delayed evidence missing from the live bootstrap needs network reads.
    for target in sorted(targets)[:4]:
        frame, error = historical_target(target)
        if error:
            warnings.append(error)
        else:
            ledger.evaluate(frame, quote, now)
    if len(targets) > 4:
        warnings.append('More historical targets remain pending; at most four are retrieved per refresh.')
    return warnings


def render_paper_trading(root, theme='Dark'):
    st.header('Paper Trading')
    st.info('Educational simulator · virtual USDT and XRP only. No real exchange accounts, orders or credentials are used. Signals are experimental; predictive probabilities remain hidden.')
    st.caption('Long XRP spot positions only. You choose every virtual entry and exit; there is no automatic closing or trading recommendation.')
    if 'paper_message' in st.session_state:
        st.success(st.session_state.pop('paper_message'))
    path = Path(os.getenv('XRP_PAPER_DB', str(Path(root) / 'data' / 'paper_trading' / 'paper_trading.sqlite3')))
    try:
        ledger = PaperAccount(path)
        snapshot = stream_snapshot(root)
        quote = snapshot.get('trade')
        now = pd.Timestamp.now(tz='UTC')
        hourly = model_candles(None, snapshot, now)
        issues = settle_observations(ledger, hourly, quote, now)
        account, trades = ledger.account(), ledger.trades()
        stats = statistics(account, trades, quote, now)
    except Exception as exc:
        st.error('The local virtual account is unavailable: ' + str(exc))
        return
    try:
        price, _, _ = fresh_quote(quote, now)
        market_ok = True
        st.metric('XRP/USDT public trade price', f'${price:,.5f}')
        st.caption(f'{quote["source"]} · Trade: {slst(quote["trade_at"])} SLST · Received: {slst(quote["received_at"])} SLST · Checked: {slst(now)} SLST')
    except ValueError as exc:
        market_ok = False
        st.warning(str(exc) + ' Opening and closing virtual positions require fresh trade data.')
    if not snapshot.get('connected'):
        st.caption('WebSocket disconnected or waiting. Fresh public REST last-trade data is accepted when available; a rolling ticker alone is insufficient.')
    st.caption('Display refresh: 30 seconds. Freshness is checked again for each action. All displayed timestamps use Asia/Colombo; the ledger stores UTC.')
    for column, label, value in zip(st.columns(4), ['Available virtual USDT', 'Estimated liquidation equity', 'Realized net P/L', 'Total simulated fees'],
            [stats['cash'], stats['equity'], stats['realized_pnl'], stats['total_fees']]):
        column.metric(label, f'{value:,.6f}' if value is not None else 'Unavailable / stale')
    st.caption('Equity includes open XRP valued at a fresh reference price, less the stored exit slippage and fee assumptions. Available cash excludes the entry budget. Open-position fees are already included in total fees.')

    st.subheader('Open a virtual position')
    token = intent('open')
    with st.form('paper_open_form_' + token):
        amount = st.number_input('Position budget · virtual USDT (entry fee included)', min_value=.01, value=10., step=1., key='paper_amount')
        fee = st.number_input('Simulated fee per side (%)', min_value=0., max_value=5., value=.1, step=.01, format='%.3f', key='paper_fee')
        slip = st.number_input('Simulated slippage per side (%)', min_value=0., max_value=5., value=.05, step=.01, format='%.3f', key='paper_slippage')
        observe = st.checkbox('Record a one-hour paper-position observation (keep position open)', value=True, key='paper_observe')
        submitted = st.form_submit_button('Open virtual XRP position', disabled=not market_ok, key='paper_open_' + token)
    if submitted:
        try:
            # Read again after submit, then use the original model on hourly data only.
            latest = stream_snapshot(root)
            complete = model_candles(paper_hourly_archive(str(root)), latest)
            from one_hour_trend import live_forecast
            forecast = live_forecast(complete, root=Path(root))
            action_time = pd.Timestamp.now(tz='UTC')
            signal = capture_signal(forecast, complete, action_time)
            # Artifact verification can take seconds; obtain the fill quote last.
            latest = stream_snapshot(root)
            trade_id = ledger.open(account['id'], token, amount, decimal(fee) / 100,
                decimal(slip) / 100, latest.get('trade'), signal, observe,
                now=pd.Timestamp.now(tz='UTC'))
            finish('open', 'Virtual position opened: ' + trade_id[:8] + '. No real funds were used.')
        except Exception as exc:
            st.error('Virtual entry was not recorded: ' + str(exc))

    st.subheader('Open virtual positions')
    opened = [t for t in trades if t['status'] == 'OPEN']
    if opened:
        display = []
        for t in opened:
            s = json.loads(t['signal_json'])
            display.append(dict(position=t['id'][:8], opened_slst=slst(t['opened_at']), quantity_xrp=float(decimal(t['quantity'])),
                entry_fill=float(decimal(t['entry_price'])), budget_usdt=float(decimal(t['budget'])),
                recorded_signal=s.get('signal') or 'Unavailable at entry', observation_due_slst=slst(t['due_at'])))
        st.dataframe(pd.DataFrame(display), hide_index=True, **STRETCH)
        selected = st.selectbox('Virtual position to close', [t['id'] for t in opened],
            format_func=lambda i: i[:8], key='paper_close_position')
        close_token = intent('close')
        if st.button('Close selected virtual position', key='paper_close_' + close_token, disabled=not market_ok):
            try:
                latest = stream_snapshot(root)
                ledger.close(account['id'], close_token, selected, latest.get('trade'))
                finish('close', 'Virtual position closed. Simulated exit fees and slippage were deducted.')
            except Exception as exc:
                st.error('Virtual exit was not recorded: ' + str(exc))
    else:
        st.caption('No open virtual positions.')

    st.subheader('Profitability · virtual positions')
    for column, label, value in zip(st.columns(4), ['Closed virtual positions', 'Net winning positions', 'Average net gain · USDT', 'Average net loss · USDT'],
            [str(stats['closed']), f'{stats["win_rate"]:.1%}' if stats['win_rate'] is not None else 'No closed positions',
             f'{stats["average_gain"]:.6f}' if stats['average_gain'] is not None else 'No gains',
             f'{stats["average_loss"]:.6f}' if stats['average_loss'] is not None else 'No losses']):
        column.metric(label, value)
    curve = balance_curve(account, trades)
    fig = go.Figure(go.Scatter(x=plot_times(curve.time), y=curve.balance, mode='lines+markers', text=curve.event,
        line=dict(shape='hv'), name='Available virtual cash'))
    fig.update_layout(template='plotly_dark' if theme == 'Dark' else 'plotly_white', height=280,
        title='Virtual cash balance · deposits into positions reduce available cash', xaxis_title='Sri Lanka time · SLST',
        yaxis_title='Virtual USDT', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, key='paper_balance_chart', **STRETCH)
    st.caption('This is an event-based cash ledger, not a continuous portfolio-equity or investment-return curve. Net P/L includes both fees and slippage; classification correctness can still accompany a losing position.')

    st.subheader('Model observation · separate from profitability')
    st.write('Recorded hourly classification match rate: **' +
        (f'{stats["model_accuracy"]:.1%}' if stats['model_accuracy'] is not None else 'Not yet available') +
        f'** · {stats["model_samples"]} unique matured model-hour observations.')
    st.caption('Each recorded experimental signal is compared with the exact next completed hourly close using its own saved training-derived threshold. Multiple positions in the same model hour count once. This user-selected, small and dependent sample does not change the model’s historical evaluation or validation status.')
    st.caption('Scoring starts after the paper entry is at least one hour old, even though the model target may mature earlier. Optional paper observations instead use the first fresh trade seen at or after entry+1h, with the actual time and delay recorded. Opening this page does not backdate fills or auto-close positions; observations resume when the app is open again.')
    if stats['model_conflicts']:
        st.warning('Conflicting model-hour observations were excluded from the classification match rate.')
    for issue in issues:
        st.caption('Historical observation pending: ' + issue)

    st.subheader('Virtual trade history')
    archived = st.checkbox('Include archived accounts', value=False, key='paper_archived')
    rows = ledger.trades(all_accounts=archived)
    history = []
    for t in rows:
        s = json.loads(t['signal_json'])
        history.append(dict(account=t['account_id'], position=t['id'],
            status=t['status'] + (' (archived)' if t['account_id'] != account['id'] else ''),
            opened_slst=slst(t['opened_at']), closed_slst=slst(t['closed_at']),
            quantity_xrp=float(decimal(t['quantity'])), reference_entry=float(decimal(t['reference_entry'])),
            budget_usdt=float(decimal(t['budget'])), fee_pct=float(decimal(t['fee_rate'])*100),
            slippage_pct=float(decimal(t['slippage_rate'])*100),
            entry_fill=float(decimal(t['entry_price'])), exit_fill=float(decimal(t['exit_price'])) if t['exit_price'] else None,
            reference_exit=float(decimal(t['reference_exit'])) if t['reference_exit'] else None,
            entry_fee=float(decimal(t['entry_fee'])), exit_fee=float(decimal(t['exit_fee'])) if t['exit_fee'] else None,
            net_pnl=float(decimal(t['net_pnl'])) if t['net_pnl'] else None,
            source=t['entry_source'], entry_trade_slst=slst(t['entry_trade_at']),
            entry_received_slst=slst(t['entry_received_at']), exit_source=t['exit_source'],
            exit_trade_slst=slst(t['exit_trade_at']), exit_received_slst=slst(t['exit_received_at']),
            signal=s.get('signal') or 'Unavailable', model_version=s.get('version'),
            threshold_pct=float(decimal(s['threshold']) * 100) if s.get('threshold') else None,
            model_origin_slst=slst(s.get('origin_at')), model_target_slst=slst(s.get('target_at')),
            actual_model_outcome=t['model_outcome'] or ('Pending' if s.get('signal') else 'No entry signal'),
            actual_model_return_pct=float(decimal(t['model_return'])*100) if t['model_return'] else None,
            actual_model_target_price=float(decimal(t['model_target_price'])) if t['model_target_price'] else None,
            model_scored_slst=slst(t['model_checked_at']),
            paper_observed_slst=slst(t['observed_at']), paper_return_pct=float(decimal(t['observed_return']) * 100) if t['observed_return'] else None,
            paper_delay_seconds=(pd.Timestamp(t['observed_at'])-pd.Timestamp(t['due_at'])).total_seconds() if t['observed_at'] else None,
            signal_note=s.get('reason')))
    if history:
        table = pd.DataFrame(history)
        st.dataframe(table[['position','status','opened_slst','closed_slst','budget_usdt','quantity_xrp',
            'entry_fill','exit_fill','net_pnl','signal','actual_model_outcome']], hide_index=True, **STRETCH)
        with st.expander('Full virtual trade evidence · fees, timestamps and model windows'):
            st.dataframe(table, hide_index=True, **STRETCH)
        st.download_button('Download virtual history CSV', table.to_csv(index=False).encode(),
            file_name='xrp_virtual_history.csv', mime='text/csv')
    else:
        st.caption('No virtual trades yet. The account starts with 100 virtual USDT.')
    with st.expander('Reset virtual account'):
        st.write('Archive this account, including any open positions, and start a new account with 100 virtual USDT. Existing history is retained; archived open positions are not assigned invented closing prices.')
        confirmed = st.checkbox('I confirm the virtual account reset', key='paper_reset_confirm_' + str(account['id']))
        reset_token = intent('reset')
        if st.button('Reset to 100 virtual USDT', disabled=not confirmed, key='paper_reset_' + reset_token):
            try:
                ledger.reset(account['id'], reset_token, confirmed=confirmed)
                finish('reset', 'A new virtual account with 100 USDT is ready. Prior history remains archived.')
            except Exception as exc:
                st.error('Virtual account reset failed: ' + str(exc))
    st.caption('Stored locally: ' + str(path) + '. This simulator uses one shared virtual account per database; it is intended for a local academic dashboard.')
