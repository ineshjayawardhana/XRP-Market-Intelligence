# Educational XRP paper simulator

Launch the existing dashboard and select **Paper Trading**:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No new dependencies or exchange credentials are required. This is a local educational simulation of **long XRP spot positions**, using virtual funds only. Every entry and exit is chosen by the user. There is no order execution, account connection, leverage, trade recommendation or change to model validation.

## Account and costs

The initial virtual cash is **100 USDT**. The default position budget is **10 USDT**, including its entry fee. Fee defaults to **0.1% per side** and slippage to **0.05% per side**; both are configurable for each new position and frozen for that position's eventual exit. No virtual borrowing is allowed.

For budget B, fresh reference price P, fee fraction f and slippage fraction s:

- Entry fill = P × (1 + s).
- XRP quantity = [B ÷ (1 + f)] ÷ entry fill.
- Entry fee = B − B ÷ (1 + f); available cash decreases by B.
- Exit fill = fresh exit reference × (1 − s).
- Exit fee = quantity × exit fill × f.
- Cash returned = quantity × exit fill − exit fee.
- Net P/L = cash returned − B, including both fees and both slippage adjustments.

Accounting uses Decimal values stored as text in SQLite, avoiding binary floating-point cash accumulation. Full precision is retained; displayed quantities/money are rounded only for presentation. Available cash, realized net P/L, liquidation-equity estimate, net win rate, average positive/negative P/L, charged fees and cash history are distinct. A break-even position is not counted as a win. Estimated equity is withheld when open positions cannot be valued using fresh data. Exit fees on still-open positions are hypothetical and are excluded from the charged-fee total.

## Freshness

The simulator reuses the public Binance stream and its REST last-trade fallback. WebSocket trades must be at most 15 seconds old, REST trades at most 45 seconds old, and receipt timestamps at most 45 seconds old. Future-dated trades, missing/nonfinite prices and ticker snapshots without actual trade timestamps cannot fill positions. Opening **and closing** are blocked if a fresh quote cannot be obtained. The action checks freshness again, including after any model-artifact verification. The page refreshes every 30 seconds, with separate trade, receipt and check times in SLST.

## Two different one-hour windows

The saved model uses **completed hourly close → next completed hourly close**. A user may open a virtual position mid-hour, so its entry → entry+60 minutes window is different. Mixing them would misstate classification performance.

At entry the ledger freezes the experimental class, model version, original hourly close/price, exact target close, inference timestamp and the actual training-derived threshold returned by verified inference. There is no hardcoded classification boundary, threshold optimization or reused 24-hour target. If hourly inference is unavailable, the position explicitly records “signal unavailable”; a fresh-price-only virtual position is still allowed. Probabilities are never retained in the simulator signal snapshot or displayed.

After entry is at least one hour old, the original class is compared with the **exact** next completed hourly candle, using that position's frozen threshold. Native live hourly history provides the evidence; a bounded, cached public REST request can recover a missing historical target after restart. Forming, missing, duplicated or partial-duration targets remain unscored. A delayed refresh never substitutes a later candle for the original target. Existing evaluation results/thresholds/validation status are untouched.

The optional paper-position observation uses the **first fresh trade observed by the application at or after entry+1h**. Its actual timestamp and delay are stored. If Streamlit was closed, this may be much later than one hour; it is explicitly not an invented exact 60-minute execution or a backfilled fill. Positions remain open until manually closed, even after an observation. Closed positions can still receive their pending model observation.

Classification match rate is calculated per unique **model version + hourly prediction origin**, so multiple positions in the same forecast hour do not multiply the sample size. Conflicting observations are excluded. It is a user-selected, small and temporally dependent educational sample, not independent validation, trading profitability or proof of future performance. It does not unlock prediction percentages.

## Persistence and reset

The default database is `data/paper_trading/paper_trading.sqlite3`. It contains only virtual accounts, positions, observation evidence and idempotent action records. The database is created when the page is first opened. Restarting Streamlit preserves the account. `XRP_PAPER_DB` can select a separate database for demos/tests. A database represents one shared local virtual account; this is not a multi-user authenticated brokerage service.

SQLite transactions serialize competing balance changes. Unique action IDs prevent repeated submissions from charging/opening/closing/resetting twice. Widget action IDs rotate after success, so replayed old submissions cannot trigger a new position. Actions from a stale account after a reset are rejected.

Reset requires checking the explicit confirmation box. It archives the old account and starts a new 100-USDT account; it never invents exit prices for archived open positions. History remains accessible using **Include archived accounts** and the CSV download. Performance summaries apply to the current account only. Existing datasets, models, news labels and historical research are not written by this ledger.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Fixtures cover accounting, fees/slippage, loss/profit statistics, persistence, concurrent actions, retry idempotency, reset protection, UTC/SLST handling, stale/future data, exact model horizons, independent paper windows, duplicate-hour scoring, delayed observations, public rate-limit cooldown and the Streamlit entry/exit/reset flows. They never create real trades or claim model accuracy from invented market observations.

Verification on 16 November 2026: **248 tests passed**, including 21 new simulator tests, with zero failures/errors/skips. SHA256 checks confirmed **all 139 protected data/model/report files unchanged**. Source backups and the original manifest are under `data/backups/paper_trading_20261116T143406/`.

A separate temporary public-Binance integration check recorded the existing `xrp1h_20261116T072805203839Z` experimental signal and its actual threshold, opened/closed a virtual position with fees/slippage, and confirmed cash conservation and absent stored probabilities. Its future outcome remained pending. The temporary database was removed, so no demo trades were inserted into the user's persistent account. Delayed one-hour behavior is tested using explicitly identified fixtures rather than a fabricated live one-hour result.

This is for education only, not financial advice. Crypto is high risk.
