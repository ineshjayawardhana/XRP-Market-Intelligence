# Multi-source real XRP news collection

Run from the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe collect_multisource_news.py --target 400 --days 84 --max-requests 65
```

Run the **same command to resume**. The target is the total corpus size, including
the existing 53 records. No model is trained. The existing Streamlit app, technical
analysis, price history, GDELT collector, original datasets and AI review files
remain unchanged. Before collection, existing root CSV/JSON files are backed up
and hash-checked in `data/backups/`. Outputs live in `data/multisource/`.

## Sources and public access

The collector supports GDELT, Google News RSS, CoinDesk, Cointelegraph, Decrypt,
CryptoSlate, NewsBTC, Bitcoinist, CryptoPotato, The Daily Hodl, The Crypto Basic,
U.Today, and attempted official Ripple/XRPL feed endpoints. Publisher topic feeds
are used where available. Standard public WordPress RSS page queries are bounded
to 20 pages; empty, repeating, nonexistent or already-collected pages stop further
pagination. No private API or article-body archive is used.

Every endpoint is checked against its host's robots rules. There is no browser
impersonation, authentication, proxy rotation, Google-link decoding, paywall bypass
or access-block bypass. Normal feed redirects get their own robots checks. Only
headline metadata and at most 180 words of a supplied feed summary are retained;
articles are not downloaded for this collection.

Source references: [CoinDesk's RSS announcement](https://www.coindesk.com/coindesk-news/2021/09/17/coindesk-rss),
[NewsBTC's invitation to use its public feed](https://www.newsbtc.com/press/), and
[GDELT's DOC API documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/).
RSS access does not constitute factual verification or a blanket redistribution
license. Publisher attribution and original links remain in every record.

During the live run, Google News RSS was blocked by robots policy. GDELT timed out
and returned 429, so its host entered cooldown. The attempted official feeds at
`xrpl.org/blog/rss.xml` and `ripple.com/insights/feed/` returned 404. The Crypto Basic
topic feed had no matching entries and later public pages returned 404. Some
publishers later timed out or returned 429; their already collected items were
retained. Diagnostics are recorded, not treated as successful connections. Other
permitted publisher feeds supplied enough records to reach the target.

## Incremental and safe operation

- `collection.sqlite3` is the collection authority. Each article, source observation,
  feed/window checkpoint, pending item queue, ETag, Last-Modified, and host cooldown
  is persisted. Queue positions and inserted articles commit together, so interrupted
  runs do not lose already fetched items or create duplicates.
- A completed target performs no HTTP requests. Completed historical search windows
  are cached; feed refreshes have a one-hour TTL and conditional HTTP headers where
  supported. GDELT uses stable calendar-week windows within its recent search range.
  `--days` controls discovery search windows; publisher feeds may supply older items.
- Requests are bounded by `--max-requests`, including robots and redirects. GDELT
  requests are at least 15 seconds apart, other requests at least five seconds apart,
  with larger declared crawl delays respected. Access/rate-limit responses persist
  host cooldowns of at least 15 minutes, honor longer Retry-After values, and do not
  cause immediate retries. Transport timeouts also pause that host. Failed jobs
  have a one-hour retry delay. The original standalone GDELT collector is preserved.
- Exact canonical URLs, normalized/publisher-stripped titles and near-identical
  headline token sets identify duplicate discoveries. Tracking parameters are
  removed; content-identifying parameters are retained. Differing numeric details
  prevent similarity-only merging. The first record and any human decision win.
- All 53 legacy records remain, including their original dates, labels, URLs and
  SKIP decisions. A pre-existing syndicated pair is flagged rather than deleted.
  New duplicates are rejected with an observation audit. Publication metadata is
  enriched only for the same canonical URL, not from a different publisher's copy.
- A lock prevents two collection/review processes from writing the same workspace.
  Keep the SQLite database and workspace when resuming. Never delete an active lock;
  after a crash, confirm no process is running before removing a stale lock.

## Outputs

| File in data/multisource/ | Purpose |
| --- | --- |
| xrp_news_master.csv | Expanded corpus, sources, original URLs, timestamps and provenance |
| xrp_real_labeled.csv | Compatible working copy: 53 legacy rows plus new blank human labels |
| xrp_auto_labeled.csv | AI_PROVISIONAL classifications/reasons or REVIEW_REQUIRED |
| xrp_batch_review.csv / xrp_batch_report.md | Flagged classifications and batch report |
| xrp_batch_decisions_template.csv | Optional multi-record human review |
| data_quality_report.json / .md | Source/publisher distribution, duplicates, timestamp quality, category counts and failures |
| collection.sqlite3 | Resumable observations, metadata and connection checkpoints |

The original root AI suggestions, proposals, human ledger and context are copied
into the new workspace. Later collection runs merge additional root records without
replacing decisions made in the workspace. Conflicting human decisions fail closed.
Original root review files are never edited by this collector.

The normal batch classifier runs automatically in the new workspace. It uses saved
assessments and publisher feed summaries through the existing conservative rules;
it makes no external LLM call. Google RSS descriptions are treated as aggregator
snippets, not article content. Missing or ambiguous context remains REVIEW_REQUIRED.
All automatic results stay AI_PROVISIONAL, with heuristic rather than calibrated
confidence. No automatic classification is written into a human `label` field.

The completed collection contains **400 records: 53 preserved originals plus
347 additions**, with 39 duplicate observations rejected. One duplicate pair was
already present in the protected legacy data, so there are 399 detected distinct
article groups. New primary discoveries are: The Daily Hodl 127, NewsBTC 78,
Bitcoinist 53, U.Today 37, CryptoPotato 36, Cointelegraph 15, and CoinDesk 1.
There are 348 source-reported publication timestamps and 52 missing ones; all
missing publication dates belong to preserved legacy records. Reported publication
times span May 2024 through November 2026 because some public feeds include older
archive entries. This does not backdate their new project observation timestamps.

The automatic file covers 390 rows without a source CSV category/SKIP label:
2 adoption, 1 speculation, 8 negative, 35 neutral and 344 REVIEW_REQUIRED.
These include an independent prediction for the previously accepted ledger item;
AI and human counts should not be added together as disjoint populations.
All 66 automated tests passed after the final changes, and a completed-target
resume was verified with zero HTTP requests. Preparation still reports only
8 human annotations and refuses training readiness.

Optional commands for the larger corpus:

```powershell
.\.venv\Scripts\python.exe ai_labeling.py batch --data-dir data/multisource
.\.venv\Scripts\python.exe ai_labeling.py batch-review --data-dir data/multisource --decisions-file data/my_batch_decisions.csv --reviewer "Inesh"
.\.venv\Scripts\python.exe prepare_real_training.py --data-dir data/multisource
```

Preparation does not retrain. The original training command still uses the original
root data; the larger workspace is not silently substituted into an existing model.

## Timestamp and evaluation safeguards

- `published_at` is an explicitly time-zoned, source-reported publication timestamp.
  Missing/date-only/invalid values stay blank and the raw value/status is retained.
  Atom `updated` is not substituted for publication. Future source dates are flagged.
- `gdelt_first_seen_at` is separate. Neither a GDELT seen date nor retrieval time is
  invented as a publication date. Original legacy GDELT dates remain unchanged.
- For **new** records, `date` and `availability_at` equal when this project retrieved
  the metadata. `timestamp_type=PROJECT_OBSERVED_AT`. Newly collected old headlines
  cannot be silently backdated into historical training or backtests. Their
  `historical_eligible=False` records this limitation. Legacy first-seen timestamps
  retain the old project's explicit publication-time caveat.
- Current feed summaries are not proven original article snapshots and are not
  model features. Preparation uses headlines only, groups detected duplicates,
  applies chronological embargoes, and permits only human labels in evaluation.
- Hundreds of collected headlines do not equal hundreds of verified labels. The
  original minimum human-label and per-partition class requirements still apply.
  No accuracy or trading-performance claim is calculated from this collection.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The collection tests use local XML/JSON fixtures and mocked HTTP. They cover RSS
and Atom, publication/observation separation, missing and future dates, GDELT
provenance, canonical/title duplicates, preservation of legacy human/SKIP values,
transactional queues, zero-download resume, conditional headers, robots policies,
redirect checks, request budgets, timeout/429 cooldowns and multi-source review
preservation. Live source failures remain in the separate quality report.
