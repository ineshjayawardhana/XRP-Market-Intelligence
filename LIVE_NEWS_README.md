# Live-news collection and freshness

Restart the existing dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The sidebar **Poll public news feeds** switch starts/pauses a process-lifetime background worker. It checks permitted sources about every five minutes while the Streamlit process remains running, independently of the selected page or browser reruns. A process restart or machine shutdown stops the worker. The normal UI fragment rereads news every 30 seconds. **Refresh data now** queues an actual collection on every page; when automatic polling is paused, a manual request still performs one collection. It never bypasses robots, Retry-After, provider backoff or the manual repeat throttle.

The worker starts after the dashboard is opened once with polling enabled, then continues without requiring further browser reruns until paused or the Streamlit process stops.

## Findings and actual source results

The old live path used only Cointelegraph's XRP feed and Daily Hodl's XRP feed, both returning HTTP 200 with old article content. At verification their latest publications were 27 October 2026 and 26 January 2026 respectively. The newer 15 November headlines visible in the app came from the historical corpus, not fresh live downloads. Current response/check/Last-Modified timestamps had been mistaken for evidence of fresh publications.

The legacy Google News/CryptoPanic functions were not called by this dashboard. A 300-second Streamlit cache was not a background timer and only triggered collection when Overview or News Intelligence reran. Clearing the cache did not independently fetch news on another page. The old JSON cache capped saved stories at 500 and did not expose per-source publication freshness, empty topic results or individual request failures. Publication sorting existed, but there was no source diversity sufficient to find the newer stories or a prominent latest-publication stale warning.

The new collector tests the existing project's public publisher/topic feeds plus Google RSS. Actual retrieval found November 17 articles through CoinDesk, CryptoSlate and U.Today. Google RSS is **ROBOTS_DISALLOWED** for this collector, and the configured XRPL and Ripple feed paths return **404**. These restrictions remain visible; there is no alternate-host, browser-impersonation or URL-decoding bypass. Other reachable feeds return older XRP articles or no XRP matches. See the [full point-in-time source audit](reports/live_news/audit.md) and `tests/live_news_source_probe.json`.

## Persistence and clocks

New articles and monitoring state live in `data/live_news/news.sqlite3`, with permitted-access/cooldown state in `data/live_news/access.sqlite3`. The previous JSON cache is migrated once, without deleting or rewriting it. The original news SQLite database remains read-only and is merged with historical research. Live rows are not capped or pruned. The historical datasets, trained models, annotations, SKIP/reject decisions and paper-trading database remain unchanged.

Canonical URL and normalized headline checks prevent duplicate stories, including across sources. First-seen timestamps remain immutable; collection/last-seen times advance only when a story is actually seen in a response. An HTTP 304 updates source fetch status without inventing a newly collected article. Publication times come only from explicit source publication fields with timezones; missing dates stay missing. Future-dated reports remain flagged and cannot become the latest current story. Source-reported publication is not independent factual verification.

Historical human labels have display priority, and skipped/rejected articles cannot reappear as fresh AI copies. Publication metadata is enriched only for the same canonical URL, never by borrowing another publisher's syndication date. Duplicate metadata cannot roll the displayed collection clock backward. Read-only SQLite handles are explicitly closed, including on Windows.

## Classification

No models are trained. The available saved TF-IDF model is a synthetic-training demonstration with **three** classes; it cannot produce Neutral_Market_News by itself. The existing `research_content_rubric_v2` supplies conservative four-category classifications from headlines and available unverified feed excerpts. The saved model's output is auxiliary evidence, clearly separate from the four-category result. A configured saved model that actually supports all four categories can supply a direct uncalibrated classifier result.

New rows record backend, model artifact/hash/provenance, score type, reasoning and **AI_PROVISIONAL** verification status. Ambiguous evidence remains REVIEW_REQUIRED; no article content or factual verification is invented. Human annotations and historical AI classifications remain separate. Scores are heuristic or uncalibrated classifier scores, not probabilities of truth or price direction. Existing model evaluation and validation statuses do not change.

## Monitoring and verification

**News Source Status** displays each source's last successful fetch, returned/XRP counts, newly saved count, latest publication, HTTP status, error and next allowed attempt. A successful fetch with old content is explicitly labeled old. The latest dated article has an age indicator; an article older than 24 hours produces a stale warning. Cards/table show publication, first-seen and collection times in Asia/Colombo, source/link, classification status, backend and provenance.

Commands:

```powershell
.\.venv\Scripts\python.exe verify_live_news.py --refresh
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The first command performs a real permitted collection and writes `reports/live_news/audit.json` and `audit.md`. A repeat run should add no duplicate stories. `probe_news_sources.py` is an explicit diagnostic of the public feed endpoints, retaining host access restrictions and cooldowns.

Verified on 17 November 2026: **262 tests passed**, zero failures/errors/skips. The actual retrieved November 17 CryptoSlate headline was present in the Streamlit news table, and all 14 source-status rows and existing dashboard pages loaded without errors. **140 protected data/model/report files remained unchanged**, including the paper-trading database. Backups and SQLite snapshots are under `data/backups/live_news_20261117T152448/`. No test trades were inserted into the user's paper account.

This is for education only, not financial advice. Crypto is high risk.
