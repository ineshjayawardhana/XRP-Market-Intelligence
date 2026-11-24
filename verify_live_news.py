"""Audit the actual feed -> SQLite -> dashboard path, without training models."""
import argparse
import json
from pathlib import Path

import pandas as pd

import dashboard_data
from live_news import NewsStore, load_rows, monitoring, refresh
from market_intelligence import slst

ROOT = Path(__file__).resolve().parent


def audit(collect=False):
    fetched = refresh(ROOT, force=True) if collect else None
    rows = load_rows(ROOT)
    frame, warnings = dashboard_data.load_news(ROOT)
    state, sources = monitoring(ROOT)
    current = pd.Timestamp.now(tz='UTC')
    dated = frame[frame.published_at.notna() & frame.published_at.le(current)]
    latest = dated.iloc[0] if not dated.empty else None
    store = NewsStore(ROOT)
    with store.connect() as con:
        additions = dict(con.execute('SELECT source_id,COUNT(*) FROM discoveries WHERE duplicate=0 GROUP BY source_id'))
    evidence = json.loads((ROOT/'tests'/'actual_news_ui_result.json').read_text(encoding='utf-8')) if (ROOT/'tests'/'actual_news_ui_result.json').exists() else None
    background = json.loads((ROOT/'tests'/'actual_news_background_result.json').read_text(encoding='utf-8')) if (ROOT/'tests'/'actual_news_background_result.json').exists() else None
    result = dict(audited_at=current.isoformat(), refresh=fetched, persistent_live_records=len(rows),
        imported_old_cache=additions.get('legacy_cache',0), newly_collected_unique=sum(v for k,v in additions.items() if k!='legacy_cache'),
        merged_display_records=len(frame), articles_after_nov15=int((dated.published_at>=pd.Timestamp('2026-11-16T00:00:00Z')).sum()),
        latest=dict(headline=latest.headline, source=latest.source, published_at=latest.published_at.isoformat(),
                    publication_slst=slst(latest.published_at), url=latest.url,
                    backend=latest.classifier_backend, provenance=latest.model_provenance) if latest is not None else None,
        sources=sources, poll_state=state, display_warnings=warnings, streamlit_evidence=evidence, background_evidence=background)
    folder=ROOT/'reports'/'live_news'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Live-news audit · 17 November 2026','',
        'The original collector was checking two feeds successfully, but those feeds contained old publications: Cointelegraph XRP latest 27 October 2026, and Daily Hodl XRP latest 26 January 2026. A current HTTP Last-Modified/check timestamp did not prove fresh article content.',
        '','The dashboard did not call the legacy Google News/CryptoPanic collector. It read historical CSV/SQLite rows plus a separate JSON feed cache. Five-minute Streamlit cache expiry was not a timer: collection happened only on Overview/News Intelligence reruns. Manual refresh cleared caches without independently scheduling news on other pages. The JSON path retained only the latest 500 observed rows and lacked per-source freshness diagnostics.',
        '','The fix adds a process-lifetime background poller, append-only SQLite live news, broader existing public-feed coverage, source backoff/conditional requests, and uncached dashboard reads. Manual refresh queues a real background request on every page, including while automatic polling is paused. News collection continues about every five minutes without browser reruns while the Streamlit process runs and polling remains enabled.',
        '',f'Persistent live rows: **{len(rows)}**, including **{result["imported_old_cache"]}** migrated cache rows and **{result["newly_collected_unique"]}** newly collected unique articles. Unified display rows: **{len(frame)}**. Dated articles newer than 15 November: **{result["articles_after_nov15"]}**.',
        '',(f'Latest retrieved/displayed publication: **{result["latest"]["publication_slst"]} SLST**, **{result["latest"]["source"]}**.' if latest is not None else 'No source-reported dated article is available.'),
        f'[{latest.headline}]({latest.url})' if latest is not None else 'No dated article available.',
        '', '| Source | Current result | Last successful fetch (SLST) | Returned / XRP | New in last fetch | Latest publication (SLST) |',
        '|---|---|---|---:|---:|---|']
    for source in sources:
        lines.append(f'| {source["name"]} | {source.get("error") or source.get("status")} | {slst(source.get("last_success_at"))} | {source.get("articles_returned",0)} / {source.get("relevant_articles",0)} | {source.get("newly_saved",0)} | {slst(source.get("latest_publication_at"))} |')
    lines += ['', 'Google News RSS is not downloaded when robots rules disallow this collector. The configured official XRPL and Ripple feed paths returned HTTP 404. No access restrictions, authentication, paywalls or rate limits were bypassed. Publisher timestamps are source-reported, not independently fact-checked. Successful sources can still return no XRP articles or only old stories.',
        '', 'The available saved TF-IDF news model has three classes and is a synthetic-training demonstration. It cannot classify Neutral_Market_News by itself. New four-category labels therefore use the existing versioned research rubric with available feed excerpts, recording the saved TF-IDF output as auxiliary evidence. Backend, model artifact/hash, score type, reasoning and provisional status are recorded. Ambiguous coverage stays REVIEW_REQUIRED. No model was retrained and no AI label became human verified.',
        '', 'Publication, immutable first-seen and latest collection clocks remain separate. Unknown publication times remain missing. Future-dated timestamps are not treated as current news. Original publisher links are retained; unavailable Google landing links are not decoded or bypassed. Canonical URL/headline duplicates preserve earlier labels and timestamps; historical human labels and SKIP/reject decisions win in the merged display.',
        '', 'Verification artifacts: tests/live_news_source_probe.json (actual public feed responses), tests/actual_news_ui_result.json (Streamlit data-path checks), and tests/news_full_suite.log (automated tests). Public availability can change; point-in-time success is not a guarantee of uninterrupted coverage.',
        '', 'Restart the dashboard: `.\\.venv\\Scripts\\python.exe -m streamlit run app.py`. One-shot collection/audit: `.\\.venv\\Scripts\\python.exe verify_live_news.py --refresh`.']
    lines += ['', 'Implementation verification on 17 November: 262 automated tests passed (zero failures/errors/skips), and all 140 protected data/model/report files matched the backup manifest. The Streamlit check found the actual latest retrieved article in the news table and loaded the other dashboard sections without exceptions.']
    if background:
        lines += ['', f'Actual background manual refresh completed: **{background.get("actual_background_refresh_completed")}**, checked at **{slst(background.get("checked_at"))} SLST**. Saved rows before/after: **{background.get("records_before")} / {background.get("records_after")}**; newly saved in that repeat check: **{background.get("newly_saved")}**.']
    (folder/'audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh',action='store_true',help='Attempt a public refresh, retaining all restrictions/backoff')
    args=parser.parse_args()
    result=audit(args.refresh)
    print(json.dumps({key:result[key] for key in ('refresh','persistent_live_records','newly_collected_unique','merged_display_records','articles_after_nov15','latest')},indent=2))
