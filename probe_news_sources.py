"""Read permitted public feeds and report real response/publication timestamps."""
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlencode

from ai_labeling import now
from collect_multisource_news import PublicClient, Store, parse_feed

ROOT = Path(__file__).resolve().parent
JOBS = [
    ('cointelegraph_xrp','Cointelegraph','https://cointelegraph.com/rss/tag/xrp'),
    ('dailyhodl_xrp','The Daily Hodl','https://dailyhodl.com/ripple-and-xrp/feed/'),
    ('google_news','Google News','https://news.google.com/rss/search?' + urlencode(dict(q='XRP OR "XRP Ledger" when:3d',hl='en-US',gl='US',ceid='US:en'))),
    ('cointelegraph','Cointelegraph','https://cointelegraph.com/rss'),
    ('coindesk','CoinDesk','https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml'),
    ('decrypt','Decrypt','https://decrypt.co/feed'),
    ('newsbtc_xrp','NewsBTC','https://www.newsbtc.com/tag/xrp/feed/'),
    ('cryptobasic_xrp','The Crypto Basic','https://thecryptobasic.com/category/ripple/feed/'),
    ('xrpl','XRP Ledger','https://xrpl.org/blog/rss.xml'),
    ('ripple','Ripple','https://ripple.com/insights/feed/'),
    ('cryptoslate','CryptoSlate','https://cryptoslate.com/feed/'),
    ('bitcoinist_xrp','Bitcoinist','https://bitcoinist.com/tag/xrp/feed/'),
    ('cryptopotato_xrp','CryptoPotato','https://cryptopotato.com/tag/ripple/feed/'),
    ('utoday','U.Today','https://u.today/rss.php'),
]


def probe(jobs=None, append=False):
    folder = ROOT / 'data' / 'live_news'
    folder.mkdir(exist_ok=True)
    path = folder / 'access.sqlite3'
    prior = ROOT / 'data' / 'dashboard_cache' / 'public_feed_access.sqlite3'
    if not path.exists() and prior.exists():
        with sqlite3.connect(prior) as src, sqlite3.connect(path) as dst:
            src.backup(dst)
    store = Store(path)
    client = PublicClient(store, budget=45)
    prior_results = ROOT/'tests'/'live_news_source_probe.json'
    results = json.loads(prior_results.read_text(encoding='utf-8')) if append and prior_results.exists() else []
    try:
        for provider, publisher, url in (jobs or JOBS):
            record = dict(provider=provider, publisher=publisher, url=url, checked_at=now())
            try:
                job = dict(provider=provider,publisher=publisher,url=url,official=provider in ('xrpl','ripple'))
                response = client.fetch(job,{})
                entries, count = parse_feed(response.content,job,now())
                record.update(http_status=response.status_code, returned=count, relevant=len(entries),
                    latest_published=max((e['published_at'] for e in entries if e['published_at']),default=None),
                    newer_than_nov15=sum(e['published_at']>='2026-11-16' for e in entries),
                    latest=sorted([dict(headline=e['headline'],url=e['url'],published_at=e['published_at']) for e in entries],
                        key=lambda e:e['published_at'],reverse=True)[:3])
            except Exception as exc:
                record['error'] = str(exc)
            results.append(record)
            (ROOT/'tests'/'live_news_source_probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(provider, record.get('http_status'), record.get('relevant'), record.get('latest_published'), record.get('error',''),flush=True)
    finally:
        store.db.close()
    return results


if __name__=='__main__':
    probe()
