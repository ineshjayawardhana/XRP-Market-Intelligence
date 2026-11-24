"""Permitted public discovery endpoints; failures remain visible in monitoring."""
from urllib.parse import urlencode

SOURCES = [dict(provider=p, publisher=n, url=u, official=p in ('xrpl', 'ripple')) for p, n, u in [
    ('coindesk', 'CoinDesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml'),
    ('cryptoslate', 'CryptoSlate', 'https://cryptoslate.com/feed/'),
    ('cointelegraph_xrp', 'Cointelegraph · XRP', 'https://cointelegraph.com/rss/tag/xrp'),
    ('dailyhodl_xrp', 'The Daily Hodl · XRP', 'https://dailyhodl.com/ripple-and-xrp/feed/'),
    ('google_news', 'Google News RSS', 'https://news.google.com/rss/search?' + urlencode(dict(q='XRP OR "XRP Ledger" when:3d', hl='en-US', gl='US', ceid='US:en'))),
    ('cointelegraph', 'Cointelegraph', 'https://cointelegraph.com/rss'),
    ('decrypt', 'Decrypt', 'https://decrypt.co/feed'),
    ('newsbtc_xrp', 'NewsBTC · XRP', 'https://www.newsbtc.com/tag/xrp/feed/'),
    ('bitcoinist_xrp', 'Bitcoinist · XRP', 'https://bitcoinist.com/tag/xrp/feed/'),
    ('cryptopotato_xrp', 'CryptoPotato · Ripple', 'https://cryptopotato.com/tag/ripple/feed/'),
    ('cryptobasic_xrp', 'The Crypto Basic · XRP', 'https://thecryptobasic.com/category/ripple/feed/'),
    ('utoday', 'U.Today', 'https://u.today/rss.php'),
    ('xrpl', 'XRP Ledger', 'https://xrpl.org/blog/rss.xml'),
    ('ripple', 'Ripple', 'https://ripple.com/insights/feed/'),
]]
