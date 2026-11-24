"""Uncached news reads and visible persistent source monitoring."""
import pandas as pd
import streamlit as st

from live_news import NewsPoller, monitoring
from market_intelligence import slst

STRETCH = {'width': 'stretch'} if tuple(int(p) for p in st.__version__.split('.')[:2]) >= (1, 50) else {'use_container_width': True}


@st.cache_resource
def news_poller(root):
    return NewsPoller(root)


def run_news_polling(root, enabled):
    poller = news_poller(str(root))
    if enabled:
        poller.start()
    else:
        poller.pause()
    return poller


def request_news_refresh(root):
    news_poller(str(root)).request()


def news_freshness(news, enabled=True):
    st.subheader('Latest available XRP news')
    now = pd.Timestamp.now(tz='UTC')
    valid = news[news.published_at.notna() & news.published_at.le(now)] if not news.empty else news
    if valid.empty:
        st.warning('No article with a usable publication timestamp is available. First-seen and collection times are not substituted for publication time.')
    else:
        latest = valid.published_at.max()
        hours = (now-latest).total_seconds()/3600
        st.metric('Time since latest published article', f'{hours:.1f} hours')
        st.caption('Latest source-reported publication: ' + slst(latest) + ' SLST. Publication time is reported by the source, not independently verified.')
        if hours > 24:
            st.warning('News is stale: the latest dated article is more than 24 hours old. Saved historical headlines are not live news. Check source status below.')
        else:
            st.info('A source-reported article within the last 24 hours is available. This does not establish factual correctness or market impact.')
    if not enabled:
        st.caption('Automatic polling is paused. Saved news remains available; Refresh data now requests one public refresh.')


def render_source_status(root, enabled):
    st.subheader('News Source Status')
    state, sources = monitoring(root)
    poller = news_poller(str(root))
    st.caption('Polling: ' + ('enabled · approximately every five minutes while this Streamlit process runs' if enabled else 'paused') +
        ' · Last completed check: ' + slst(state.get('checked_at')) + ' SLST · Next scheduled check: ' +
        slst(state.get('next_poll_at')) + ' SLST')
    if state.get('running'):
        st.info('Public news collection is running in the background. Saved rows and source results appear on the next display refresh.')
    if poller.error:
        st.warning('Background collector error: ' + poller.error)
    rows = []
    now = pd.Timestamp.now(tz='UTC')
    from live_news import source_status
    for item in sources:
        rows.append(dict(source=item['name'], status=source_status(item, now),
            last_success_slst=slst(item.get('last_success_at')), returned=item.get('articles_returned'),
            xrp_articles=item.get('relevant_articles'), newly_saved=item.get('newly_saved'),
            latest_publication_slst=slst(item.get('latest_publication_at')),
            next_attempt_slst=slst(item.get('next_attempt_at')), error=item.get('error') or '',
            http_status=item.get('http_status'), feed_url=item.get('feed_url')))
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH,
        column_config={'feed_url': st.column_config.LinkColumn('Public feed')})
    st.caption('Returned counts describe the last attempt (304 means unchanged, with zero downloaded articles). A successful fetch can contain no XRP news or only old articles. Restrictions and Retry-After/backoff remain active during manual refresh.')
