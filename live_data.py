"""Phase 2 + 3: public XRP market data, news collection, signals, and backtests.

Times are UTC. All technical calculations use CLOSED 1-hour candles only.
This application is research-only and never places trades.
"""
from __future__ import annotations

import os
import hashlib
import sqlite3
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from model import predict_news

load_dotenv()
ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "news_history.sqlite3"
BINANCE = "https://data-api.binance.vision"
CRYPTOPANIC = "https://cryptopanic.com/api/developer/v2/posts/"
HOUR_MS = 3_600_000
USER_AGENT = "XRP-Research-Dashboard/1.0 (academic, read-only)"


def _json_get(url: str, params: dict) -> dict | list:
    response = requests.get(url, params=params, timeout=18,
                            headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.json()


def get_live_price() -> dict:
    """Public REST ticker, refreshed by Streamlit every ~30s; NOT a tick-by-tick feed."""
    raw = _json_get(f"{BINANCE}/api/v3/ticker/24hr", {"symbol": "XRPUSDT"})
    return {
        "price": float(raw["lastPrice"]),
        "change_24h_pct": float(raw["priceChangePercent"]),
        "high_24h": float(raw["highPrice"]),
        "low_24h": float(raw["lowPrice"]),
        "volume_24h_xrp": float(raw["volume"]),
        "volume_24h_usdt": float(raw["quoteVolume"]),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "exchange_close_at": (pd.to_datetime(raw['closeTime'], unit='ms', utc=True).isoformat()
                              if raw.get('closeTime') is not None else None),
    }


def get_historical_prices(days: int = 30) -> pd.DataFrame:
    """Fetch paginated 1-hour XRPUSDT candles, plus 7 warm-up days.

    Binance returns up to 1000 candles per request. Drops the forming candle.
    """
    if not 1 <= days <= 365:
        raise ValueError("days should be 1 to 365")
    now = datetime.now(timezone.utc)
    start_ms = int((now - timedelta(days=days + 7)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    cursor = start_ms
    collected = []
    for _ in range(12):  # Bound repeated API calls for safety.
        batch = _json_get(f"{BINANCE}/api/v3/klines", {
            "symbol": "XRPUSDT", "interval": "1h",
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not batch:
            break
        collected.extend(batch)
        next_cursor = int(batch[-1][0]) + HOUR_MS
        if next_cursor <= cursor:
            raise RuntimeError("Binance returned non-advancing candle pagination")
        cursor = next_cursor
        if cursor >= end_ms or len(batch) < 1000:
            break
    if not collected:
        raise RuntimeError("No historical XRP candles returned from Binance")
    columns = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "trades", "taker_base",
               "taker_quote", "unused"]
    df = pd.DataFrame(collected, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("open_time").drop_duplicates("open_time")
    df = df[df["close_time"] <= pd.Timestamp(now)].reset_index(drop=True)
    if len(df) < 60:
        raise RuntimeError("Insufficient complete candles for technical indicators")
    return df[["open_time", "close_time", "open", "high", "low", "close",
               "volume", "quote_volume"]]


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().sort_values("close_time").reset_index(drop=True)
    c = df["close"].astype(float)
    change = c.diff()
    avg_gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ratio = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + ratio)).where(avg_loss.ne(0), 100.0)
    df.loc[(avg_gain == 0) & (avg_loss == 0), "rsi"] = 50.0
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    previous_vol_avg = df["volume"].shift(1).rolling(window=20, min_periods=20).mean()
    df["volume_spike"] = (df["volume"] >= 1.8 * previous_vol_avg).fillna(False)
    bullish = (df["ema20"] > df["ema50"]) & df["rsi"].between(50, 70)
    bearish = (df["ema20"] < df["ema50"]) & (df["rsi"] < 50)
    df["technical"] = np.select([bullish, bearish], ["Bullish", "Bearish"],
                                 default="Neutral")
    df["overbought"] = df["rsi"] > 70
    return df


def get_technical_signals(candles: pd.DataFrame | None = None) -> dict:
    """Bullish / Bearish / Neutral is a RULE, not a verified forecast."""
    df = calculate_indicators(get_historical_prices(30) if candles is None else candles)
    last = df.iloc[-1]
    return {
        "technical": str(last["technical"]),
        "rsi": round(float(last["rsi"]), 2),
        "ema20": round(float(last["ema20"]), 6),
        "ema50": round(float(last["ema50"]), 6),
        "macd": round(float(last["macd"]), 6),
        "macd_signal": round(float(last["macd_signal"]), 6),
        "volume_spike": bool(last["volume_spike"]),
        "overbought": bool(last["overbought"]),
        "last_closed_candle_utc": str(last["close_time"]),
    }


def combine_signal(news_label: str | None, technical: str,
                   volume_spike: bool, rsi: float) -> str:
    if news_label == "Real_Adoption" and technical == "Bullish" and volume_spike:
        return "STRONG BULLISH"
    if news_label == "Hype_Speculation" and rsi > 70:
        return "FAKE PUMP WARNING"
    if news_label == "Negative_FUD" and technical == "Bearish":
        return "BEARISH"
    return "NEUTRAL"


def _google_news_rss() -> list[dict]:
    """No-key public RSS fallback. Indexing is delayed and may change without notice."""
    response = requests.get("https://news.google.com/rss/search", params={
        "q": '(XRP OR "XRP Ledger") when:1d',
        "hl": "en-US", "gl": "US", "ceid": "US:en",
    }, timeout=18, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError("Google News RSS could not be parsed") from exc
    items = []
    for entry in root.findall("./channel/item")[:75]:
        headline = (entry.findtext("title") or "").strip()
        if not headline:
            continue
        source = (entry.findtext("source") or "Google News RSS").strip()
        # RSS guid is stable across refreshes, unlike its list position.
        guid = entry.findtext("guid") or entry.findtext("link") or headline
        key = "GNEWS-" + hashlib.sha256(guid.encode("utf-8")).hexdigest()[:24]
        pubdate = entry.findtext("pubDate")
        try:
            published_iso = parsedate_to_datetime(pubdate).astimezone(timezone.utc).isoformat() if pubdate else ""
        except (TypeError, ValueError):
            published_iso = ""
        items.append({"id": key, "published_at": published_iso,
                      "headline": headline, "source": source})
    return items


def fetch_latest_news() -> list[dict]:
    """Use CryptoPanic with an entitled token; otherwise use free Google News RSS.

    The CryptoPanic Developer/free plan is not guaranteed available. The RSS
    fallback is not a commercial real-time API and may lag publication.
    """
    token = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
    if not token or token == "PASTE_YOUR_KEY_HERE":
        return _google_news_rss()
    response = _json_get(CRYPTOPANIC, {
        "auth_token": token,
        "currencies": "XRP",
        "public": "true",
    })
    posts = response.get("results", []) if isinstance(response, dict) else []
    return [{
        "id": "CP-" + str(post["id"]),
        "published_at": str(post.get("published_at") or ""),
        "headline": str(post.get("title") or "").strip(),
        "source": str((post.get("source") or {}).get("title") or "Unknown"),
    } for post in posts if post.get("title") and post.get("id") is not None]


def _db_connection():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=12)
    con.execute("PRAGMA busy_timeout = 12000")
    con.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id TEXT PRIMARY KEY,
            published_at TEXT,
            observed_at TEXT NOT NULL,
            headline TEXT NOT NULL,
            source TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            price_at_observation REAL NOT NULL
        )
    """)
    return con


def save_new_headlines(items: list[dict], price_at_observation: float,
                       backend: str = "tfidf") -> int:
    """Store only previously unseen posts; price is at first FETCH, not publish time."""
    inserted = 0
    with _db_connection() as con:
        known = {row[0] for row in con.execute("SELECT id FROM news")}
        for item in items:
            if item["id"] in known:
                continue
            pred = predict_news(item["headline"], backend=backend)
            cursor = con.execute("""
                INSERT OR IGNORE INTO news
                (id, published_at, observed_at, headline, source, label, confidence,
                 price_at_observation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (item["id"], item["published_at"],
                  datetime.now(timezone.utc).isoformat(), item["headline"],
                  item["source"], pred["label"], pred["confidence"],
                  price_at_observation))
            inserted += cursor.rowcount
    return inserted


def read_news() -> pd.DataFrame:
    with _db_connection() as con:
        df = pd.read_sql_query("SELECT * FROM news ORDER BY observed_at DESC", con)
    if not df.empty:
        for col in ["published_at", "observed_at"]:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        df = df.sort_values(["published_at", "observed_at"], ascending=False).reset_index(drop=True)
    return df


def load_archived_news(path: Path = ROOT / "data" / "historical_news.csv",
                       backend: str = "tfidf") -> pd.DataFrame:
    """Optional real timestamped headlines CSV: date,headline,source.

    Assumption: article was publicly available at its recorded date.
    """
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if not {"date", "headline", "source"}.issubset(frame.columns):
        raise ValueError("historical_news.csv needs date,headline,source columns")
    frame["observed_at"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["observed_at", "headline"]).copy()
    frame = frame[frame["headline"].astype(str).str.strip().ne("")]
    # Never include the 200 SYNTHETIC training examples in financial backtesting.
    frame = frame[~frame["source"].astype(str).str.contains("SYNTHETIC", case=False)]
    predictions = [predict_news(str(h), backend) for h in frame["headline"]]
    frame["label"] = [p["label"] for p in predictions]
    frame["confidence"] = [p["confidence"] for p in predictions]
    return frame


def _forward_outcomes(candles: pd.DataFrame) -> pd.DataFrame:
    data = calculate_indicators(candles)
    # Exactly 24 complete 1-hour candles beyond the signal close.
    data["future_price_24h"] = data["close"].shift(-24)
    # A missing candle must not turn a 25+ hour return into a claimed 24h outcome.
    exact_horizon = data['close_time'].shift(-24) - data['close_time'] == pd.Timedelta(hours=24)
    data["return_24h_pct"] = (data["future_price_24h"] / data["close"] - 1) * 100
    data["outcome"] = np.select(
        [data["return_24h_pct"] >= 1, data["return_24h_pct"] <= -1],
        ["UP", "DOWN"], default="NEUTRAL"
    )
    return data[exact_horizon].dropna(subset=["future_price_24h", "rsi"]).copy()


def technical_baseline_backtest(candles: pd.DataFrame, days: int = 180) -> pd.DataFrame:
    """Daily non-overlapping evaluations, based only on past CLOSED hourly candles."""
    data = _forward_outcomes(candles)
    earliest = data["close_time"].max() - pd.Timedelta(days=days)
    data = data[data["close_time"] >= earliest].copy()
    data["day"] = data["close_time"].dt.floor("D")
    daily = data.groupby("day", as_index=False).tail(1).copy()
    daily["signal"] = daily["technical"].str.upper()
    daily["win"] = ((daily["signal"] == "BULLISH") & (daily["outcome"] == "UP")) | (
        (daily["signal"] == "BEARISH") & (daily["outcome"] == "DOWN")) | (
        (daily["signal"] == "NEUTRAL") & (daily["outcome"] == "NEUTRAL"))
    return daily


def combined_news_backtest(candles: pd.DataFrame, news: pd.DataFrame,
                           days: int = 180) -> pd.DataFrame:
    """Approximate event-based backtest; no fake performance when news is absent.

    Event entry approximation = last closed 1h candle price before news timestamp.
    More rigorous work should use minute candles and walk-forward retraining.
    """
    if news is None or news.empty:
        return pd.DataFrame()
    historical = _forward_outcomes(candles).sort_values("close_time")
    latest = historical["close_time"].max()
    events = news.copy()
    events["observed_at"] = pd.to_datetime(events["observed_at"], utc=True, errors="coerce")
    events = events.dropna(subset=["observed_at", "label", "headline"])
    events = events[(events["observed_at"] >= latest - pd.Timedelta(days=days)) &
                    (events["observed_at"] <= latest)]
    if events.empty:
        return pd.DataFrame()
    # Don't trust headlines that have the same exact news event twice.
    events = events.sort_values("observed_at").drop_duplicates(["headline", "observed_at"])
    chosen = ["close_time", "close", "rsi", "technical", "volume_spike",
              "future_price_24h", "return_24h_pct", "outcome"]
    merged = pd.merge_asof(
        events.sort_values("observed_at"),
        historical[chosen].sort_values("close_time"),
        left_on="observed_at", right_on="close_time", direction="backward",
        tolerance=pd.Timedelta(hours=2),
    ).dropna(subset=["close", "rsi", "future_price_24h"])
    if merged.empty:
        return pd.DataFrame()
    merged["signal"] = merged.apply(
        lambda row: combine_signal(str(row["label"]), str(row["technical"]),
                                   bool(row["volume_spike"]), float(row["rsi"])), axis=1
    )
    merged["day"] = merged["observed_at"].dt.floor("D")
    # One event per signal/day to limit overlapping evaluation windows.
    merged = merged.drop_duplicates(["day", "signal"]).copy()
    merged["win"] = np.select([
        merged["signal"].eq("STRONG BULLISH"),
        merged["signal"].eq("BEARISH"),
        merged["signal"].eq("NEUTRAL"),
    ], [merged["outcome"].eq("UP"), merged["outcome"].eq("DOWN"),
        merged["outcome"].eq("NEUTRAL")], default=False)
    return merged


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["signal", "cases", "up_rate_pct", "down_rate_pct",
                                     "neutral_rate_pct", "rule_win_rate_pct"])
    rows = []
    for signal, group in results.groupby("signal"):
        rows.append({
            "signal": signal,
            "cases": len(group),
            "up_rate_pct": round(100 * group["outcome"].eq("UP").mean(), 1),
            "down_rate_pct": round(100 * group["outcome"].eq("DOWN").mean(), 1),
            "neutral_rate_pct": round(100 * group["outcome"].eq("NEUTRAL").mean(), 1),
            "rule_win_rate_pct": (round(100 * group["win"].mean(), 1)
                                  if signal != "FAKE PUMP WARNING" else np.nan),
        })
    return pd.DataFrame(rows).sort_values("cases", ascending=False).reset_index(drop=True)
