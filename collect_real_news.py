"""Collect real XRP news metadata from the GDELT DOC 2.0 API.

Designed to respect API rate limits and preserve previously collected headlines
and human labels. Dates from GDELT's `seendate` are FIRST SEEN, not necessarily
original publication times. No synthetic or automatically labeled data.

Example: python collect_real_news.py --days 30 --window-days 5
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "xrp_real_news_unlabeled.csv"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '(XRP OR XRPL OR "XRP Ledger" OR RLUSD) sourcelang:english'
COLUMNS = ["id", "date", "headline", "source", "url", "timestamp_type", "label"]


class GDELTRateLimitError(RuntimeError):
    """Raised after the API continues returning 429 following cautious retries."""


def parse_gdelt_response(data: dict) -> list[dict]:
    rows: list[dict] = []
    for item in data.get("articles", []):
        title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
        url = str(item.get("url") or "").strip()
        raw_seen = str(item.get("seendate") or "")
        # Filter out unrelated stories containing the word 'ripple'.
        if not title or not url or not re.search(r"\b(XRP|XRPL|RLUSD)\b|XRP Ledger", title, re.I):
            continue
        try:
            seen = pd.to_datetime(raw_seen, utc=True, errors="raise").to_pydatetime()
        except (ValueError, TypeError):
            continue
        source = str(item.get("domain") or "Unknown").strip()
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        rows.append({
            "id": "GDELT-" + digest,
            "date": seen.isoformat(),
            "headline": title,
            "source": source,
            "url": url,
            "timestamp_type": "GDELT_FIRST_SEEN",
            "label": "",  # Human annotation ONLY
        })
    return rows


def _retry_after_seconds(response: requests.Response) -> float:
    """Honor numeric Retry-After headers, within a practical upper bound."""
    raw = response.headers.get("Retry-After", "")
    try:
        return max(0.0, min(float(raw), 120.0))
    except (ValueError, TypeError):
        return 0.0


def fetch_window(start: datetime, end: datetime, maximum: int = 250) -> list[dict]:
    params = {
        "query": QUERY,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "maxrecords": max(1, min(maximum, 250)),
        "sort": "DateAsc",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        was_429 = False
        retry_after = 0.0
        try:
            response = requests.get(
                API,
                params=params,
                timeout=35,
                headers={"User-Agent": "XRP-Research-Project/1.2 (academic)"},
            )
            if response.status_code == 429:
                was_429 = True
                retry_after = _retry_after_seconds(response)
                raise GDELTRateLimitError("GDELT returned HTTP 429 (rate limited)")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("GDELT response was not a JSON object")
            return parse_gdelt_response(payload)
        except (requests.RequestException, ValueError, GDELTRateLimitError) as exc:
            last_error = exc
            if attempt < 2:
                if was_429:
                    delay = max(20.0 * (2 ** attempt), retry_after)
                    print(f"  HTTP 429: pausing {delay:.0f} seconds before retry {attempt + 2}/3")
                else:
                    delay = 6.0 * (attempt + 1)
                    print(f"  Request problem: {exc}. Retry {attempt + 2}/3")
                time.sleep(delay)
            elif was_429:
                raise GDELTRateLimitError(str(exc)) from exc
    raise RuntimeError(f"GDELT request failed after retries: {last_error}")


def merge_and_save(rows: list[dict]) -> pd.DataFrame:
    """Merge with existing CSV, preserving old human labels and de-duplicating."""
    new_df = pd.DataFrame(rows, columns=COLUMNS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype=str).fillna("")
        for col in COLUMNS:
            if col not in old.columns:
                old[col] = ""
        old = old[COLUMNS]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    if combined.empty:
        print("No headlines available; nothing fabricated or overwritten.")
        return combined

    combined["label"] = combined["label"].fillna("")
    combined["_has_label"] = combined["label"].astype(str).str.strip().ne("")
    combined = combined.sort_values("_has_label", ascending=False, kind="stable")
    combined = combined.drop_duplicates("id", keep="first")
    combined["_norm"] = combined["headline"].fillna("").str.lower().str.replace(r"\W+", "", regex=True)
    combined = combined.drop_duplicates("_norm", keep="first")
    combined = combined.drop(columns=["_has_label", "_norm"]).sort_values("date").reset_index(drop=True)
    # A temporary file prevents partially overwriting the dataset if writing fails.
    tmp = OUT.with_suffix(".tmp")
    combined.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(OUT)
    print(f"Saved {len(combined)} unique real article metadata rows: {OUT}")
    return combined


def collect(days: int = 60, window_days: int = 3,
            limit: int = 250, pause_seconds: float = 15) -> pd.DataFrame:
    if not 1 <= days <= 85:
        raise ValueError("--days must be between 1 and 85")
    if not 1 <= window_days <= 14:
        raise ValueError("--window-days must be between 1 and 14")
    if pause_seconds < 5:
        raise ValueError("--pause-seconds must be at least 5")

    end = datetime.now(timezone.utc) - timedelta(minutes=15)
    start = end - timedelta(days=days)
    rows: list[dict] = []
    cursor = start
    consecutive_429_windows = 0
    success_windows = 0
    while cursor < end:
        stop = min(cursor + timedelta(days=window_days), end)
        try:
            chunk = fetch_window(cursor, stop, limit)
            rows.extend(chunk)
            consecutive_429_windows = 0
            success_windows += 1
            if len(chunk) >= limit:
                print("WARNING: max record limit reached; try a smaller --window-days.")
            print(f"{cursor.date()} to {stop.date()}: {len(chunk)} relevant headlines")
        except GDELTRateLimitError:
            consecutive_429_windows += 1
            print(f"WARNING: {cursor.date()} to {stop.date()} not collected (GDELT HTTP 429)")
            if consecutive_429_windows >= 2:
                print("Stopping early to respect GDELT rate limits. "
                      "Try again another time; the existing CSV will be kept.")
                break
        except RuntimeError as exc:
            print(f"WARNING: {cursor.date()} to {stop.date()} not collected: {exc}")
        cursor = stop
        if cursor < end:
            time.sleep(pause_seconds)

    combined = merge_and_save(rows)
    print(f"Successful date windows this run: {success_windows}")
    print("Reminder: GDELT timestamps are FIRST SEEN, not verified publication time.")
    print("Next: python label_real_news.py  (human annotation required)")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Respectfully collect real XRP news metadata")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--max-per-window", type=int, default=250)
    parser.add_argument("--pause-seconds", type=float, default=15)
    args = parser.parse_args()
    collect(args.days, args.window_days, args.max_per_window, args.pause_seconds)
