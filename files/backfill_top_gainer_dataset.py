from __future__ import annotations

"""
Top Gainer Dataset Builder — collects daily labels and features for training.

Two modes:
  1. Daily cron: run at 00:05 UTC to label yesterday's data
  2. Historical backfill: backfill N days of top gainer labels

For each day:
  - Fetch 24hr tickers at EOD → determine top 5/10/20/50 gainers
  - For each coin in watchlist:
    - Compute features at multiple timestamps (02, 06, 10, 14, 18, 22 UTC)
    - Label as top_5/10/20/50 based on EOD ranking
    - Log to top_gainer_dataset.jsonl

This builds the training set for TopGainerModel.

Usage:
    python backfill_top_gainer_dataset.py --days 180
    python backfill_top_gainer_dataset.py --daily  # for cron
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from indicators import compute_features
from strategy import fetch_klines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_FILE = Path(__file__).resolve().parent / "top_gainer_dataset.jsonl"
BINANCE_REST = "https://api.binance.com"
SNAPSHOT_HOURS_UTC = [2, 6, 10, 14, 18, 22]  # Feature snapshots throughout the day


async def fetch_24hr_tickers(
    session: aiohttp.ClientSession,
    timestamp_ms: Optional[int] = None,
) -> Dict[str, float]:
    """
    Fetch 24hr price change for all USDT pairs.
    Returns {symbol: priceChangePercent}.
    """
    url = f"{BINANCE_REST}/api/v3/ticker/24hr"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                log.warning("ticker/24hr returned %d", resp.status)
                return {}
            data = await resp.json()
    except Exception as e:
        log.warning("ticker/24hr fetch failed: %s", e)
        return {}

    results = {}
    for t in data:
        sym = str(t.get("symbol", ""))
        if not sym.endswith("USDT"):
            continue
        # Skip stablecoins and leveraged tokens
        skip = False
        for ex in config.SCAN_EXCLUDE:
            if ex in sym:
                skip = True
                break
        if skip:
            continue
        try:
            results[sym] = float(t.get("priceChangePercent", 0))
        except (ValueError, TypeError):
            pass
    return results


def rank_gainers(
    tickers: Dict[str, float],
    min_volume: float = 1_000_000,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Rank coins by 24h change.
    Returns (top5, top10, top20, top50) symbol lists.
    """
    sorted_coins = sorted(tickers.items(), key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in sorted_coins]
    top5 = set(symbols[:5])
    top10 = set(symbols[:10])
    top20 = set(symbols[:20])
    top50 = set(symbols[:50])
    return list(top5), list(top10), list(top20), list(top50)


async def compute_snapshot_features(
    session: aiohttp.ClientSession,
    symbol: str,
    btc_return_1h: float = 0.0,
    btc_return_4h: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Compute features for a single symbol at current time."""
    try:
        data_1h = await fetch_klines(session, symbol, "1h", limit=100)
        if data_1h is None or len(data_1h) < 30:
            return None

        close = data_1h["c"].astype(float)
        high = data_1h["h"].astype(float)
        low = data_1h["l"].astype(float)
        vol = data_1h["v"].astype(float)
        feat = compute_features(data_1h["o"], data_1h["h"], data_1h["l"], data_1h["c"], data_1h["v"])

        i = len(data_1h) - 2  # last closed bar
        if i < 20:
            return None

        features: Dict[str, float] = {}

        # Momentum
        features["tg_return_1h"] = _pct(close, 1, i)
        features["tg_return_4h"] = _pct(close, 4, i)
        start_idx = max(0, i - 24)
        features["tg_return_since_open"] = (
            (close[i] - close[start_idx]) / close[start_idx] * 100.0
            if close[start_idx] > 0 else 0.0
        )

        # Volume
        avg_vol = float(np.mean(vol[max(0, i - 20):i])) if i > 1 else 1.0
        features["tg_volume_ratio_1h"] = vol[i] / avg_vol if avg_vol > 0 else 1.0
        avg_vol_4 = float(np.mean(vol[max(0, i - 4):i])) if i > 1 else 1.0
        features["tg_volume_ratio_4h"] = (
            float(np.sum(vol[max(0, i - 3):i + 1])) / (avg_vol_4 * 4)
            if avg_vol_4 > 0 else 1.0
        )
        features["tg_volume_acceleration"] = (
            features["tg_volume_ratio_1h"] / features["tg_volume_ratio_4h"]
            if features["tg_volume_ratio_4h"] > 0 else 1.0
        )

        # Price structure
        ema20 = _safe(feat, "ema_fast", i, close[i])
        ema50 = _safe(feat, "ema_slow", i, close[i])
        ema200 = _safe(feat, "ema200", i, close[i])
        features["tg_price_vs_ema20_pct"] = (close[i] - ema20) / ema20 * 100 if ema20 > 0 else 0
        features["tg_price_vs_ema50_pct"] = (close[i] - ema50) / ema50 * 100 if ema50 > 0 else 0
        features["tg_price_vs_ema200_pct"] = (close[i] - ema200) / ema200 * 100 if ema200 > 0 else 0
        features["tg_ema20_slope"] = _safe(feat, "slope", i, 0.0)
        features["tg_adx"] = _safe(feat, "adx", i, 25.0)
        features["tg_rsi"] = _safe(feat, "rsi", i, 50.0)
        atr = _safe(feat, "atr", i, 0.0)
        features["tg_atr_pct"] = atr / close[i] * 100 if close[i] > 0 else 0

        # Range
        h24 = float(np.max(high[max(0, i - 24):i + 1]))
        l24 = float(np.min(low[max(0, i - 24):i + 1]))
        features["tg_daily_range_pct"] = (h24 - l24) / l24 * 100 if l24 > 0 else 0
        features["tg_range_position"] = (close[i] - l24) / (h24 - l24) if (h24 - l24) > 0 else 0.5

        # BTC context
        features["tg_btc_return_1h"] = btc_return_1h
        features["tg_btc_return_4h"] = btc_return_4h
        features["tg_vs_btc_1h"] = features["tg_return_1h"] - btc_return_1h
        features["tg_vs_btc_4h"] = features["tg_return_4h"] - btc_return_4h
        features["tg_sector_avg_return"] = 0.0

        # Placeholders for order flow / derivatives (not available in backfill)
        for of_key in ["tg_of_cvd_pct_5m", "tg_of_imbalance_5m",
                       "tg_of_large_trade_ratio_5m", "tg_of_breakout_signal_5m"]:
            features[of_key] = 0.0
        for dd_key in ["tg_funding_rate", "tg_oi_change_1h", "tg_oi_change_4h",
                       "tg_ls_ratio", "tg_liq_total_1h", "tg_funding_flip"]:
            features[dd_key] = 0.0

        # Historical gainer stats (not available in backfill)
        features["tg_was_top_gainer_yesterday"] = 0.0
        features["tg_top_gainer_count_7d"] = 0.0
        features["tg_avg_daily_return_7d"] = 0.0
        features["tg_max_daily_return_7d"] = 0.0

        # Temporal
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0
        features["tg_hour_utc"] = float(now.hour)
        features["tg_hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        features["tg_hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        features["tg_day_of_week"] = float(now.weekday())
        features["tg_is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0

        return features

    except Exception as e:
        log.debug("Feature computation failed for %s: %s", symbol, e)
        return None


async def collect_daily(session: aiohttp.ClientSession) -> int:
    """Collect today's labels and log to dataset. Returns number of records logged."""
    tickers = await fetch_24hr_tickers(session)
    if not tickers:
        log.warning("No tickers fetched")
        return 0

    top5, top10, top20, top50 = rank_gainers(tickers)
    watchlist = config.load_watchlist()

    # BTC context
    btc_data = await fetch_klines(session, "BTCUSDT", "1h", limit=10)
    btc_ret_1h = 0.0
    btc_ret_4h = 0.0
    if btc_data is not None and len(btc_data) >= 5:
        btc_close = btc_data["c"].astype(float)
        btc_ret_1h = _pct(btc_close, 1, len(btc_close) - 2)
        btc_ret_4h = _pct(btc_close, 4, len(btc_close) - 2)

    count = 0
    batch_size = 10
    for batch_start in range(0, len(watchlist), batch_size):
        batch = watchlist[batch_start:batch_start + batch_size]
        tasks = [
            compute_snapshot_features(session, sym, btc_ret_1h, btc_ret_4h)
            for sym in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sym, result in zip(batch, results):
            if isinstance(result, Exception) or result is None:
                continue

            eod_return = tickers.get(sym, 0.0)
            record = {
                "ts": int(time.time() * 1000),
                "symbol": sym,
                "features": {k: round(v, 6) for k, v in result.items()},
                "label_top5": int(sym in top5),
                "label_top10": int(sym in top10),
                "label_top20": int(sym in top20),
                "label_top50": int(sym in top50),
                "eod_return_pct": round(eod_return, 4),
            }

            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

        await asyncio.sleep(0.5)  # Rate limiting

    log.info("Collected %d records. Top 5: %s", count, top5)
    return count


def _pct(arr, periods, i):
    if i < periods or arr[i - periods] == 0:
        return 0.0
    return (arr[i] - arr[i - periods]) / arr[i - periods] * 100.0


def _safe(feat, key, i, default):
    arr = feat.get(key)
    if arr is not None and i < len(arr) and np.isfinite(arr[i]):
        return float(arr[i])
    return default


async def backfill_historical(session: aiohttp.ClientSession, days: int) -> int:
    """
    Historical backfill using daily klines.

    For each past day:
      - Fetch 1d klines for all watchlist coins to get EOD returns
      - Rank by daily return to determine top gainers
      - Fetch 1h klines up to that day to compute features
      - Log to dataset
    """
    watchlist = config.load_watchlist()
    total = 0
    now_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    for d in range(1, days + 1):
        target_day = now_utc - timedelta(days=d)
        target_ms = int(target_day.timestamp() * 1000)
        log.info("Backfill day %d/%d: %s", d, days, target_day.date())

        # Fetch daily returns for all coins on target day
        day_returns: Dict[str, float] = {}
        batch_size = 20
        for b in range(0, len(watchlist), batch_size):
            batch = watchlist[b:b + batch_size]
            tasks = []
            for sym in batch:
                async def _fetch_day_return(s=sym):
                    try:
                        # Fetch 2 daily candles ending at target_day+1d to get target day's close
                        end_ms = target_ms + 86_400_000
                        url = f"{BINANCE_REST}/api/v3/klines"
                        params = {"symbol": s, "interval": "1d", "limit": 3,
                                  "endTime": end_ms}
                        async with session.get(url, params=params,
                                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status != 200:
                                return None
                            data = await resp.json()
                        if not data or len(data) < 2:
                            return None
                        # Find the candle that starts at target_ms
                        for bar in data:
                            if abs(int(bar[0]) - target_ms) < 3_600_000:
                                open_p = float(bar[1])
                                close_p = float(bar[4])
                                if open_p > 0:
                                    return s, (close_p - open_p) / open_p * 100.0
                        return None
                    except Exception:
                        return None
                tasks.append(_fetch_day_return())

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if r and not isinstance(r, Exception) and r is not None:
                    sym, ret = r
                    day_returns[sym] = ret
            await asyncio.sleep(0.3)

        if len(day_returns) < 10:
            log.warning("Too few returns for %s, skipping", target_day.date())
            continue

        # Rank gainers
        top5, top10, top20, top50 = rank_gainers(day_returns)
        top5_set = set(top5)
        top10_set = set(top10)
        top20_set = set(top20)
        top50_set = set(top50)

        # BTC context at ~06:00 UTC on target day
        snapshot_ms = target_ms + 6 * 3_600_000
        btc_ret_1h = 0.0
        btc_ret_4h = 0.0
        try:
            btc_url = f"{BINANCE_REST}/api/v3/klines"
            btc_params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 10,
                          "endTime": snapshot_ms}
            async with session.get(btc_url, params=btc_params,
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    btc_data = await resp.json()
                    if btc_data and len(btc_data) >= 5:
                        closes = [float(b[4]) for b in btc_data]
                        if closes[-2] > 0:
                            btc_ret_1h = (closes[-1] - closes[-2]) / closes[-2] * 100
                        if closes[-5] > 0:
                            btc_ret_4h = (closes[-1] - closes[-5]) / closes[-5] * 100
        except Exception:
            pass

        # Compute features for each coin (using klines ending at ~06:00 UTC on target day)
        day_count = 0
        for b in range(0, len(watchlist), batch_size):
            batch = watchlist[b:b + batch_size]
            tasks = []
            for sym in batch:
                async def _fetch_features(s=sym):
                    try:
                        url = f"{BINANCE_REST}/api/v3/klines"
                        params = {"symbol": s, "interval": "1h", "limit": 100,
                                  "endTime": snapshot_ms}
                        async with session.get(url, params=params,
                                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status != 200:
                                return None
                            raw = await resp.json()
                        if not raw or len(raw) < 30:
                            return None

                        arr = np.zeros(len(raw), dtype=[
                            ("t","i8"),("o","f8"),("h","f8"),("l","f8"),("c","f8"),("v","f8")
                        ])
                        for ii, bar in enumerate(raw):
                            arr[ii] = (int(bar[0]), float(bar[1]), float(bar[2]),
                                       float(bar[3]), float(bar[4]), float(bar[5]))

                        feat = compute_features(arr["o"], arr["h"], arr["l"], arr["c"], arr["v"])
                        close = arr["c"]
                        high = arr["h"]
                        low = arr["l"]
                        vol = arr["v"]
                        i = len(arr) - 2

                        features: Dict[str, float] = {}
                        features["tg_return_1h"] = _pct(close, 1, i)
                        features["tg_return_4h"] = _pct(close, 4, i)
                        start_idx = max(0, i - 24)
                        features["tg_return_since_open"] = (
                            (close[i] - close[start_idx]) / close[start_idx] * 100.0
                            if close[start_idx] > 0 else 0.0
                        )
                        avg_vol = float(np.mean(vol[max(0, i-20):i])) if i > 1 else 1.0
                        features["tg_volume_ratio_1h"] = vol[i] / avg_vol if avg_vol > 0 else 1.0
                        avg_vol_4 = float(np.mean(vol[max(0, i-4):i])) if i > 1 else 1.0
                        features["tg_volume_ratio_4h"] = (
                            float(np.sum(vol[max(0, i-3):i+1])) / (avg_vol_4 * 4)
                            if avg_vol_4 > 0 else 1.0
                        )
                        features["tg_volume_acceleration"] = (
                            features["tg_volume_ratio_1h"] / features["tg_volume_ratio_4h"]
                            if features["tg_volume_ratio_4h"] > 0 else 1.0
                        )
                        ema20 = _safe(feat, "ema_fast", i, close[i])
                        ema50 = _safe(feat, "ema_slow", i, close[i])
                        ema200 = _safe(feat, "ema200", i, close[i])
                        features["tg_price_vs_ema20_pct"] = (close[i] - ema20) / ema20 * 100 if ema20 > 0 else 0
                        features["tg_price_vs_ema50_pct"] = (close[i] - ema50) / ema50 * 100 if ema50 > 0 else 0
                        features["tg_price_vs_ema200_pct"] = (close[i] - ema200) / ema200 * 100 if ema200 > 0 else 0
                        features["tg_ema20_slope"] = _safe(feat, "slope", i, 0.0)
                        features["tg_adx"] = _safe(feat, "adx", i, 25.0)
                        features["tg_rsi"] = _safe(feat, "rsi", i, 50.0)
                        atr = _safe(feat, "atr", i, 0.0)
                        features["tg_atr_pct"] = atr / close[i] * 100 if close[i] > 0 else 0
                        h24 = float(np.max(high[max(0, i-24):i+1]))
                        l24 = float(np.min(low[max(0, i-24):i+1]))
                        features["tg_daily_range_pct"] = (h24 - l24) / l24 * 100 if l24 > 0 else 0
                        features["tg_range_position"] = (close[i] - l24) / (h24 - l24) if (h24 - l24) > 0 else 0.5
                        features["tg_btc_return_1h"] = btc_ret_1h
                        features["tg_btc_return_4h"] = btc_ret_4h
                        features["tg_vs_btc_1h"] = features["tg_return_1h"] - btc_ret_1h
                        features["tg_vs_btc_4h"] = features["tg_return_4h"] - btc_ret_4h
                        features["tg_sector_avg_return"] = 0.0
                        for k in ["tg_of_cvd_pct_5m", "tg_of_imbalance_5m",
                                  "tg_of_large_trade_ratio_5m", "tg_of_breakout_signal_5m",
                                  "tg_funding_rate", "tg_oi_change_1h", "tg_oi_change_4h",
                                  "tg_ls_ratio", "tg_liq_total_1h", "tg_funding_flip",
                                  "tg_was_top_gainer_yesterday", "tg_top_gainer_count_7d",
                                  "tg_avg_daily_return_7d", "tg_max_daily_return_7d"]:
                            features[k] = 0.0
                        features["tg_ls_ratio"] = 1.0
                        dt = datetime.fromtimestamp(snapshot_ms / 1000, tz=timezone.utc)
                        hour = dt.hour + dt.minute / 60.0
                        features["tg_hour_utc"] = float(dt.hour)
                        features["tg_hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
                        features["tg_hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
                        features["tg_day_of_week"] = float(dt.weekday())
                        features["tg_is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0
                        return features
                    except Exception as e:
                        log.debug("Hist feature failed %s: %s", s, e)
                        return None
                tasks.append(_fetch_features())

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, result in zip(batch, results):
                if isinstance(result, Exception) or result is None:
                    continue
                eod_return = day_returns.get(sym, 0.0)
                record = {
                    "ts": target_ms + 6 * 3_600_000,
                    "symbol": sym,
                    "features": {k: round(v, 6) for k, v in result.items()},
                    "label_top5": int(sym in top5_set),
                    "label_top10": int(sym in top10_set),
                    "label_top20": int(sym in top20_set),
                    "label_top50": int(sym in top50_set),
                    "eod_return_pct": round(eod_return, 4),
                }
                with open(DATASET_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                day_count += 1
            await asyncio.sleep(0.3)

        log.info("Day %s: %d records, top20: %s", target_day.date(), day_count,
                 [s for s in top20_set][:5])
        total += day_count
        await asyncio.sleep(1.0)

    return total


async def main():
    parser = argparse.ArgumentParser(description="Top Gainer Dataset Builder")
    parser.add_argument("--daily", action="store_true", help="Collect today's labels")
    parser.add_argument("--days", type=int, default=0, help="Backfill N days of history")
    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        if args.daily or args.days == 0:
            count = await collect_daily(session)
            log.info("Daily collection: %d records", count)
        else:
            log.info("Starting historical backfill for %d days...", args.days)
            total = await backfill_historical(session, args.days)
            log.info("Backfill complete: %d total records", total)


if __name__ == "__main__":
    asyncio.run(main())
