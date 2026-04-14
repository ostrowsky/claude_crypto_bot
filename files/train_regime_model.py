from __future__ import annotations

"""
Train Regime Detector HMM on historical BTC data.

Fetches 6+ months of BTC 4h klines from Binance, computes features,
trains a 6-state Gaussian HMM, and saves to regime_model.json.

States are automatically ordered by mean return:
  0 = strong_bull  (highest return)
  1 = weak_bull
  2 = ranging
  3 = volatile_chop
  4 = weak_bear
  5 = strong_bear  (lowest return)

Usage:
    python train_regime_model.py                    # default 180 days
    python train_regime_model.py --days 365         # 1 year
    python train_regime_model.py --iterations 100   # more EM iterations
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from indicators import compute_features
from strategy import fetch_klines
from regime_detector import RegimeDetector, GaussianHMM, REGIME_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_FILE = Path(__file__).resolve().parent / "regime_model.json"


async def fetch_btc_history(
    session: aiohttp.ClientSession,
    tf: str = "4h",
    days: int = 180,
) -> np.ndarray:
    """Fetch BTC historical klines by paginating backwards."""
    all_bars = []
    limit = 1000  # Binance max per request
    tf_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[tf]
    total_bars = days * 86_400_000 // tf_ms

    end_time = None
    remaining = total_bars

    while remaining > 0:
        batch = min(limit, remaining)
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": tf, "limit": batch}
        if end_time:
            params["endTime"] = end_time

        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                log.warning("Kline fetch failed: %d", resp.status)
                break
            data = await resp.json()

        if not data:
            break

        for bar in data:
            all_bars.append((
                int(bar[0]),     # open time
                float(bar[1]),   # open
                float(bar[2]),   # high
                float(bar[3]),   # low
                float(bar[4]),   # close
                float(bar[5]),   # volume
            ))

        end_time = int(data[0][0]) - 1  # before first bar of this batch
        remaining -= len(data)
        log.info("Fetched %d bars, total %d/%d", len(data), total_bars - remaining, total_bars)

        if len(data) < batch:
            break
        await asyncio.sleep(0.3)

    # Sort by time (oldest first)
    all_bars.sort(key=lambda x: x[0])

    # Deduplicate by timestamp
    seen = set()
    unique = []
    for bar in all_bars:
        if bar[0] not in seen:
            seen.add(bar[0])
            unique.append(bar)

    arr = np.zeros(len(unique), dtype=[
        ("t", "i8"), ("o", "f8"), ("h", "f8"),
        ("l", "f8"), ("c", "f8"), ("v", "f8"),
    ])
    for idx, bar in enumerate(unique):
        arr[idx] = bar

    return arr


def analyze_regimes(hmm: GaussianHMM, features: np.ndarray, btc_data: np.ndarray) -> None:
    """Print regime analysis and statistics."""
    states, log_prob = hmm.viterbi(features)
    probs = hmm.predict_proba(features)

    close = btc_data["c"][20:].astype(float)  # skip warmup
    returns = np.diff(np.log(close.clip(1e-8))) * 100.0

    log.info("\n=== REGIME MODEL ANALYSIS ===")
    log.info("Total bars: %d, Log probability: %.2f", len(states), log_prob)
    log.info("")

    for s_idx, name in enumerate(REGIME_NAMES):
        mask = states == s_idx
        count = int(mask.sum())
        pct = count / len(states) * 100
        if count > 0:
            # Returns in this regime (aligned with states which start at warmup+1)
            regime_returns = returns[mask[1:]] if mask.sum() > 0 else np.array([0])
            avg_ret = float(np.mean(regime_returns)) if len(regime_returns) > 0 else 0
            std_ret = float(np.std(regime_returns)) if len(regime_returns) > 0 else 0
            log.info(
                "  %d. %-15s: %4d bars (%5.1f%%) | avg_ret=%+.3f%% | std=%.3f%%",
                s_idx, name, count, pct, avg_ret, std_ret,
            )
        else:
            log.info("  %d. %-15s: %4d bars (%5.1f%%)", s_idx, name, count, pct)

    # Transition matrix
    log.info("\nTransition matrix (rows=from, cols=to):")
    header = "  " + "".join(f"{n[:8]:>10}" for n in REGIME_NAMES)
    log.info(header)
    for i, name in enumerate(REGIME_NAMES):
        row = f"  {name[:8]:<10}"
        for j in range(len(REGIME_NAMES)):
            row += f"{hmm.A[i, j]:10.3f}"
        log.info(row)

    # Current regime
    current = REGIME_NAMES[states[-1]]
    current_prob = float(probs[-1, states[-1]])
    log.info("\nCurrent regime: %s (%.1f%% confidence)", current, current_prob * 100)

    # Regime durations
    log.info("\nAverage regime durations (bars):")
    for s_idx, name in enumerate(REGIME_NAMES):
        # Count consecutive runs
        runs = []
        current_run = 0
        for s in states:
            if s == s_idx:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        if runs:
            log.info("  %-15s: avg=%.1f bars, max=%d, count=%d episodes",
                     name, np.mean(runs), max(runs), len(runs))


async def main():
    parser = argparse.ArgumentParser(description="Train Regime HMM Model")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--tf", type=str, default="4h", help="Timeframe (1h, 4h, 1d)")
    parser.add_argument("--iterations", type=int, default=50, help="EM iterations")
    parser.add_argument("--output", type=str, default=str(MODEL_FILE), help="Output path")
    args = parser.parse_args()

    log.info("Fetching %d days of BTC %s data...", args.days, args.tf)

    async with aiohttp.ClientSession() as session:
        btc_data = await fetch_btc_history(session, tf=args.tf, days=args.days)

    log.info("Got %d BTC bars", len(btc_data))
    if len(btc_data) < 100:
        log.error("Insufficient data for training (need 100+ bars)")
        return

    # Compute features
    btc_features = compute_features(btc_data["o"], btc_data["h"], btc_data["l"], btc_data["c"], btc_data["v"])

    # Train
    detector = RegimeDetector()
    success = detector.train_offline(
        btc_data=btc_data,
        btc_features=btc_features,
        n_iter=args.iterations,
        save_path=args.output,
    )

    if success:
        log.info("Model saved to %s", args.output)
        # Analyze
        features = detector._extract_features(btc_data, btc_features, None)
        if features is not None:
            analyze_regimes(detector._hmm, features, btc_data)
    else:
        log.error("Training failed")


if __name__ == "__main__":
    asyncio.run(main())
