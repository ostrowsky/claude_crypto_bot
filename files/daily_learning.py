"""
daily_learning.py — Daily Top Gainer Learning Pipeline

Orchestrates the full daily learning cycle:
  1. Fetch 24h tickers from Binance → identify top gainers
  2. Compute features for ALL watchlist symbols → add to dataset
  3. Resolve pending bandit decisions with actual top gainer data
  4. Train entry bandit on ALL symbols (universal learning)
  5. Retrain top_gainer_model (CatBoost)
  6. Evaluate bandit prediction accuracy
  7. Generate progress report → send to Telegram

Scheduled to run:
  - EOD full cycle at 00:30 UTC (after labels available)
  - Intraday snapshot at 06, 12, 18 UTC (features only, no labels)

Usage:
    python daily_learning.py              # full EOD cycle
    python daily_learning.py --snapshot   # intraday snapshot only
    python daily_learning.py --report     # report only (no training)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from backfill_top_gainer_dataset import (
    fetch_24hr_tickers,
    rank_gainers,
    compute_snapshot_features,
    DATASET_FILE,
)
from strategy import fetch_klines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("daily_learning")

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
REPORT_DIR = RUNTIME_DIR / "reports"
CHAT_IDS_FILE = ROOT / ".chat_ids"
PROGRESS_FILE = RUNTIME_DIR / "learning_progress.jsonl"


# ── Telegram ────────────────────────────────────────────────────────────────

def _load_chat_ids() -> list[int]:
    if not CHAT_IDS_FILE.exists():
        return []
    try:
        payload = json.loads(CHAT_IDS_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    out: list[int] = []
    for item in payload:
        try:
            out.append(int(item))
        except Exception:
            continue
    return sorted(set(out))


async def _send_telegram(text: str) -> None:
    token = str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    if not token:
        log.warning("No TELEGRAM_BOT_TOKEN — skipping notification")
        return
    chat_ids = _load_chat_ids()
    if not chat_ids:
        log.warning("No chat IDs — skipping notification")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for chat_id in chat_ids:
            payload = {
                "chat_id": chat_id,
                "text": text[:4000],
                "disable_web_page_preview": True,
            }
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
            except Exception as exc:
                log.warning("Telegram send failed for %s: %s", chat_id, exc)


# ── Step 1: Collect today's data ────────────────────────────────────────────

async def collect_today_snapshot(
    session: aiohttp.ClientSession,
) -> Dict:
    """
    Collect features + 24h returns for ALL watchlist symbols.
    Returns dict with tickers, top gainers, and count of records logged.
    """
    tickers = await fetch_24hr_tickers(session)
    if not tickers:
        return {"status": "error", "error": "no tickers fetched"}

    top5, top10, top20, top50 = rank_gainers(tickers)
    watchlist = config.load_watchlist()

    # BTC context
    btc_data = await fetch_klines(session, "BTCUSDT", "1h", limit=10)
    btc_ret_1h = 0.0
    btc_ret_4h = 0.0
    if btc_data is not None and len(btc_data) >= 5:
        btc_close = btc_data["c"].astype(float)
        i = len(btc_close) - 2
        btc_ret_1h = (btc_close[i] - btc_close[i - 1]) / btc_close[i - 1] * 100 if btc_close[i - 1] > 0 else 0
        btc_ret_4h = (btc_close[i] - btc_close[max(0, i - 4)]) / btc_close[max(0, i - 4)] * 100 if btc_close[max(0, i - 4)] > 0 else 0

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

        await asyncio.sleep(0.5)

    # Top gainers from watchlist
    top20_in_watchlist = [s for s in top20 if s in watchlist]
    top20_returns = {s: tickers.get(s, 0.0) for s in top20_in_watchlist}

    log.info("Collected %d records. Top20 in watchlist: %s", count, top20_in_watchlist)
    return {
        "status": "ok",
        "n_records": count,
        "n_watchlist": len(watchlist),
        "top5": top5,
        "top10": top10,
        "top20": top20,
        "top20_in_watchlist": top20_in_watchlist,
        "top20_returns": top20_returns,
    }


# ── Step 2: Resolve bandit + train ──────────────────────────────────────────

def resolve_and_train(top_gainer_syms: List[str]) -> Dict:
    """
    Resolve pending decisions, then run full offline training.
    Returns combined results.
    """
    from contextual_bandit import resolve_pending_decisions
    from offline_rl import run_offline_training

    results = {}

    # Resolve pending bandit decisions with actual top gainers
    try:
        n_resolved = resolve_pending_decisions(top_gainer_syms)
        results["resolved_pending"] = n_resolved
        log.info("Resolved %d pending decisions", n_resolved)
    except Exception as e:
        log.error("Resolve pending failed: %s", e)
        results["resolved_pending"] = 0
        results["resolve_error"] = str(e)

    # Run full offline training (includes universal entry bandit)
    try:
        train_results = run_offline_training()
        results.update(train_results)
    except Exception as e:
        log.error("Offline training failed: %s", e)
        results["training_error"] = str(e)

    return results


# ── Step 3: Retrain top_gainer_model ────────────────────────────────────────

def retrain_top_gainer_model() -> Dict:
    """Retrain the CatBoost top gainer model on accumulated dataset."""
    try:
        from train_top_gainer import train_and_save
        result = train_and_save()
        log.info("Top gainer model retrained: %s", result.get("status"))
        return result
    except ImportError:
        log.warning("train_top_gainer module not available")
        return {"status": "skipped", "reason": "module_not_found"}
    except Exception as e:
        log.error("Top gainer model retrain failed: %s", e)
        return {"status": "error", "error": str(e)}


# ── Step 4: Build progress report ──────────────────────────────────────────

def build_progress_report(
    collect_result: Dict,
    train_result: Dict,
    model_result: Dict,
) -> str:
    """Build human-readable progress report for Telegram."""
    now = datetime.now(timezone.utc)
    lines = [f"Daily Learning Report — {now.strftime('%Y-%m-%d %H:%M UTC')}"]
    lines.append("")

    # Collection
    cr = collect_result
    if cr.get("status") == "ok":
        lines.append(f"Data: {cr['n_records']}/{cr['n_watchlist']} symbols collected")
        top20_wl = cr.get("top20_in_watchlist", [])
        top20_ret = cr.get("top20_returns", {})
        if top20_wl:
            lines.append(f"Top20 in watchlist ({len(top20_wl)}):")
            for s in sorted(top20_wl, key=lambda x: top20_ret.get(x, 0), reverse=True)[:10]:
                lines.append(f"  {s:<14} {top20_ret.get(s, 0):+.1f}%")
    else:
        lines.append(f"Data collection: {cr.get('error', 'failed')}")

    lines.append("")

    # Training
    eb = train_result.get("entry_bandit", {})
    if eb.get("status") == "ok":
        lines.append("Entry Bandit Training:")
        lines.append(f"  samples: {eb.get('n_universal_samples', 0)} universal + {eb.get('n_signal_samples', 0)} signal")
        lines.append(f"  top gainers: {eb.get('n_universal_top_gainers', 0)} universal + {eb.get('n_signal_top_gainers', 0)} signal")
        lines.append(f"  days: {eb.get('n_days', 0)}  total_updates: {eb.get('total_updates', 0)}")
        stats = eb.get("arm_stats", [])
        for s in stats:
            lines.append(f"  [{s['name']}] n={s['n_updates']}  theta_norm={s['theta_norm']}  bias={s['bias_est']}")

    # Bandit accuracy
    ba = train_result.get("bandit_accuracy", {})
    if ba.get("status") == "ok":
        lines.append("")
        lines.append("Bandit Prediction Accuracy (backtest):")
        lines.append(f"  recall@20: {ba['overall_recall_top20']*100:.1f}% ({ba['total_top20_enter']}/{ba['total_top20']})")
        lines.append(f"  UCB gap: top_gainers={ba['avg_ucb_gap_top_gainers']:+.4f}  others={ba['avg_ucb_gap_non_top']:+.4f}")
        lines.append(f"  separation: {ba['ucb_separation']:+.4f} (higher = better discrimination)")
        daily = ba.get("daily", [])
        if daily:
            lines.append("  Per day:")
            for d in daily[:7]:
                lines.append(f"    {d['day']}: recall={d['recall_top20']*100:.0f}% ({d['n_top20_enter']}/{d['n_top20']})")

    lines.append("")

    # Resolved pending
    n_resolved = train_result.get("resolved_pending", 0)
    lines.append(f"Pending decisions resolved: {n_resolved}")

    # Model retrain
    if model_result.get("status") == "ok":
        lines.append("")
        lines.append("Top Gainer Model Retrained:")
        for k in ["n_records", "auc_top20", "recall_at_03_top20", "precision_at_03_top20"]:
            if k in model_result:
                lines.append(f"  {k}: {model_result[k]}")

    return "\n".join(lines)


# ── Step 5: Save progress ──────────────────────────────────────────────────

def save_progress(
    collect_result: Dict,
    train_result: Dict,
    model_result: Dict,
) -> None:
    """Append progress record to learning_progress.jsonl for trend tracking."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    ba = train_result.get("bandit_accuracy", {})
    eb = train_result.get("entry_bandit", {})
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_collected": collect_result.get("n_records", 0),
        "n_top20_in_watchlist": len(collect_result.get("top20_in_watchlist", [])),
        "bandit_recall_top20": ba.get("overall_recall_top20"),
        "bandit_ucb_separation": ba.get("ucb_separation"),
        "bandit_total_updates": eb.get("total_updates"),
        "bandit_n_universal": eb.get("n_universal_samples", 0),
        "bandit_n_signal": eb.get("n_signal_samples", 0),
        "model_status": model_result.get("status"),
        "model_auc_top20": model_result.get("auc_top20"),
    }
    with PROGRESS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Main orchestrator ──────────────────────────────────────────────────────

async def run_full_cycle(
    *,
    retrain_model: bool = True,
    send_telegram: bool = True,
) -> Dict:
    """
    Full daily learning cycle:
      1. Collect features + returns for all watchlist symbols
      2. Resolve pending bandit decisions
      3. Train entry bandit (universal)
      4. Optionally retrain top_gainer_model
      5. Evaluate bandit accuracy
      6. Generate + send progress report
    """
    log.info("=== Daily Learning Cycle START ===")

    # Step 1: Collect
    async with aiohttp.ClientSession() as session:
        collect_result = await collect_today_snapshot(session)

    # Step 2+3: Resolve + Train
    top20 = collect_result.get("top20", [])
    train_result = resolve_and_train(top20)

    # Step 4: Model retrain
    model_result = {}
    if retrain_model:
        model_result = retrain_top_gainer_model()

    # Step 5: Report
    report_text = build_progress_report(collect_result, train_result, model_result)
    log.info("\n%s", report_text)

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = REPORT_DIR / f"daily_learning_{today}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    # Save progress for trend tracking
    save_progress(collect_result, train_result, model_result)

    # Telegram
    if send_telegram:
        await _send_telegram(report_text)

    log.info("=== Daily Learning Cycle DONE ===")
    return {
        "collect": collect_result,
        "train": train_result,
        "model": model_result,
        "report_path": str(report_path),
    }


async def run_snapshot_only() -> Dict:
    """Intraday snapshot: collect features + returns, no training."""
    log.info("=== Intraday Snapshot START ===")
    async with aiohttp.ClientSession() as session:
        result = await collect_today_snapshot(session)
    log.info("=== Intraday Snapshot DONE: %d records ===", result.get("n_records", 0))
    return result


async def run_report_only() -> None:
    """Generate and send report without training."""
    from offline_rl import evaluate_bandit_accuracy
    ba = evaluate_bandit_accuracy(n_recent_days=7)

    lines = [f"Bandit Status Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"]
    lines.append("")
    if ba.get("status") == "ok":
        lines.append(f"recall@20: {ba['overall_recall_top20']*100:.1f}% ({ba['total_top20_enter']}/{ba['total_top20']})")
        lines.append(f"UCB separation: {ba['ucb_separation']:+.4f}")
        daily = ba.get("daily", [])
        for d in daily[:7]:
            lines.append(f"  {d['day']}: recall={d['recall_top20']*100:.0f}% ({d['n_top20_enter']}/{d['n_top20']})")
    else:
        lines.append(f"Status: {ba.get('status')}")

    # Progress trend
    if PROGRESS_FILE.exists():
        recent = []
        for line in PROGRESS_FILE.read_text(encoding="utf-8").strip().splitlines()[-7:]:
            try:
                recent.append(json.loads(line))
            except Exception:
                pass
        if recent:
            lines.append("")
            lines.append("Learning Progress (last 7):")
            for r in recent:
                recall = r.get("bandit_recall_top20")
                sep = r.get("bandit_ucb_separation")
                ts = r.get("ts", "")[:10]
                recall_str = f"{recall*100:.0f}%" if recall is not None else "n/a"
                sep_str = f"{sep:+.3f}" if sep is not None else "n/a"
                lines.append(f"  {ts}: recall={recall_str}  sep={sep_str}")

    text = "\n".join(lines)
    log.info("\n%s", text)
    await _send_telegram(text)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Top Gainer Learning Pipeline")
    parser.add_argument("--snapshot", action="store_true", help="Intraday snapshot only (no training)")
    parser.add_argument("--report", action="store_true", help="Report only (no collection/training)")
    parser.add_argument("--no-telegram", action="store_true", help="Skip Telegram notification")
    parser.add_argument("--no-model-retrain", action="store_true", help="Skip top_gainer_model retraining")
    args = parser.parse_args()

    if args.report:
        asyncio.run(run_report_only())
        return 0

    if args.snapshot:
        result = asyncio.run(run_snapshot_only())
        return 0 if result.get("status") == "ok" else 1

    result = asyncio.run(run_full_cycle(
        retrain_model=not args.no_model_retrain,
        send_telegram=not args.no_telegram,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
