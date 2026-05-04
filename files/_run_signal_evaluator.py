"""Wrapper / adapter for skills/signal-efficiency-evaluator.

Bridges the gap between our bot_events.jsonl schema and the skill's
expected schema, and pre-stages kline cache so the skill doesn't
need the `requests` library (we have aiohttp instead).

Translation:
  our bot_events.jsonl                 →  skill format
  ---------------------                    ------------
  {"event":"entry", "sym":"BTCUSDT",       {"ts":..., "symbol":"BTCUSDT",
   "tf":"15m", "price":..., ...}            "event":"BUY", "price":...,
                                            "confidence": ml_proba or 0.5}
  {"event":"exit",  "sym":..., ...}        {"event":"SELL", ...}

Klines cache: <repo>/history/<symbol>_<tf>.csv

Usage:
  pyembed/python.exe files/_run_signal_evaluator.py --window-days 7

Spec: docs/specs/features/signal-evaluator-integration-spec.md
"""
from __future__ import annotations
import asyncio
import io
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Force UTF-8 stdout (Windows cp1251 default chokes on arrows / em-dashes)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PYEMBED = ROOT / "pyembed" / "python.exe"
SKILL_SCRIPT = ROOT / "skills" / "signal-efficiency-evaluator" / "scripts" / "evaluate_signals.py"
EVENTS_PATH = ROOT / "files" / "bot_events.jsonl"
HISTORY_DIR = ROOT / "history"
EVAL_DIR = ROOT / "evaluation_output"
TMP_EVENTS = ROOT / ".runtime" / "_skill_events_translated.jsonl"


def translate_events(window_start_ms: int, window_end_ms: int,
                     mode_filter: str | None = None) -> tuple[set[str], int]:
    """
    Read our bot_events.jsonl, translate entry/exit events into BUY/SELL,
    write to TMP_EVENTS. Return (set_of_symbols_seen, n_events).

    If mode_filter is set, keeps only entries with mode==mode_filter
    (and their matching exits — exits don't carry mode but are paired
    by symbol so we keep all exits for symbols that had a matching entry).
    """
    syms: set[str] = set()
    n = 0
    # If filtering by mode, first pass: collect symbols whose entries match
    syms_with_matching_entry: set[str] = set()
    if mode_filter:
        with io.open(EVENTS_PATH, encoding="utf-8") as src:
            for ln in src:
                try: e = json.loads(ln)
                except: continue
                if e.get("event") == "entry" and e.get("mode") == mode_filter:
                    sym = e.get("sym") or e.get("symbol")
                    if sym: syms_with_matching_entry.add(sym)
    TMP_EVENTS.parent.mkdir(exist_ok=True)
    with io.open(EVENTS_PATH, encoding="utf-8") as src, \
         io.open(TMP_EVENTS, "w", encoding="utf-8") as dst:
        for ln in src:
            try:
                e = json.loads(ln)
            except Exception:
                continue
            ev = e.get("event", "")
            if ev not in ("entry", "exit"):
                continue
            ts_str = e.get("ts", "")
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue
            ts_ms = int(dt.timestamp() * 1000)
            if ts_ms < window_start_ms or ts_ms > window_end_ms:
                continue
            sym = e.get("sym") or e.get("symbol") or ""
            if not sym:
                continue
            # Mode filter: keep only events for syms where matching entry was seen.
            # Exits don't have mode field, but we keep them for already-tracked syms.
            if mode_filter:
                if ev == "entry" and e.get("mode") != mode_filter:
                    continue
                if sym not in syms_with_matching_entry:
                    continue
            price_key = "exit_price" if ev == "exit" else "price"
            price = e.get(price_key) or e.get("price") or e.get("entry_price")
            if price is None:
                continue
            translated = {
                "ts": ts_str,
                "symbol": sym,
                "event": "BUY" if ev == "entry" else "SELL",
                "price": float(price),
                "confidence": float(e.get("ml_proba") or 0.5),
            }
            # Pass-through useful metadata
            for k in ("tf", "mode", "ranker_top_gainer_prob", "ranker_ev",
                      "candidate_score", "ranker_quality_proba", "signal_mode",
                      "bars_held", "reason"):
                if e.get(k) is not None:
                    translated[k] = e[k]
            dst.write(json.dumps(translated) + "\n")
            n += 1
            syms.add(sym)
    return syms, n


async def prefetch_klines(symbols: set[str], timeframe: str,
                          start: datetime, end: datetime) -> int:
    """
    Use bot's existing aiohttp-based fetch_klines to pre-stage kline cache
    in <repo>/history/<symbol>_<tf>.csv. Returns count of cached symbols.
    """
    sys.path.insert(0, str(ROOT / "files"))
    try:
        import aiohttp
        from strategy import fetch_klines  # bot's own loader
    except Exception as e:
        print(f"[wrapper] cannot import fetch_klines: {e}", file=sys.stderr)
        return 0

    HISTORY_DIR.mkdir(exist_ok=True)
    bar_minutes = {"1m":1, "5m":5, "15m":15, "30m":30, "1h":60, "4h":240, "1d":1440}.get(timeframe, 15)
    span_min = (end - start).total_seconds() / 60
    n_bars_needed = int(span_min / bar_minutes) + 50

    cached = 0
    async with aiohttp.ClientSession() as session:
        for sym in sorted(symbols):
            cache_path = HISTORY_DIR / f"{sym}_{timeframe}.csv"
            try:
                data = await fetch_klines(session, sym, timeframe, limit=min(1500, n_bars_needed))
                if data is None or len(data["c"]) == 0:
                    continue
                # Convert to CSV with skill-expected columns: ts,open,high,low,close,volume
                with io.open(cache_path, "w", encoding="utf-8") as f:
                    f.write("ts,open,high,low,close,volume\n")
                    for i in range(len(data["c"])):
                        ts_ms = int(data["t"][i])
                        ts_iso = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()
                        f.write(f"{ts_iso},{data['o'][i]},{data['h'][i]},"
                                f"{data['l'][i]},{data['c'][i]},{data['v'][i]}\n")
                cached += 1
            except Exception as e:
                print(f"[wrapper] fetch failed for {sym}: {e}", file=sys.stderr)
    return cached


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--timeframe", type=str, default="15m")
    p.add_argument("--swing-threshold-pct", type=float, default=4.0,
                   help="default 4.0 for crypto-1h-15m (skill default 3.0 too noisy)")
    p.add_argument("--max-intratrend-drawdown-pct", type=float, default=2.0)
    p.add_argument("--min-trend-duration-bars", type=int, default=4)
    p.add_argument("--fee-pct", type=float, default=0.075)
    p.add_argument("--symbols", type=str, default=None,
                   help="comma-separated; default = all syms with events in window")
    p.add_argument("--skip-prefetch", action="store_true",
                   help="skip kline pre-fetch (use existing cache)")
    p.add_argument("--append-to-rl-memory", action="store_true",
                   help="append evaluation_feedback record to .runtime/eval_feedback.jsonl "
                        "(separate file — bot's rl_memory.jsonl is bandit-state)")
    p.add_argument("--per-mode", action="store_true",
                   help="run skill once per signal_mode (trend, impulse_speed, "
                        "alignment, etc.) — separate report dir per mode. "
                        "Helps when modes operate on different scales (15m vs 1h).")
    args = p.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.window_days)
    print(f"[wrapper] window: {start.isoformat()} → {end.isoformat()}")

    # Discover modes if --per-mode
    if args.per_mode:
        modes_seen: set[str] = set()
        with io.open(EVENTS_PATH, encoding="utf-8") as f:
            for ln in f:
                try: e = json.loads(ln)
                except: continue
                if e.get("event") == "entry" and e.get("mode"):
                    ts_str = e.get("ts", "")
                    try:
                        dt = datetime.fromisoformat(ts_str.replace("Z","+00:00"))
                    except: continue
                    if start <= dt <= end:
                        modes_seen.add(e["mode"])
        if not modes_seen:
            print("[wrapper] --per-mode: no modes found in window"); return
        print(f"[wrapper] --per-mode: will run skill for: {sorted(modes_seen)}")
        # Run once per mode
        for mode in sorted(modes_seen):
            print(f"\n{'='*60}\n[wrapper] === mode={mode} ===\n{'='*60}")
            _run_one(args, start, end, mode_filter=mode,
                     output_subdir=f"per_mode/{mode}")
        return

    # Default: single run, all modes
    _run_one(args, start, end, mode_filter=None, output_subdir=None)


def _run_one(args, start, end, mode_filter: str | None, output_subdir: str | None):
    syms, n_events = translate_events(int(start.timestamp()*1000),
                                       int(end.timestamp()*1000),
                                       mode_filter=mode_filter)
    label = f" mode={mode_filter}" if mode_filter else ""
    print(f"[wrapper]{label} translated events: {n_events} across {len(syms)} symbols")
    if not n_events:
        print(f"[wrapper]{label} no events — skipping"); return

    if args.symbols:
        target_syms = set(s.strip().upper() for s in args.symbols.split(","))
        syms &= target_syms

    if not args.skip_prefetch:
        print(f"[wrapper]{label} pre-fetching klines for {len(syms)} symbols...")
        cached = asyncio.run(prefetch_klines(syms, args.timeframe, start, end))
        print(f"[wrapper]{label} cached {cached} kline files in {HISTORY_DIR}")

    # Invoke the skill
    cmd = [
        str(PYEMBED), str(SKILL_SCRIPT),
        "--project-root", str(ROOT),
        "--events-file", str(TMP_EVENTS),
        "--window-days", str(args.window_days),
        "--timeframe", args.timeframe,
        "--swing-threshold-pct", str(args.swing_threshold_pct),
        "--max-intratrend-drawdown-pct", str(args.max_intratrend_drawdown_pct),
        "--min-trend-duration-bars", str(args.min_trend_duration_bars),
        "--fee-pct", str(args.fee_pct),
        "--output-dir", str(EVAL_DIR / output_subdir) if output_subdir else str(EVAL_DIR),
    ]
    if args.append_to_rl_memory:
        cmd.append("--append-to-rl-memory")
    if syms:
        # Pass symbol filter to the skill so it doesn't try to fetch syms we
        # didn't pre-cache (skill needs `requests` for live fetch — we don't have it).
        cmd.extend(["--symbols", *sorted(syms)])

    out_label = f" ({output_subdir})" if output_subdir else ""
    print(f"\n[wrapper]{out_label} running skill (output dir: {cmd[cmd.index('--output-dir')+1]})")
    rc = subprocess.call(cmd)
    print(f"\n[wrapper]{out_label} skill exited with code {rc}")


if __name__ == "__main__":
    main()
