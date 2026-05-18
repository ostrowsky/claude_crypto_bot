"""Silent-miss root-cause breakdown (North-Star diagnostic).

A "silent miss" = a watchlist coin that became a top-20 daily gainer but
the bot produced ZERO per-symbol events for it that day (no entry, no
blocked, no candidate). The coverage-funnel reports the rate; this script
answers WHY, which determines the fix:

  - never_scanned         : not in that day's morning `confirmed_symbols`
                            -> never entered the intraday monitor loop ->
                            physically could not signal. Fix = scan
                            coverage (widen morning filter / monitor all
                            105 watchlist coins intraday).
  - scanned_no_signal      : was confirmed/monitored but no entry
                            condition ever fired. Fix = entry-rule
                            sensitivity, not coverage.
  - out_of_watchlist       : not one of the 105 watchlist coins ->
                            EXPECTED and acceptable per CLAUDE.md §14.

All silent-miss is North-Star-critical: the bot exists to early-capture
top-20 gainers, and `watchlist_top_early_capture_pct` is hard-capped by
the never_scanned bucket.

Usage:
    pyembed\\python.exe files\\_backtest_silent_miss_breakdown.py
    pyembed\\python.exe files\\_backtest_silent_miss_breakdown.py --window-days 60
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOP_GAINER = HERE / "top_gainer_dataset.jsonl"
BOT_EVENTS = HERE / "bot_events.jsonl"
WATCHLIST = HERE / "watchlist.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Silent-miss root-cause breakdown")
    ap.add_argument("--window-days", type=int, default=60)
    args = ap.parse_args()

    cut = datetime.now(timezone.utc) - timedelta(days=args.window_days)

    try:
        wl = set(json.loads(WATCHLIST.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        wl = set()

    top20 = defaultdict(set)
    with io.open(TOP_GAINER, encoding="utf-8") as f:
        for ln in f:
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ts = e.get("ts")
            if not ts:
                continue
            try:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                continue
            if dt < cut:
                continue
            if e.get("label_top20") == 1:
                top20[dt.strftime("%Y-%m-%d")].add(e.get("symbol"))

    sym_ev = defaultdict(set)
    scanned = defaultdict(set)
    with io.open(BOT_EVENTS, encoding="utf-8") as f:
        for ln in f:
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            t = e.get("ts", "")
            try:
                dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if dt < cut:
                continue
            d = dt.strftime("%Y-%m-%d")
            ev = e.get("event", "")
            s = e.get("sym") or e.get("symbol") or ""
            if s:
                sym_ev[(d, s)].add(ev)
            if ev == "analysis_done":
                for cs in (e.get("confirmed_symbols") or []):
                    scanned[d].add(cs)

    cls = Counter()
    never_scanned = scanned_no_signal = out_wl = 0
    never_syms = Counter()
    tot = 0
    for d, syms in top20.items():
        for s in syms:
            tot += 1
            if sym_ev.get((d, s)):
                if "entry" in sym_ev[(d, s)]:
                    cls["entered"] += 1
                elif "blocked" in sym_ev[(d, s)]:
                    cls["blocked_only"] += 1
                else:
                    cls["other_event"] += 1
                continue
            cls["silent_miss"] += 1
            if wl and s not in wl:
                out_wl += 1
            elif s in scanned.get(d, set()):
                scanned_no_signal += 1
            else:
                never_scanned += 1
                never_syms[s] += 1

    if tot == 0:
        print("[!] no top-20 winner-days in window")
        return 1

    print(f"=== Silent-miss breakdown — last {args.window_days}d ===")
    print(f"top-20 winner-days: {tot}  (watchlist size: {len(wl)})")
    for k, v in cls.most_common():
        print(f"  {k:14s} {v:4d} ({100*v/tot:.1f}%)")
    sm = cls.get("silent_miss", 0)
    print(f"\nSilent-miss root cause ({sm} total):")
    print(f"  never_scanned      {never_scanned:4d} "
          f"({100*never_scanned/max(sm,1):.0f}% of silent / {100*never_scanned/tot:.1f}% of all) "
          f"-> SCAN-COVERAGE gap (North-Star cap)")
    print(f"  scanned_no_signal  {scanned_no_signal:4d} "
          f"({100*scanned_no_signal/max(sm,1):.0f}% of silent) -> entry-rule sensitivity")
    print(f"  out_of_watchlist   {out_wl:4d} "
          f"({100*out_wl/max(sm,1):.0f}% of silent) -> acceptable per CLAUDE.md §14")
    print(f"\nTop never-scanned symbols: "
          f"{', '.join(f'{s}({n})' for s, n in never_syms.most_common(10))}")

    print(json.dumps({
        "metric": "SILENT_MISS_root_cause",
        "window_days": args.window_days,
        "top20_winner_days": tot,
        "silent_miss": sm,
        "never_scanned": never_scanned,
        "scanned_no_signal": scanned_no_signal,
        "out_of_watchlist": out_wl,
        "actionable_silent_miss_pct": round(100 * never_scanned / tot, 2),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
