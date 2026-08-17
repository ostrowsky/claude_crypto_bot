"""Portfolio return vs equal-weight buy-and-hold — the TH-11 canonical metric.

The scorecard has carried `portfolio_alpha: unknown` because the only source was
a signal-evaluator report from 2026-05-04. This computes it from what the bot
actually did.

**This is a diagnostic, not an optimisation target.** The bot is an alert system
with no position sizing; the standing objective is early detection and coverage,
not per-trade income. The number answers one question — "is this alert stream
worth acting on at all?" — and answering it does not make it the thing to
maximise. TH-11 asks for it precisely because a proxy cannot answer it.

Why a slot simulation rather than a mean of trade P&Ls: a mean over trades is not
a portfolio return. Twenty trades of +1% are not +20% if they overlapped on the
same capital, and they are not +1% either. The bot holds at most `MAX_OPEN`
positions, so capital is modelled as `MAX_OPEN` equal slots — the smallest model
that makes the two sides comparable.

    pyembed\\python.exe files\\portfolio_alpha.py --days 30

Spec: docs/specs/features/portfolio-alpha-spec.md
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import label_store as LS  # noqa: E402

EVENTS = HERE / "bot_events.jsonl"
WATCHLIST = HERE / "watchlist.json"


def _day(ts) -> str | None:
    """UTC day from whichever timestamp shape a log happens to use.

    This project writes at least three: ISO strings (`bot_events.exit.ts`,
    `critic_dataset.ts_signal`), epoch milliseconds (`bar_ts`) and epoch
    seconds. Assuming a number here silently dropped EVERY exit event and
    reported "no closed trades" for a bot that had just closed one — the same
    mistake the gate replay made two hours earlier, which is why the parsing
    lives in one place with both shapes handled explicitly.
    """
    if isinstance(ts, str):
        return ts[:10] if len(ts) >= 10 else None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts > 1e11:
        ts /= 1000.0
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def load_closed_trades(since_day: str) -> list[dict]:
    """Realized trades: one per exit event, carrying its own P&L."""
    out = []
    with io.open(EVENTS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (e.get("event") or e.get("type")) != "exit":
                continue
            day = _day(e.get("ts"))
            if day is None or day < since_day:
                continue
            pnl = e.get("pnl_pct")
            if pnl is None:
                continue
            try:
                pnl = float(pnl)
            except (TypeError, ValueError):
                continue
            out.append({"day": day, "sym": e.get("sym"), "pnl_pct": pnl,
                        "bars_held": e.get("bars_held"), "mode": e.get("mode")})
    out.sort(key=lambda t: t["day"])
    return out


def bot_return_pct(trades: list[dict], *, slots: int) -> tuple[float, dict]:
    """Compound the realized trades over `slots` equal parcels of capital.

    Each closed trade consumes one slot for the day it closed on. A day with
    more closes than slots is compounded in order; the model does not pretend to
    reconstruct intraday slot availability, and says so.
    """
    by_day: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_day[t["day"]].append(t["pnl_pct"])

    equity = 1.0
    per_day = {}
    for day in sorted(by_day):
        pnls = by_day[day]
        # Each trade moves 1/slots of capital.
        day_ret = sum(p / 100.0 for p in pnls) / slots
        equity *= (1.0 + day_ret)
        per_day[day] = round(day_ret * 100.0, 6)
    return (equity - 1.0) * 100.0, per_day


def buy_and_hold_pct(days: list[str], *, watchlist: set[str],
                     store: LS.LabelStore | None = None) -> tuple[float, dict]:
    """Equal-weight watchlist, held across the window, from exchange closes.

    Uses the immutable label store, so the benchmark and the bot's own outcome
    are not measured against two different price sources.
    """
    store = store or LS.LabelStore()
    per_sym_day: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in store.records():
        if r["symbol"] in watchlist and r.get("complete"):
            per_sym_day[r["symbol"]][r["utc_day"]] = r

    window = set(days)
    rets, covered = [], 0
    for sym, days_map in per_sym_day.items():
        have = sorted(d for d in days_map if d in window)
        if len(have) < 2:
            continue
        first, last = days_map[have[0]], days_map[have[-1]]
        if first["open"] <= 0:
            continue
        rets.append((last["close"] / first["open"] - 1.0) * 100.0)
        covered += 1
    value = sum(rets) / len(rets) if rets else 0.0
    return value, {"symbols_used": covered, "watchlist": len(watchlist)}


def compute(days_window: int = 30, *, slots: int = 10) -> dict[str, Any]:
    since = (datetime.now(timezone.utc) - timedelta(days=days_window)
             ).strftime("%Y-%m-%d")
    trades = load_closed_trades(since)
    if not trades:
        return {"available": False, "reason": f"no closed trades since {since}"}

    days = sorted({t["day"] for t in trades})
    watchlist = set(json.loads(WATCHLIST.read_text(encoding="utf-8")))
    bot, per_day = bot_return_pct(trades, slots=slots)
    bh, bh_meta = buy_and_hold_pct(days, watchlist=watchlist)

    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    return {
        "available": True,
        "window_days": days_window,
        "days_with_trades": len(days),
        "window": f"{days[0]}..{days[-1]}",
        "n_trades": len(trades),
        "slots": slots,
        "bot_return_pct": round(bot, 4),
        "buy_and_hold_pct": round(bh, 4),
        "alpha_vs_buy_and_hold_pct": round(bot - bh, 4),
        # TH-01: the rate never travels without what it is a rate of.
        "win_rate_pct": round(100.0 * wins / len(trades), 2),
        "mean_trade_pnl_pct": round(sum(t["pnl_pct"] for t in trades) / len(trades), 4),
        "benchmark_symbols_used": bh_meta["symbols_used"],
        "benchmark_watchlist_size": bh_meta["watchlist"],
        "per_day_return_pct": per_day,
        "caveats": [
            "capital is modelled as MAX_OPEN equal slots; intraday slot "
            "availability is not reconstructed",
            "closed trades only — an open position at the window edge is "
            "excluded, not marked to market",
            "diagnostic, not an optimisation target: the bot is an alert system "
            "without position sizing",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    res = compute(args.days, slots=args.slots)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if not res["available"]:
        print(res["reason"])
        return 1

    print("=" * 70)
    print(f"Portfolio vs buy-and-hold · {res['window']} "
          f"({res['days_with_trades']} trading days)")
    print("=" * 70)
    print(f"  trades closed          {res['n_trades']}  "
          f"(win rate {res['win_rate_pct']}%, mean {res['mean_trade_pnl_pct']:+.3f}%)")
    print(f"  bot portfolio          {res['bot_return_pct']:+.2f}%   "
          f"({res['slots']} equal slots)")
    print(f"  buy-and-hold watchlist {res['buy_and_hold_pct']:+.2f}%   "
          f"({res['benchmark_symbols_used']}/{res['benchmark_watchlist_size']} symbols)")
    print(f"  ALPHA                  {res['alpha_vs_buy_and_hold_pct']:+.2f}%   "
          f"(target > 0)")
    print()
    for c in res["caveats"]:
        print(f"  · {c}")
    print("\nMETRIC_JSON:" + json.dumps(
        {k: v for k, v in res.items() if k != "per_day_return_pct"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
