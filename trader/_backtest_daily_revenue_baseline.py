"""Daily revenue of the CURRENT signal stream if it had been traded -- baseline for the
autonomous-trading roadmap (2026-10-10, autonomous-trading-roadmap-1010-spec.md).

Nothing deployed. Question: before building execution, does the bot's own signal
stream make money per day after fees, and how many days does it take to tell a
change in daily revenue from noise?

FIXED BEFORE THE RUN
  trades   every real entry->exit pair in bot_events.jsonl since 2026-03-01 (15m and 1h),
           paired per (sym, tf), realized pnl_pct from the exit event (bot's own exits)
  sizing   equal slots: each trade gets 1/MAX_OPEN_POSITIONS of equity (10 -> 10%);
           a trade that would be the 11th open position is skipped
  fees     Binance spot taker 0.10% per side (0.20% round trip); BNB discount 0.075%
           per side (0.15%); the morning report's 0.10% round trip shown for reference
  day      UTC day of the exit (realized only; open positions are not marked to market)
  outputs  per day: revenue % of equity; mean, sd, share of positive days, t-stat,
           by month; days needed to detect +0.10%/day at 80% power (two-sided 5%)
Run from trader/:  ../pyembed/python.exe _backtest_daily_revenue_baseline.py
"""
from __future__ import annotations

import collections
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trader_config as TC  # noqa: E402
TC.bot_readonly()
import config  # noqa: E402  (signal bot config, read-only)

EVENTS = TC.BOT_EVENTS
SINCE = "2026-03-01"
OUT = TC.RUNTIME / "daily_revenue_baseline.json"
FEES = {"report_0.10rt": 0.10, "bnb_0.15rt": 0.15, "taker_0.20rt": 0.20}


def dt(ts):
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def compute(events_path=None, slots=None, since=None, verbose=False) -> dict:
    """The baseline report on the given events file (the live bot keeps appending, so callers that
    compare against it -- execution_replay V1 -- must compute it on the SAME data)."""
    events_path, since = events_path or EVENTS, since or SINCE
    say = print if verbose else (lambda *a, **k: None)
    evs = collections.defaultdict(list)
    with io.open(events_path, "rb") as fh:
        for raw in fh:
            if b'"entry"' not in raw and b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") not in ("entry", "exit") or str(e.get("ts", ""))[:10] < since:
                continue
            evs[(e.get("sym"), e.get("tf"))].append((dt(e["ts"]), e))
    trades = []
    for k, es in evs.items():
        es.sort(key=lambda x: x[0])
        op = None
        for d, e in es:
            if e["event"] == "entry":
                op = (d, e)
            elif op is not None and isinstance(e.get("pnl_pct"), (int, float)):
                trades.append({"sym": k[0], "tf": k[1], "t0": op[0], "t1": d, "pnl": float(e["pnl_pct"]),
                               "mode": op[1].get("mode") or op[1].get("signal_mode")})
                op = None
    trades.sort(key=lambda t: t["t0"])
    slots = int(slots or getattr(config, "MAX_OPEN_POSITIONS", 10))
    taken, open_ = [], []
    for t in trades:
        open_ = [x for x in open_ if x["t1"] > t["t0"]]
        if len(open_) >= slots:
            continue
        open_.append(t)
        taken.append(t)
    days_all = sorted({t["t1"].strftime("%Y-%m-%d") for t in trades})
    first, last = days_all[0], days_all[-1]
    say(f"trades {len(trades)} ({first}..{last}), taken with {slots} slots: {len(taken)}, "
          f"per day {len(taken) / max(1, len(days_all)):.1f}")
    gross = [t["pnl"] for t in taken]
    say(f"per trade gross mean {sum(gross) / len(gross):+.3f}%  median {sorted(gross)[len(gross) // 2]:+.3f}%  "
          f"win {sum(1 for g in gross if g > 0) / len(gross):.1%}")
    rep = {"trades": len(trades), "taken": len(taken), "slots": slots, "from": first, "to": last,
           "per_trade_gross_mean": sum(gross) / len(gross), "fees": {}}
    for name, rt in FEES.items():
        per_day = collections.defaultdict(float)
        for t in taken:
            per_day[t["t1"].strftime("%Y-%m-%d")] += (t["pnl"] - rt) / slots
        days = [per_day.get(d, 0.0) for d in days_all]
        n = len(days)
        m = sum(days) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in days) / (n - 1))
        t_stat = m / (sd / math.sqrt(n)) if sd else 0.0
        need = math.ceil(((1.96 + 0.84) * sd / 0.10) ** 2) if sd else None
        bym = collections.defaultdict(list)
        for d, x in zip(days_all, days):
            bym[d[:7]].append(x)
        months = {mo: round(sum(v), 2) for mo, v in sorted(bym.items())}
        cum, peak, mdd = 0.0, 0.0, 0.0
        for x in days:
            cum += x
            peak = max(peak, cum)
            mdd = min(mdd, cum - peak)
        rep["fees"][name] = {"roundtrip_pct": rt, "days": n, "mean_day_pct": round(m, 4), "sd_day_pct": round(sd, 4),
                             "t": round(t_stat, 2), "positive_days": round(sum(1 for x in days if x > 0) / n, 3),
                             "total_pct": round(sum(days), 2), "max_drawdown_pct": round(mdd, 2),
                             "months_total_pct": months, "days_to_detect_0.10pp": need,
                             "per_trade_net_mean": round(sum(gross) / len(gross) - rt, 4)}
        say(f"\n[{name}] per trade net {sum(gross) / len(gross) - rt:+.3f}%  day mean {m:+.4f}%  sd {sd:.3f}%  "
              f"t {t_stat:+.2f}  positive days {rep['fees'][name]['positive_days']:.0%}  total {sum(days):+.1f}%  "
              f"max DD {mdd:.1f}%\n  months {months}\n  days to detect +0.10%/day (80% power): {need}")
    bymode = collections.defaultdict(list)
    for t in taken:
        bymode[f"{t['mode']}/{t['tf']}"].append(t["pnl"])
    rep["by_mode_gross_mean"] = {k: {"n": len(v), "mean": round(sum(v) / len(v), 3)} for k, v in
                                 sorted(bymode.items(), key=lambda kv: -len(kv[1]))}
    say("\nby mode (gross mean, n):", {k: (v["mean"], v["n"]) for k, v in list(rep["by_mode_gross_mean"].items())[:10]})
    return rep


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rep = compute(verbose=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
