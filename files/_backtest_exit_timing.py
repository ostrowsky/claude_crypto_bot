"""Did the exit come before the uptrend ended, and how much was left?

Goal 3 — signal exit only just before the uptrend ends. Exiting early is a loss,
not prudence, so the question is not "was the trade green" but "did the trend
continue after we left".

Two measures, because they answer different things and disagree usefully:

* **against the trend** — the matched ZigZag uptrend's end. This is goal 3 in
  its own words: an exit before `trend.end_ts` is premature by definition, and
  the gap says by how much.
* **against the rest of the day** — the highest price between the exit and the
  UTC close. Cruder, but it needs no trend to be matched, so it covers the rows
  the labeler cannot match and keeps the sample from being silently selective.

Restricted to winner-days (global top-20 INTERSECT watchlist from the immutable
store), because that is the population the North Star's capture factor averages
over. Every number carries its n; per-class medians on five trades are not
findings and are printed as such.

  pyembed\\python.exe files\\_backtest_exit_timing.py

Spec: docs/specs/features/exit-timing-spec.md
"""
from __future__ import annotations

import csv
import io
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import immutable_labels as IL  # noqa: E402

try:
    from zigzag_labeler import detect_uptrends
    ZIGZAG = True
except Exception:                                    # pragma: no cover
    ZIGZAG = False

HISTORY = ROOT / "history"
MIN_N_TO_REPORT = 10          # below this a class median is noise, and says so
_BARS: dict = {}


def bars(sym: str, tf: str) -> list:
    """[(ts, high, low, close)] from the kline cache, empty when absent."""
    key = (sym, tf)
    if key in _BARS:
        return _BARS[key]
    path = HISTORY / f"{sym}_{tf}.csv"
    out = []
    if path.exists():
        with io.open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    out.append((datetime.fromisoformat(r["ts"]), float(r["high"]),
                                float(r["low"]), float(r["close"])))
                except (KeyError, ValueError):
                    continue
    _BARS[key] = out
    return out


def bars_covering(sym: str, tf: str, day: str) -> list:
    """Bars from whichever cache actually spans `day`.

    `bars(sym, tf) or bars(sym, "1h")` falls back only when the file is MISSING.
    A 15m cache that exists but holds the last 30 days returns a non-empty list
    for a trade in June, so the fallback never fired and the row silently had no
    forward path: 318 of 423 trades are on 15m, and only 81 ended up with any
    post-exit data at all. Pick by coverage, not by existence.
    """
    want = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    for candidate in (tf, "1h", "15m"):
        b = bars(sym, candidate)
        if b and b[0][0] <= want <= b[-1][0] + timedelta(days=1):
            return b
    return []


def exit_class(reason: str) -> str:
    r = (reason or "").lower()
    for needle, name in (("atr", "atr_trail"), ("трейл", "atr_trail"),
                         ("trail", "atr_trail"), ("max_hold", "time_max_hold"),
                         ("время", "time_max_hold"), ("лимит", "time_max_hold"),
                         ("ema20", "ema20_weakness"), ("ema 20", "ema20_weakness"),
                         ("rsi", "rsi"), ("macd", "macd")):
        if needle in r:
            return name
    return "other"


def load_trades(winners: set, winners_only: bool = True) -> list:
    """Entry/exit pairs. `winners_only=False` returns every trade.

    The distinction is load-bearing. Replaying "what if we had held longer" on
    winner-days ALONE conditions the sample on the outcome the policy cannot
    know: of course holding wins where the coin ended in the day's top-20. That
    is how this backtest first produced "an 8% trail triples capture and raises
    P&L", flatly contradicting the live 2026-06-05 rollback of exactly that
    change. The policy applies to every trade, so it must be judged on every
    trade (TH-06).
    """
    opens: dict = {}
    rows = []
    with io.open(HERE / "bot_events.jsonl", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"event"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev, sym = e.get("event"), e.get("sym")
            if ev not in ("entry", "exit") or not sym:
                continue
            try:
                dt = datetime.fromisoformat(str(e.get("ts", "")).replace("Z", "+00:00"))
            except ValueError:
                continue
            if ev == "entry":
                opens[sym] = (dt, float(e.get("price") or 0), e.get("tf"),
                              e.get("mode"))
                continue
            opened = opens.pop(sym, None)
            if not opened:
                continue
            exit_px = float(e.get("exit_price") or e.get("price") or 0)
            if opened[1] <= 0 or exit_px <= 0:
                continue
            day = opened[0].strftime("%Y-%m-%d")
            if winners_only and (day, sym) not in winners:
                continue
            rows.append({"day": day, "sym": sym, "entry_dt": opened[0],
                         "exit_dt": dt, "entry": opened[1], "exit": exit_px,
                         "tf": opened[2] or "1h", "mode": opened[3],
                         "cls": exit_class(e.get("reason"))})
    return rows


def measure(trade: dict) -> dict | None:
    """Left-on-the-table against the rest of the day and against the trend."""
    b = bars_covering(trade["sym"], trade["tf"], trade["day"])
    if not b:
        return {"why": "no_klines"}
    day_end = (datetime.strptime(trade["day"], "%Y-%m-%d")
               .replace(tzinfo=timezone.utc) + timedelta(days=1))
    after = [h for t, h, _l, _c in b if trade["exit_dt"] < t < day_end]
    out = {"why": "measured",
           "realized_pct": (trade["exit"] / trade["entry"] - 1) * 100.0}
    if after:
        out["left_day_pct"] = (max(after) / trade["exit"] - 1) * 100.0
    if not ZIGZAG:
        return out

    trends = detect_uptrends([{"ts": t, "open": c, "high": h, "low": l, "close": c}
                              for t, h, l, c in b],
                             symbol=trade["sym"], swing_pct=4.0,
                             max_drawdown_pct=2.0, min_duration_bars=4)
    covering = [tr for tr in trends
                if tr.start_ts <= trade["exit_dt"] <= tr.end_ts]
    if covering:
        tr = covering[0]
        out["exited_inside_trend"] = True
        out["minutes_before_trend_end"] = round(
            (tr.end_ts - trade["exit_dt"]).total_seconds() / 60.0, 1)
        peak = [h for t, h, _l, _c in b if trade["exit_dt"] < t <= tr.end_ts]
        if peak:
            out["left_trend_pct"] = (max(peak) / trade["exit"] - 1) * 100.0
    else:
        out["exited_inside_trend"] = False
    return out


def replay_trailing(trade: dict, trail_pct: float) -> float | None:
    """P&L if we had ignored the exit and trailed from there to the day's close.

    A counterfactual, not a promise: it replays the ACTUAL forward price path
    under one policy. It cannot know that holding would not have changed what
    the bot did elsewhere (a slot stays occupied), and it silently assumes the
    stop fills at the trigger.
    """
    b = bars_covering(trade["sym"], trade["tf"], trade["day"])
    if not b:
        return None
    day_end = (datetime.strptime(trade["day"], "%Y-%m-%d")
               .replace(tzinfo=timezone.utc) + timedelta(days=1))
    path = [(t, h, l, c) for t, h, l, c in b if trade["exit_dt"] < t < day_end]
    if not path:
        return None
    peak = trade["exit"]
    for _t, high, low, close in path:
        stop = peak * (1.0 - trail_pct / 100.0)
        if low <= stop:                       # stop hit inside this bar
            return (stop / trade["entry"] - 1) * 100.0
        peak = max(peak, high)
    return (path[-1][3] / trade["entry"] - 1) * 100.0


def counterfactual(measured: list, cls: str | None = None,
                   title: str | None = None) -> None:
    """Replay: hold past the exit and trail instead. `cls=None` = every exit.

    Reported in CAPTURE as well as P&L, because goal 3 is "exit just before the
    trend ends" and the North Star multiplies capture — not profit. The
    2026-06-05 widening of the impulse_speed trail to 8% was rolled back on a
    P&L basis (-54.9% over five days); that is the wrong criterion for this goal
    and the two columns are printed side by side so the trade is visible rather
    than argued.
    """
    rows = [t for t in measured if cls is None or t["cls"] == cls]
    print()
    if len(rows) < MIN_N_TO_REPORT:
        print("  ema20_weakness counterfactual: n=%d < %d, not reported"
              % (len(rows), MIN_N_TO_REPORT))
        return
    actual = [t["realized_pct"] for t in rows]
    print("  Counterfactual - %s, trail instead (n=%d)"
          % (title or ("suppress " + (cls or "every exit")), len(rows)))
    # n per row, because a "beats actual 100%" computed over six replayable
    # trades is not the same claim as one computed over fifty-six (TH-01).
    print("  %-22s%6s%13s%12s%10s%11s%14s"
          % ("policy", "n", "median pnl%", "mean pnl%", "capture", "win rate",
             "beats actual"))
    def capture_of(realized, left):
        """Share of the available move that was taken — only meaningful when
        the population made money overall. With a negative realized total the
        ratio flips sign and prints things like "-5.8% capture", which is not a
        smaller capture but an undefined one."""
        tot = realized + left
        if realized <= 0 or tot <= 0:
            return None
        return 100.0 * realized / tot
    act_real = sum(actual)
    act_left = sum(t.get("left_day_pct", 0.0) for t in rows)
    def cap_str(v):
        return "     n/a" if v is None else "%8.1f%%" % v
    print("  %-22s%6d%12.2f%%%11.2f%%%s%10.0f%%%14s"
          % ("actual exit", len(actual), st.median(actual),
             sum(actual) / len(actual), cap_str(capture_of(act_real, act_left)),
             100 * sum(1 for x in actual if x > 0) / len(actual), "-"))
    for trail in (1.5, 3.0, 5.0, 8.0):
        got = [(t, replay_trailing(t, trail)) for t in rows]
        got = [(t, v) for t, v in got if v is not None]
        if not got:
            continue
        vals = [v for _t, v in got]
        better = sum(1 for t, v in got if v > t["realized_pct"])
        # Capture under the policy: what it took of what was still ahead at
        # the ORIGINAL exit, so the denominator is the same for every row.
        pol_real = sum(vals)
        pol_left = sum(max(0.0, t.get("left_day_pct", 0.0)
                           - (v - t["realized_pct"])) for t, v in got)
        print("  %-22s%6d%12.2f%%%11.2f%%%s%10.0f%%%13.0f%%"
              % ("trail %.1f%%" % trail, len(got), st.median(vals),
                 sum(vals) / len(vals), cap_str(capture_of(pol_real, pol_left)),
                 100 * sum(1 for x in vals if x > 0) / len(vals),
                 100 * better / len(got)))
    print("  A counterfactual replays the real forward path under one policy. It")
    print("  cannot account for the slot staying occupied, and assumes the stop")
    print("  fills at the trigger - both flatter holding.")


def main() -> int:
    watchlist = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
    winners, _ = IL.winners_by_day(top_n=20, watchlist=watchlist,
                                   rank_before_filter=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-trades", action="store_true",
                    help="every trade, not just winner-days (the honest "
                         "population for a policy comparison)")
    args = ap.parse_args()
    trades = load_trades(winners, winners_only=not args.all_trades)
    print("=" * 78)
    print("Exit timing — goal 3 — population: %s"
          % ("ALL trades" if args.all_trades else "winner-days only"))
    print("=" * 78)
    print(f"trades on winner-days   {len(trades)}")

    measured, no_klines = [], 0
    for t in trades:
        m = measure(t)
        if m is None or m.get("why") == "no_klines":
            no_klines += 1
            continue
        t.update(m)
        measured.append(t)
    print(f"with price data after the exit   {len(measured)}   "
          f"(no klines: {no_klines})")
    if not measured:
        print("nothing to measure")
        return 1

    inside = [t for t in measured if t.get("exited_inside_trend")]
    print(f"exited INSIDE a detected uptrend {len(inside)} "
          f"({100*len(inside)/len(measured):.0f}%) — premature by goal 3's own "
          f"definition")

    by_cls = defaultdict(list)
    for t in measured:
        by_cls[t["cls"]].append(t)
    print(f"\n  {'exit class':<18}{'n':>5}{'realized%':>11}{'left in day%':>14}"
          f"{'left in trend%':>16}{'inside trend':>14}")
    for cls, rows in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
        rea = st.median([r["realized_pct"] for r in rows])
        day = [r["left_day_pct"] for r in rows if "left_day_pct" in r]
        trd = [r["left_trend_pct"] for r in rows if "left_trend_pct" in r]
        ins = sum(1 for r in rows if r.get("exited_inside_trend"))
        thin = "  (n<%d: noise)" % MIN_N_TO_REPORT if len(rows) < MIN_N_TO_REPORT else ""
        # Each median carries the n it was computed on, not the class total:
        # printing the class n beside a median taken over a fifth of it is the
        # ratio-without-its-base-rate failure this repo has a rule about.
        print("  %-18s%5d%10.2f%%%7d%12.2f%%%7d%13.2f%%%8.0f%%%s"
              % (cls, len(rows), rea, len(day),
                 st.median(day) if day else float("nan"), len(trd),
                 st.median(trd) if trd else float("nan"),
                 100 * ins / len(rows), thin))

    tot_real = sum(t["realized_pct"] for t in measured)
    tot_left = sum(t.get("left_day_pct", 0.0) for t in measured)
    denom = tot_real + tot_left
    print(f"\n  OVERALL realized {tot_real:.0f}%, left after exit {tot_left:.0f}%"
          f"  ->  captured {100*tot_real/denom if denom else 0:.1f}% of what was "
          f"still ahead")
    print()
    counterfactual(measured, cls=None, title="hold every exit")
    counterfactual(measured, cls="atr_trail")
    counterfactual(measured, cls="ema20_weakness")
    print()
    print("Reading it: 'left in trend' is the goal-3 number — how much the coin")
    print("still had inside the uptrend we were already in when we left. 'left in")
    print("day' is the cruder version that needs no trend match, kept so the")
    print("sample is not silently restricted to matchable rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
