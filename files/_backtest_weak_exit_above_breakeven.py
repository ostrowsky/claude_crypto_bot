"""Should a WEAK exit still fire once the trade is past break-even?

WHY THIS EXISTS

H5 hands control to the ATR trail once a position clears H5_BREAK_EVEN_PCT --
except that `_h5_should_suppress` returns False for every WEAK reason:

    if r.startswith(<warning emoji>) or "weak" in r:
        return False

That exclusion is deliberate ("WEAK signals have their own guard"), and on
2026-09-09 it cost a three-day ATOM trend: the bot entered at 1.559 -- the base
of the move -- and left on `WEAK: RSI divergence` at +0.26%, re-entered at 1.621,
left on EMA20 at -0.31%, re-entered at 1.687 and left on `WEAK: volume
exhaustion` at +2.55% with the coin at 1.73 on its way to 1.967. Three stubs
totalling +2.50% out of a +26.2% move.

THE QUESTION, ASKED SO IT CAN COME BACK NEGATIVE

For every WEAK exit that fired while the trade was already past break-even, what
would the ATR trail have returned instead? The trail is not "hold forever" -- it
still exits, just on price rather than on a momentum pattern. So this measures
one substitution, not the removal of an exit.

The honest counter-hypothesis is that WEAK is a good early warning: it fires
before a reversal the trail would only catch after giving back the buffer. That
is exactly what the `worse` column below counts, and if it dominates, the answer
is no and this file records it.

METHOD

  population   every exit event whose reason is WEAK and whose realized
               pnl_pct >= the H5 break-even floor. The bot's OWN trades (TH-06).
  actual       the pnl the bot booked.
  counterfactual  the trade stays open at the exit bar and an ATR trail runs
               from there: stop = peak * (1 - width), width = max(trail_k*ATR%,
               the mode's TRAIL_MIN_BUFFER_PCT floor). The peak is seeded with
               the running high since entry (bars_held back), because the real
               trail would have been tracking it all along.
  ordering     the stop is tested against this bar's LOW *before* the peak
               absorbs this bar's HIGH. Doing it the other way lets a bar's own
               spike widen the stop that the same bar then fails to hit, which
               inflates every result silently.
  control      WEAK exits BELOW break-even -- H5 would not touch them, so any
               "improvement" there is the market rising, not the rule working.

Base rates and n travel with every ratio (TH-01); results are broken out by
month so a single regime cannot carry the verdict. There is no fitted model
here to hold out, so stability across months replaces a train/test split.

VERDICT 2026-09-09: REFUTED. Do not re-test this without new evidence.

Maximum period, 2026-03-03 .. 2026-09-09, 4898 exits, 1271 of them WEAK (26%),
846 above break-even, 699 resolvable against klines:

    group                    n   actual med   trail med    delta   better   avg
    WEAK, above break-even  699      1.36%       0.86%     -0.79%    34%   -0.21%
    WEAK, below (control)   370      0.17%      -0.51%     -0.33%    38%   -0.11%

The trail made 460 of 699 trades WORSE (66%), -931.5% against +786.7% of gains:
net -144.8%, or -0.207% per trade. Negative in all five entry modes and in six
months of seven. WEAK is an early warning that works, and the exclusion inside
`_h5_should_suppress` is correct.

The one dissenting cell was 2026-09 (n=28, +1.97% per trade, 54% better) -- the
regime that raised the question. Tested as a regime effect against the RM-22
precedent and it does NOT survive: btc_up -0.22% and btc_dn -0.54% carry the
same sign, no flip, and September's btc_up cell is n=16 -- below the n>=20 bar
this file set for itself before the numbers were seen. 2026-05 btc_up was also
positive (+0.75%, n=46) while 2026-05 btc_dn was -0.84%, so the pattern is
inconsistent rather than regime-conditional.

What this does NOT settle: ATOM really did give up ~24pp, and the exits above
are graded on the trades the bot took (TH-06). The leak is real; substituting
the ATR trail is not the fix for it.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_trend_start_detector as TD  # noqa: E402

EVENTS = HERE / "bot_events.jsonl"
_IDX: dict = {}

WARN = "⚠"


def index_for(sym: str, tf: str):
    key = (sym, tf)
    if key not in _IDX:
        try:
            bars = TD.bars_15m(sym) if tf == "15m" else TD.load_bars(sym, "1h")
        except Exception:
            bars = []
        _IDX[key] = (bars, {b[0]: i for i, b in enumerate(bars)})
    return _IDX[key]


def atr_pct(bars, i, n=14):
    """ATR as a percentage of close, over the n bars ending at i."""
    if i < n:
        return None
    trs = []
    for k in range(i - n + 1, i + 1):
        hi, lo, prev_c = bars[k][2], bars[k][3], bars[k - 1][4]
        trs.append(max(hi - lo, abs(hi - prev_c), abs(lo - prev_c)))
    close = bars[i][4]
    if close <= 0 or not trs:
        return None
    return 100.0 * (sum(trs) / len(trs)) / close


def min_buffer_pct(cfg, mode):
    if not getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", True):
        return 0.0
    key = "TRAIL_MIN_BUFFER_PCT_" + str(mode or "").upper()
    return 100.0 * float(getattr(cfg, key, 0.0))


def replay(bars, i_exit, i_entry, entry_price, trail_k, mode, cfg, max_bars):
    """Return the pnl% the ATR trail would have booked, or None if unresolvable."""
    a = atr_pct(bars, i_exit)
    if a is None:
        return None
    width = max(float(trail_k) * a, min_buffer_pct(cfg, mode))
    if width <= 0:
        return None
    # seed the peak with the high the trail would already have been tracking
    peak = max((b[2] for b in bars[max(0, i_entry):i_exit + 1]), default=bars[i_exit][2])
    last = i_exit
    for j in range(i_exit + 1, min(len(bars), i_exit + 1 + max_bars)):
        hi, lo = bars[j][2], bars[j][3]
        stop = peak * (1.0 - width / 100.0)
        if lo <= stop:                      # tested BEFORE the peak sees this bar
            return (stop / entry_price - 1.0) * 100.0
        peak = max(peak, hi)
        last = j
    if last <= i_exit:
        return None
    return (bars[last][4] / entry_price - 1.0) * 100.0   # still open at the cap


def load_exits(since):
    out = []
    with io.open(EVENTS, "rb") as fh:
        for raw in fh:
            if b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") != "exit":
                continue
            ts = str(e.get("ts") or "")
            if ts < since:
                continue
            if not all(isinstance(e.get(k), (int, float))
                       for k in ("entry_price", "exit_price", "pnl_pct", "bars_held")):
                continue
            if not e.get("sym") or e["entry_price"] <= 0:
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            r = str(e.get("reason") or "")
            e["_weak"] = r.startswith(WARN) or "weak" in r.lower()
            e["_dt"] = d
            out.append(e)
    out.sort(key=lambda r: r["_dt"])
    return out


def stats(name, pairs):
    """pairs = [(actual, counterfactual)]"""
    if not pairs:
        print("%-30s%7s" % (name, "n=0"))
        return None
    d = sorted(cf - ac for ac, cf in pairs)
    ac = sorted(a for a, _ in pairs)
    cf = sorted(c for _, c in pairs)

    def med(v):
        return v[len(v) // 2]

    better = sum(1 for x in d if x > 0)
    print("%-30s%7d%11.2f%%%12.2f%%%12.2f%%%10.0f%%%11.2f%%" % (
        name, len(pairs), med(ac), med(cf), med(d),
        100.0 * better / len(pairs),
        sum(c - a for a, c in pairs) / len(pairs)))
    return med(d)



_BTC = {}


def btc_regime(dt):
    """BTC above or below its own EMA50 on the 1h series at `dt`.

    The project has a precedent for this exact question (RM-22 step A): the
    forward returns of gate-blocked candidates FLIP SIGN across regime cells,
    spread 0.548pp. So before concluding that an exit rule is right or wrong on
    average, it is worth asking whether the average is hiding two populations.
    Returns "btc_up", "btc_dn", or None when the series does not reach.
    """
    if not _BTC:
        bars = TD.load_bars("BTCUSDT", "1h")
        ema, k, out = None, 2.0 / 51.0, {}
        for b in bars:
            ema = b[4] if ema is None else (b[4] - ema) * k + ema
            out[b[0]] = b[4] > ema
        _BTC.update(out)
        if not _BTC:
            _BTC[None] = None
    hour = dt.replace(minute=0, second=0, microsecond=0)
    v = _BTC.get(hour)
    return None if v is None else ("btc_up" if v else "btc_dn")


def bar_key(dt, tf):
    if tf == "15m":
        return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--max-bars", type=int, default=96,
                    help="cap on how long the trail may hold after the WEAK exit")
    ap.add_argument("--break-even", type=float, default=None,
                    help="override H5_BREAK_EVEN_PCT")
    args = ap.parse_args()

    import config as cfg
    be = args.break_even if args.break_even is not None else float(
        getattr(cfg, "H5_BREAK_EVEN_PCT", 0.5))

    rows = load_exits(args.since)
    if not rows:
        print("no exits in window")
        return
    print("exit events since %s: %d  (%s .. %s)"
          % (args.since, len(rows), rows[0]["_dt"].date(), rows[-1]["_dt"].date()))
    weak = [r for r in rows if r["_weak"]]
    print("WEAK exits: %d (%.0f%% of all exits) -- base rate for every ratio below"
          % (len(weak), 100.0 * len(weak) / max(1, len(rows))))
    above = [r for r in weak if r["pnl_pct"] >= be]
    below = [r for r in weak if r["pnl_pct"] < be]
    print("of those, %d are at or above break-even (%.1f%% of WEAK) -- the ones H5 "
          "would suppress; %d are below and form the control"
          % (len(above), 100.0 * len(above) / max(1, len(weak)), len(below)))
    print("break-even floor in force: %.2f%%, trail capped at %d bars"
          % (be, args.max_bars))

    def build(group):
        out, unresolved = [], 0
        for e in group:
            tf = str(e.get("tf") or "1h")
            bars, idx = index_for(e["sym"], tf)
            if not bars:
                unresolved += 1
                continue
            i = idx.get(bar_key(e["_dt"], tf))
            if i is None:
                unresolved += 1
                continue
            cf = replay(bars, i, i - int(e["bars_held"]), float(e["entry_price"]),
                        float(e.get("trail_k") or 2.0), e.get("mode"), cfg,
                        args.max_bars)
            if cf is None:
                unresolved += 1
                continue
            out.append((float(e["pnl_pct"]), cf, e))
        return out, unresolved

    res_a, miss_a = build(above)
    res_b, miss_b = build(below)
    print("resolved against klines: %d of %d above, %d of %d below "
          "(the rest fall outside the cached window)"
          % (len(res_a), len(above), len(res_b), len(below)))
    if not res_a:
        print("nothing resolvable -- refresh the kline cache and rerun")
        return

    hdr = "%-30s%7s%12s%13s%13s%10s%12s" % (
        "group", "n", "actual med", "trail med", "delta med", "better", "delta avg")
    print()
    print("=" * 100)
    print("WOULD THE ATR TRAIL HAVE DONE BETTER?")
    print("=" * 100)
    print(hdr)
    print("-" * 100)
    stats("WEAK, above break-even", [(a, c) for a, c, _ in res_a])
    stats("WEAK, below (control)", [(a, c) for a, c, _ in res_b])
    print("-" * 100)
    print("'better' = share of trades where the trail beat the WEAK exit.")
    print("The control is NOT a target: H5 leaves those alone. It is here to show")
    print("how much of any gain is simply a market that kept rising.")

    print()
    print("=" * 100)
    print("BY MODE  (a rule that only works in one mode is a mode-level lever)")
    print("=" * 100)
    print(hdr)
    print("-" * 100)
    by_mode = collections.defaultdict(list)
    for a, c, e in res_a:
        by_mode[str(e.get("mode"))].append((a, c))
    for m, v in sorted(by_mode.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 10:
            stats(m, v)

    print()
    print("=" * 100)
    print("BY MONTH  (one regime must not carry the verdict)")
    print("=" * 100)
    print(hdr)
    print("-" * 100)
    by_month = collections.defaultdict(list)
    for a, c, e in res_a:
        by_month[e["_dt"].strftime("%Y-%m")].append((a, c))
    for m in sorted(by_month):
        if len(by_month[m]) >= 10:
            stats(m, by_month[m])

    print()
    print("=" * 100)
    print("THE COUNTER-HYPOTHESIS: is WEAK an early warning?")
    print("=" * 100)
    hurt = sorted(c - a for a, c, _ in res_a if c < a)
    gain = sorted(c - a for a, c, _ in res_a if c > a)
    if hurt:
        print("trades the trail made WORSE: %d of %d (%.0f%%)"
              % (len(hurt), len(res_a), 100.0 * len(hurt) / len(res_a)))
        print("   median damage %.2f%%, worst %.2f%%, total %.1f%%"
              % (hurt[len(hurt) // 2], hurt[0], sum(hurt)))
    if gain:
        print("trades the trail made BETTER: %d, median gain %.2f%%, best %.2f%%, "
              "total %.1f%%"
              % (len(gain), gain[len(gain) // 2], gain[-1], sum(gain)))
    print("net across the whole population: %+.1f%% over %d trades (%+.3f%% per trade)"
          % (sum(c - a for a, c, _ in res_a), len(res_a),
             sum(c - a for a, c, _ in res_a) / len(res_a)))
    print("If the damage outweighs the gain, WEAK is doing its job, the hypothesis")
    print("is dead, and that is the result to record.")

    print()
    print("=" * 100)
    print("BY REGIME  (does the September flip survive as a regime effect?)")
    print("=" * 100)
    print(hdr)
    print("-" * 100)
    by_reg = collections.defaultdict(list)
    for a, c, e in res_a:
        reg = btc_regime(e["_dt"])
        if reg:
            by_reg[reg].append((a, c, e))
    for reg in ("btc_up", "btc_dn"):
        v = by_reg.get(reg) or []
        if len(v) >= 10:
            stats(reg, [(a, c) for a, c, _ in v])
    print("-" * 100)
    print("If the sign flips between these two rows, the average above was hiding")
    print("two populations and the lever is regime-conditional, not on/off.")
    print()
    print("The same cells month by month -- a regime effect must hold in more than")
    print("the one month that raised the question:")
    print()
    print("%-12s%10s%9s%12s%10s%9s%12s" % (
        "month", "up n", "up dlt", "up better", "dn n", "dn dlt", "dn better"))
    print("-" * 76)
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for reg, v in by_reg.items():
        for a, c, e in v:
            cells[e["_dt"].strftime("%Y-%m")][reg].append(c - a)
    for m in sorted(cells):
        up = cells[m].get("btc_up") or []
        dn = cells[m].get("btc_dn") or []
        f = lambda v: (len(v),
                       sum(v) / len(v) if v else float("nan"),
                       100.0 * sum(1 for x in v if x > 0) / len(v) if v else float("nan"))
        nu, du, bu = f(up)
        nd, dd, bd = f(dn)
        print("%-12s%10d%+8.2f%%%10.0f%%%10d%+8.2f%%%10.0f%%"
              % (m, nu, du, bu, nd, dd, bd))
    print("-" * 76)
    print("n below ~20 in a cell is an anecdote; the September question came from")
    print("28 trades in one month and cannot be settled by another small cell.")

    print()
    print("READ THIS")
    print("  Replaying one exit rule on trades the bot actually took (TH-06) says")
    print("  nothing about trades it never entered.")
    print("  The trail still exits -- on price instead of on a pattern. This is a")
    print("  substitution, not the removal of a stop.")
    print("  The %d-bar cap bounds the counterfactual; a longer cap flatters it."
          % args.max_bars)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
