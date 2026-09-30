"""Late-entry fix candidate: enter AT the +2.5% crossing when the coin already leads.

Anatomy (_research_late_entry.py, 531 winner-days 2026-03-01..09-28): 56% of
winner-days are entered late -- median 3.1 h after the first +2.5% close, with
46% of the move done (early entries: 30%); in 70% of them no entry rule fired
before the crossing, so no gate change can reach them. Prediction of the start
has failed five times (trend-start research). The remaining lever is the DELAY:
act at the crossing itself instead of ~3 h after it.

Rule (variants fixed before the run): at the close of the first 15m bar whose
close is >= +2.5% above the UTC open, enter if the coin's rank by return since
the open across the watchlist is <= K (V0: no rank filter; K = 10, 5, 3).
One signal per coin per day. Days with the bot up all day, 2026-03-01 onward.

Exit: the live 15m exit stack as replayed by exit_validator -- calibrated trail
(k = median trail_k of real impulse_speed/15m entries, impulse_speed floor) with
the live leader mode (X-9c: rank <= 5 at >= +3% -> wide trail, exit on losing
the day's top-10, 7-day cap).

Measured per variant, beside the bot's own 15m entries over the same days:
  signals/day, precision = share on immutable top-20 winner-days (base rate and
  lift beside it), rocket share, mean pnl per trade with 95% bootstrap,
  winner-day reach (share of winner-days with a signal), and of the winner-days
  the bot entered LATE or never: how many the rule enters at the crossing, and
  the share of the move done there vs at the bot's actual late entry.

Pre-registered verdict (operator: "no junk signals"):
  supported  precision >= the bot's own 15m entries AND mean pnl lower 95% bound
             >= bot mean - 0.10 pp AND it reaches >= 10% of the late/none days
  refuted    otherwise
"""
import collections
import io
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import exit_validator as EV  # noqa: E402
import _compute_early_capture as E  # noqa: E402
import immutable_labels as IL  # noqa: E402

START = "2026-03-01"
KS = (None, 10, 5, 3)
TRIG = 0.025
UTC = timezone.utc


def sim_exit(g, j, ie, k, floor, P):
    """Calibrated trail from the entry close, switching to leader mode when the rule says so."""
    C, A, R = g.C, g.ATR, g.RANK
    ep = C[ie, j]
    stop = ep - max(k * A[ie, j] if math.isfinite(A[ie, j]) else 0.0, floor * ep)
    last = min(g.N - 1, ie + 96)
    lead = g.lead(P["rank_max"], P["min_ret"])
    for q in range(ie, last + 1):
        if lead[q, j]:
            t = {"j": j, "ep": ep, "k": k}
            return EV.wide_from(g, t, q, P), True
        if q == ie:
            continue
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(k * A[q, j], floor * cq))
        if cq < stop:
            return (cq / ep - 1) * 100, False
    return (C[last, j] / ep - 1) * 100, False


def ci(v, rnd):
    bs = sorted(np.mean(rnd.choices(v, k=len(v))) for _ in range(1000))
    return bs[25], bs[-26]


def main():
    g = EV.load_grid(cfg.ATR_PERIOD)
    P = EV.leader_params(cfg)
    wl = E.load_watchlist()
    full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=UTC))
    win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
    winners = set(win)
    rockets, _ = EV.day_labels()
    trades = EV.load_trades(g, START)
    ks = sorted(t["k"] for t in trades if t["mode"] == "impulse_speed")
    K = ks[len(ks) // 2] if ks else 2.0
    floor = EV.floor_of("impulse_speed", cfg)
    print("grid %d coins, bars until %s; trail k=%.2f floor %.3f; leader %s" % (
        len(g.syms), g.last_bar, K, floor, {k: P[k] for k in ("rank_max", "min_ret", "floor")}))

    # the bot's own 15m entries on the same days (live exits as replayed = the real exit unless a switch)
    bot_days = [t for t in trades if t["day"] in full]
    bot_vals, _ = EV.leader_arm(g, bot_days, P)
    bot = [(t, v) for t, v in zip(bot_days, bot_vals) if v is not None]
    bot_prec = sum(1 for t, _ in bot if (t["day"], t["sym"]) in winners) / len(bot)
    bot_mean = float(np.mean([v for _, v in bot]))
    ndays = len({t["day"] for t, _ in bot})

    # late / never-entered winner-days from the anatomy
    rows = json.load(io.open(FILES.parent / ".runtime/backtests/late_entry_rows.json", encoding="utf-8"))
    late_like = {(r["day"], r["sym"]): r for r in rows if r["stage"] in ("late", "none")}

    # crossings: first 15m close >= +2.5% of the UTC open, per coin per day
    cands = []
    for d in range(g.days):
        day = (g.t0 + timedelta(days=d)).strftime("%Y-%m-%d")
        if day < START or day not in full:
            continue
        seg = g.RET[d * 96:(d + 1) * 96]
        for j, sym in enumerate(g.syms):
            hit = np.where(seg[:, j] >= TRIG)[0]
            if len(hit) == 0:
                continue
            i = d * 96 + int(hit[0])
            if i + 1 >= g.N:
                continue
            cands.append({"day": day, "sym": sym, "j": j, "i": i, "rank": int(g.RANK[i, j]),
                          "hour": int(hit[0]) / 4.0})
    base_prec = sum(1 for c in cands if (c["day"], c["sym"]) in winners) / len(cands)
    print("coin-days crossing +2.5%%: %d on %d days (%.1f/day); share on winner-days (base) %.3f" % (
        len(cands), len({c['day'] for c in cands}), len(cands) / max(1, len({c['day'] for c in cands})), base_prec))
    print("BOT 15m entries on the same days: n=%d (%.1f/day), precision %.3f, mean %+.3f%%" % (
        len(bot), len(bot) / max(1, ndays), bot_prec, bot_mean))

    rnd = random.Random(7)
    print("\n%-8s %6s %7s %9s %6s %7s %22s %9s %16s %18s" % (
        "variant", "n", "per day", "precision", "lift", "rocket", "mean pnl [95%]", "reach", "late/none->cross", "move done @cross"))
    verdicts = {}
    for Kr in KS:
        sel = [c for c in cands if Kr is None or c["rank"] <= Kr]
        vals = []
        for c in sel:
            v, _ = sim_exit(g, c["j"], c["i"], K, floor, P)
            c["pnl"] = v
            vals.append(v)
        prec = sum(1 for c in sel if (c["day"], c["sym"]) in winners) / len(sel)
        rk = sum(1 for c in sel if (c["day"], c["sym"]) in rockets) / len(sel)
        lo, hi = ci(vals, rnd)
        reach = sum(1 for c in sel if (c["day"], c["sym"]) in winners)
        hit_late = [c for c in sel if (c["day"], c["sym"]) in late_like]
        done_cross, done_bot = [], []
        for c in hit_late:
            r = late_like[(c["day"], c["sym"])]
            j = c["j"]
            op_i = (c["i"] // 96) * 96
            px = g.C[c["i"], j]
            # move low as in the anatomy: min low since D-1 00:00 up to the day high
            lo_i = max(0, op_i - 96)
            day_hi_i = op_i + int(np.nanargmax(g.H[op_i:op_i + 96, j]))
            low = np.nanmin(g.C[lo_i:day_hi_i + 1, j])
            hi_px = np.nanmax(g.H[op_i:op_i + 96, j])
            if hi_px > low:
                done_cross.append((px - low) / (hi_px - low))
            if r.get("entry_done") is not None:
                done_bot.append(r["entry_done"])
        name = "V0 all" if Kr is None else "rank<=%d" % Kr
        dc = "%.0f%% vs bot %.0f%%" % (100 * np.median(done_cross), 100 * np.median(done_bot)) if done_cross and done_bot else "-"
        print("%-8s %6d %7.1f %9.3f %6.2f %7.3f %+7.3f [%+.3f,%+.3f] %4d/%-4d %8d/%-6d %18s" % (
            name, len(sel), len(sel) / max(1, len({c['day'] for c in cands})), prec, prec / base_prec, rk,
            np.mean(vals), lo, hi, reach, len([w for w in winners if w[0] >= START and w[0] in full]),
            len(hit_late), len(late_like), dc))
        ok = prec >= bot_prec and lo >= bot_mean - 0.10 and len(hit_late) >= 0.10 * len(late_like)
        verdicts[name] = "SUPPORTED" if ok else "refuted"
        bym = collections.defaultdict(list)
        for c in sel:
            bym[c["day"][:7]].append(c["pnl"])
        print("         by month: " + "  ".join("%s %+.2f(n=%d)" % (m[5:], np.mean(v), len(v)) for m, v in sorted(bym.items())))
    print("\nVERDICT (precision >= bot %.3f, pnl lower bound >= bot %+.3f - 0.10, reach >= 10%% of %d late/none days):" % (
        bot_prec, bot_mean, len(late_like)))
    for k, v in verdicts.items():
        print("  %-8s %s" % (k, v))


if __name__ == "__main__":
    main()
