"""Third look at the crossing entry (reported as such): a market-regime filter,
chosen on one period and judged on another.

The rank<=3/5 crossing entries lost in every month except September (+1.12 /
+0.55 %/trade), which reads as regime. A filter picked by looking at all months
would be curve-fitting, so:

  family   breadth = share of watchlist coins above their UTC open at the entry
           bar, >= 0.4 / 0.5 / 0.6 / 0.7; BTC return since the open >= -inf / 0 /
           +1% -> 12 filters x rank<=3, rank<=5
  choose   on TRAIN = 2026-03-01 .. 06-30 only: highest mean pnl with n >= 100
  judge    on TEST = 2026-07-01 .. last day, never used for the choice, with the
           parent's pre-registered criterion against the bot's own 15m entries
           in the TEST period: precision >= bot, pnl lower 95% >= bot mean - 0.10
           pp, reach >= 10% of the TEST late/none winner-days
Exit: the parent's live stack (calibrated trail + leader-mode switch).
"""
import io
import json
import random
import sys
from datetime import timedelta, datetime, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import exit_validator as EV  # noqa: E402
import _compute_early_capture as E  # noqa: E402
import immutable_labels as IL  # noqa: E402
import _backtest_crossing_entry as X  # noqa: E402

TRAIN_END = "2026-07-01"
BREADTH = (0.4, 0.5, 0.6, 0.7)
BTC = (None, 0.0, 0.01)


def main():
    g = EV.load_grid(cfg.ATR_PERIOD)
    P = EV.leader_params(cfg)
    wl = E.load_watchlist()
    full, _, _ = E.load_uptime(datetime.strptime(X.START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
    winners = set(win)
    trades = EV.load_trades(g, X.START)
    ks = sorted(t["k"] for t in trades if t["mode"] == "impulse_speed")
    K = ks[len(ks) // 2]
    floor = EV.floor_of("impulse_speed", cfg)
    jb = g.col.get("BTCUSDT")
    rows = json.load(io.open(FILES.parent / ".runtime/backtests/late_entry_rows.json", encoding="utf-8"))
    late_like = {(r["day"], r["sym"]) for r in rows if r["stage"] in ("late", "none")}

    cands = []
    for d in range(g.days):
        day = (g.t0 + timedelta(days=d)).strftime("%Y-%m-%d")
        if day < X.START or day not in full:
            continue
        seg = g.RET[d * 96:(d + 1) * 96]
        for j, sym in enumerate(g.syms):
            hit = np.where(seg[:, j] >= X.TRIG)[0]
            if len(hit) == 0:
                continue
            i = d * 96 + int(hit[0])
            if i + 1 >= g.N or g.RANK[i, j] > 5:
                continue
            row = g.RET[i]
            ok = np.isfinite(row)
            breadth = float(np.mean(row[ok] > 0)) if ok.any() else 0.0
            btc = float(g.RET[i, jb]) if jb is not None and np.isfinite(g.RET[i, jb]) else 0.0
            pnl, _ = X.sim_exit(g, j, i, K, floor, P)
            cands.append({"day": day, "sym": sym, "rank": int(g.RANK[i, j]), "breadth": breadth, "btc": btc,
                          "pnl": pnl, "win": (day, sym) in winners, "late": (day, sym) in late_like})
    tr = [c for c in cands if c["day"] < TRAIN_END]
    te = [c for c in cands if c["day"] >= TRAIN_END]
    print("rank<=5 crossing candidates: train %d (%s..06-30), test %d (07-01..)" % (len(tr), X.START, len(te)))

    def pick(rows_, rk, b, bt):
        return [c for c in rows_ if c["rank"] <= rk and c["breadth"] >= b and (bt is None or c["btc"] >= bt)]

    best = None
    print("\nTRAIN (choice): rank, breadth>=, btc>=  -> n, mean pnl, precision")
    for rk in (3, 5):
        for b in BREADTH:
            for bt in BTC:
                s = pick(tr, rk, b, bt)
                if len(s) < 100:
                    continue
                m = float(np.mean([c["pnl"] for c in s]))
                print("  rank<=%d breadth>=%.1f btc>=%s  n=%4d mean %+.3f%% precision %.3f" % (
                    rk, b, bt, len(s), m, np.mean([c["win"] for c in s])))
                if best is None or m > best[0]:
                    best = (m, rk, b, bt)
    _, rk, b, bt = best
    print("\nchosen on TRAIN: rank<=%d, breadth>=%.1f, btc>=%s (train mean %+.3f%%)" % (rk, b, bt, best[0]))

    bot_te = [t for t in trades if t["day"] >= TRAIN_END and t["day"] in full]
    vals, _ = EV.leader_arm(g, bot_te, P)
    bot = [(t, v) for t, v in zip(bot_te, vals) if v is not None]
    bot_prec = float(np.mean([(t["day"], t["sym"]) in winners for t, _ in bot]))
    bot_mean = float(np.mean([v for _, v in bot]))
    s = pick(te, rk, b, bt)
    v = [c["pnl"] for c in s]
    rnd = random.Random(7)
    bs = sorted(np.mean(rnd.choices(v, k=len(v))) for _ in range(1000)) if v else [0.0] * 1000
    prec = float(np.mean([c["win"] for c in s])) if s else 0.0
    late_te = {k for k in late_like if k[0] >= TRAIN_END}
    reach = len({(c["day"], c["sym"]) for c in s if c["late"]})
    ndays = len({c["day"] for c in te})
    print("\nTEST (judged): n=%d (%.1f/day), precision %.3f vs bot %.3f; mean %+.3f%% [%+.3f, %+.3f] vs bot %+.3f%%; "
          "late/none winner-days reached at the crossing %d of %d" % (
              len(s), len(s) / max(1, ndays), prec, bot_prec, np.mean(v) if v else 0, bs[25], bs[-26], bot_mean,
              reach, len(late_te)))
    for m in sorted({c["day"][:7] for c in s}):
        mv = [c["pnl"] for c in s if c["day"][:7] == m]
        print("   %s %+.2f%% (n=%d)" % (m, np.mean(mv), len(mv)))
    ok = prec >= bot_prec and bs[25] >= bot_mean - 0.10 and reach >= 0.10 * len(late_te)
    print("\nVERDICT (third look at these data): %s" % ("SUPPORTED on the held-out months" if ok else "refuted"))


if __name__ == "__main__":
    main()
