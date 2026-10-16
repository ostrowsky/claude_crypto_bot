"""H2 reference control (h2-selectivity-1010-spec.md §8): does the winner's score beat a RANDOM
selection of the same intensity? Added after the H2 verdict; describes, does not change it.

  (from trader/) ../pyembed/python.exe experiment_h2_control.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import execution_replay as R      # noqa: E402
import experiment_h2 as H         # noqa: E402
import fill_model as FM           # noqa: E402
import money_validator as MV      # noqa: E402
import trading_policy as TP       # noqa: E402

SEEDS = range(20)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    rep = json.loads((TC.RUNTIME / "experiments" / "h2_selectivity_1010.json").read_text(encoding="utf-8"))
    winner = rep["winner"]
    pool, key, _, caps = winner.split("|")
    cap = None if caps == "capnone" else int(caps[3:])
    thr = rep["train"][winner]["threshold"]
    TC.bot_readonly()
    import config as BOT
    trades, blocked = R.load_events(include_blocked=True)
    grid = R._grid()
    R.attach_grid(trades, grid)
    R.attach_grid(blocked, grid)
    rows = [c for c in trades if c["ie"] is not None] if pool == "real_entries" else \
        sorted([c for c in trades + blocked if c["ie"] is not None], key=lambda c: c["ts"])
    train_rows = [c for c in rows if c["ts"].strftime("%Y-%m-%d") < H.SPLIT]
    frac = sum(1 for c in train_rows if isinstance(c.get(key), (int, float)) and c[key] >= thr) / len(train_rows)
    print(f"winner {winner}: threshold {thr}, admitted share of the pool on train {frac:.3f}", flush=True)
    fm, fl, cache = FM.model("taker", 1), R._floor(), {}
    cooldown = int(getattr(BOT, "COOLDOWN_BARS", 19))
    win_pol = H._policy(pool, key, thr, cap, cooldown)
    out = {"winner": winner, "frac": frac, "halves": {}}
    for half, lo, hi in (("train", None, H.SPLIT), ("test", H.SPLIT, None)):
        w = R.run(win_pol, fm, grid, rows, floor_of=fl, day_from=lo, day_to=hi, plan_cache=cache)
        wr, wt = w["agg"]["mean_pct"], MV.trade_ci([t["ret_pct"] for t in w["trades"]])["mean"]
        ctrl = []
        for seed in SEEDS:
            pol = replace(win_pol, name=f"random{seed}",
                          admission=TP.Admission(kind=pool, max_per_day=cap, random_frac=frac, random_seed=seed))
            r = R.run(pol, fm, grid, rows, floor_of=fl, day_from=lo, day_to=hi, plan_cache=cache)
            ctrl.append({"seed": seed, "R": r["agg"]["mean_pct"], "trades": len(r["trades"]),
                         "net_per_trade": MV.trade_ci([t["ret_pct"] for t in r["trades"]])["mean"]})
        rs = sorted(c["R"] for c in ctrl)
        ts = sorted(c["net_per_trade"] for c in ctrl)
        pct_r = sum(1 for v in rs if v < wr) / len(rs)
        pct_t = sum(1 for v in ts if v < wt) / len(ts)
        out["halves"][half] = {"winner_R": wr, "winner_net_per_trade": wt, "winner_trades": len(w["trades"]),
                               "control": ctrl, "control_R_median": rs[len(rs) // 2], "control_R_range": [rs[0], rs[-1]],
                               "control_net_median": ts[len(ts) // 2], "winner_R_percentile": pct_r,
                               "winner_net_percentile": pct_t}
        print(f"[{half}] winner R {wr:+.3f}%/d net/trade {wt:+.3f} trades {len(w['trades'])} | random x{len(ctrl)}: "
              f"R median {rs[len(rs) // 2]:+.3f} [{rs[0]:+.3f}..{rs[-1]:+.3f}], net/trade median {ts[len(ts) // 2]:+.3f}, "
              f"trades ~{sorted(c['trades'] for c in ctrl)[len(ctrl) // 2]} | winner beats {pct_r:.0%} (R), {pct_t:.0%} (net/trade)",
              flush=True)
    p = TC.RUNTIME / "experiments" / "h2_random_control_1010.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
