"""H3 reference control (h3-exits-money-1011-spec.md §7): is X4 better than simply trading less?
The bot's entries thinned at random to X4's trade count on the test half, with the bot's own exits
(X0), 20 seeds. Added after the H3 verdict; describes, does not change it.

  (from trader/) ../pyembed/python.exe experiment_h3_control.py
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
import experiment_h3 as H         # noqa: E402
import fill_model as FM           # noqa: E402
import money_validator as MV      # noqa: E402
import trading_policy as TP       # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    rep = json.loads((TC.RUNTIME / "experiments" / "h3_exits_money_1011.json").read_text(encoding="utf-8"))
    target = int(rep["test"]["trades"])
    win_R = rep["test"]["winner"]["mean_pct"]
    win_net = rep["test"]["trade_ci"]["mean"]
    trades, _ = R.load_events()
    grid = R._grid()
    R.attach_grid(trades, grid)
    fm, fl, cache = FM.model("taker", 1), R._floor(), {}

    def run(frac, seed):
        pol = replace(TP.CHAMPION, name=f"thin{seed}",
                      admission=TP.Admission(kind="real_entries", random_frac=frac, random_seed=seed))
        return R.run(pol, fm, grid, trades, floor_of=fl, day_from=H.SPLIT, plan_cache=cache)

    frac = H.match_frac(lambda f: len(run(f, 0)["trades"]), target)
    print(f"target trades {target}; matched frac {frac:.4f}  [{time.time() - t0:.0f}s]", flush=True)
    ctrl = []
    for seed in range(20):
        r = run(frac, seed)
        ctrl.append({"seed": seed, "R": r["agg"]["mean_pct"], "trades": len(r["trades"]),
                     "net_per_trade": MV.trade_ci([t["ret_pct"] for t in r["trades"]])["mean"]})
    rs = sorted(c["R"] for c in ctrl)
    ns = sorted(c["net_per_trade"] for c in ctrl)
    out = {"target_trades": target, "frac": frac, "x4_R": win_R, "x4_net_per_trade": win_net, "control": ctrl,
           "R_median": rs[len(rs) // 2], "R_range": [rs[0], rs[-1]], "net_median": ns[len(ns) // 2],
           "x4_R_percentile": sum(1 for v in rs if v < win_R) / len(rs),
           "x4_net_percentile": sum(1 for v in ns if v < win_net) / len(ns)}
    print(f"X4 test R {win_R:+.3f}%/d net/trade {win_net:+.3f} | thinned X0 x20 (trades ~{sorted(c['trades'] for c in ctrl)[10]}): "
          f"R median {out['R_median']:+.3f} [{rs[0]:+.3f}..{rs[-1]:+.3f}], net/trade median {out['net_median']:+.3f} | "
          f"X4 beats {out['x4_R_percentile']:.0%} (R), {out['x4_net_percentile']:.0%} (net/trade)")
    p = TC.RUNTIME / "experiments" / "h3_thinning_control_1011.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
