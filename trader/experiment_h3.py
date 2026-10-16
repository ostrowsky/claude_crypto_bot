"""H3 -- exits in money: the bot's own entries, other exit rules (h3-exits-money-1011-spec.md).

X0 real (champion, base) | X1 rsi_hold | X2 leader | X3 leader_rsi_hold | X4 p1_floor8.
Choose on days < split by the paired daily difference vs X0 (lo95 > 0, >= 100 changed exits);
open the test half ONCE for the winner (k = 1). Verdicts by money_validator.

  (from trader/) ../pyembed/python.exe experiment_h3.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import daily_revenue as DR        # noqa: E402
import execution_replay as R      # noqa: E402
import fill_model as FM           # noqa: E402
import money_validator as MV      # noqa: E402
import trading_policy as TP       # noqa: E402

SPLIT = "2026-07-01"
MIN_CHANGED = 100
OUT = TC.RUNTIME / "experiments"


def select_winner(train_results: dict):
    """Largest train paired mean vs X0 among candidates with paired lo95 > 0 and >= MIN_CHANGED changed exits."""
    ok = {n: r for n, r in train_results.items()
          if r["changed"] >= MIN_CHANGED and r["paired"].get("lo95") is not None and r["paired"]["lo95"] > 0}
    if not ok:
        return None
    return max(ok, key=lambda n: ok[n]["paired"]["mean_pct"])


def split_train_results(res: dict) -> dict:
    """Candidates vs auxiliary entries ('_'-prefixed, e.g. the base's numbers): only candidates are selectable."""
    return {"candidates": {k: v for k, v in res.items() if not k.startswith("_")},
            "aux": {k: v for k, v in res.items() if k.startswith("_")}}


def pipeline(validate, select, test):
    res = validate()
    w = select(split_train_results(res)["candidates"])
    return {"train": res, "winner": w, "test": test(w) if w is not None else None}


def match_frac(run_count, target: int, tol: float = 0.05, iters: int = 20) -> float:
    """Bisection on frac in (0, 1] so that run_count(frac) is within tol of target (h3 spec §7)."""
    lo, hi = 0.0, 1.0
    f = 0.5
    for _ in range(iters):
        f = (lo + hi) / 2
        n = run_count(f)
        if abs(n - target) <= tol * target:
            return f
        if n < target:
            lo = f
        else:
            hi = f
    return f


def candidates(leader_params: dict) -> dict:
    lp = tuple(sorted(leader_params.items()))
    base = TP.CHAMPION
    return {"X1_rsi_hold": replace(base, name="X1_rsi_hold", exits=TP.Exits(kind="rsi_hold")),
            "X2_leader": replace(base, name="X2_leader", exits=TP.Exits(kind="leader", leader=lp)),
            "X3_leader_rsi_hold": replace(base, name="X3_leader_rsi_hold", exits=TP.Exits(kind="leader_rsi_hold", leader=lp)),
            "X4_p1_floor8": replace(base, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))}


def _changed(trades) -> int:
    return sum(1 for t in trades if t["reason"] != "real")


def _rocket_mean(trades, rockets) -> dict:
    v = [t["ret_pct"] for t in trades if (t["entry_ts"].strftime("%Y-%m-%d"), t["sym"]) in rockets]
    return {"n": len(v), "mean_net_pct": round(sum(v) / len(v), 4) if v else None}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    TC.bot_readonly()
    import config as BOT
    import exit_validator as EV
    P = EV.leader_params(BOT)
    trades, _ = R.load_events()
    grid = R._grid()
    R.attach_grid(trades, grid)
    fm, fl, cache = FM.model("taker", 1), R._floor(), {}
    cands = candidates(P)
    try:
        rockets, _ = EV.day_labels()
    except Exception:
        rockets = set()
    print(f"trades {len(trades)}, leader params {P}  [{time.time() - t0:.0f}s]", flush=True)

    def go(pol, lo, hi, f=fm):
        return R.run(pol, f, grid, trades, floor_of=fl, day_from=lo, day_to=hi, plan_cache=cache)

    base_tr = go(TP.CHAMPION, None, SPLIT)
    print(f"X0 train: R {base_tr['agg']['mean_pct']:+.3f}%/d trades {len(base_tr['trades'])}", flush=True)
    report = {"spec": "h3-exits-money-1011-spec.md", "split": SPLIT, "leader_params": P, "base_train": base_tr["agg"]}

    def validate():
        res = {}
        for name, pol in cands.items():
            r = go(pol, None, SPLIT)
            pr = DR.paired(r["series"], base_tr["series"])
            res[name] = {"paired": pr, "changed": _changed(r["trades"]), "trades": len(r["trades"]), "agg": r["agg"],
                         "net_per_trade": R.decompose(r["trades"]).get("net_mean_pct"),
                         "rocket_trades": _rocket_mean(r["trades"], rockets)}
            print(f"  {name:20s} trades {len(r['trades']):5d} changed {res[name]['changed']:5d} R {r['agg']['mean_pct']:+.3f}%/d "
                  f"paired {pr['mean_pct']:+.3f} [{pr['lo95']:+.3f},{pr['hi95']:+.3f}] net/trade {res[name]['net_per_trade']:+.3f}",
                  flush=True)
        res["_X0"] = {"rocket_trades": _rocket_mean(base_tr["trades"], rockets),
                      "net_per_trade": R.decompose(base_tr["trades"]).get("net_mean_pct")}
        return res

    def test(winner):
        pol = cands[winner]
        w, b = go(pol, SPLIT, None), go(TP.CHAMPION, SPLIT, None)
        pr = DR.paired(w["series"], b["series"])
        bmap = dict(b["series"])
        months = DR.month_means([(d, v - bmap[d]) for d, v in w["series"] if d in bmap])
        imp = MV.improves(pr, w["agg"].get("max_dd_pct", 0), b["agg"].get("max_dd_pct", 0), months, len(w["trades"]))
        tci = MV.trade_ci([t["ret_pct"] for t in w["trades"]])
        prof = MV.profitable(w["agg"], tci)
        ref = {f"slip{lvl}": go(pol, SPLIT, None, FM.model("taker", lvl))["agg"] for lvl in (0, 2)}
        out = {"policy": asdict(pol), "epoch": pol.epoch(), "winner": w["agg"], "base": b["agg"], "paired_vs_base": pr,
               "months_paired": months, "trades": len(w["trades"]), "changed": _changed(w["trades"]), "trade_ci": tci,
               "decomposition": R.decompose(w["trades"]), "base_decomposition": R.decompose(b["trades"]),
               "rocket_trades": {"winner": _rocket_mean(w["trades"], rockets), "base": _rocket_mean(b["trades"], rockets)},
               "improves": imp, "profitable": prof, "slip_reference": ref}
        MV.log_trial({"family": "H3", "name": winner, "policy_epoch": pol.epoch(), "k_opened_on_test": 1,
                      "improves": imp["verdict"], "profitable": prof["verdict"], "paired_test": pr})
        return out

    res = pipeline(validate, select_winner, test)
    train = {k: v for k, v in res["train"].items() if not k.startswith("_")}
    report.update(train=res["train"], winner=res["winner"], test=res["test"])
    if res["winner"] is None:
        report["verdict"] = "STOP: no exit rule passes on train; the test half was not opened"
        print("\n" + report["verdict"])
    else:
        t = res["test"]
        print(f"\nWINNER on train: {res['winner']}  paired {train[res['winner']]['paired']}")
        print(f"TEST winner R {t['winner']['mean_pct']:+.3f}%/d [{t['winner']['lo95']:+.3f},{t['winner']['hi95']:+.3f}] "
              f"DD {t['winner']['max_dd_pct']:.1f}% trades {t['trades']} changed {t['changed']} "
              f"net/trade {t['trade_ci']['mean']:+.3f} [{t['trade_ci']['lo95']:+.3f},{t['trade_ci']['hi95']:+.3f}]")
        print(f"TEST X0     R {t['base']['mean_pct']:+.3f}%/d DD {t['base']['max_dd_pct']:.1f}%")
        print(f"paired vs X0 {t['paired_vs_base']['mean_pct']:+.3f} [{t['paired_vs_base']['lo95']:+.3f},{t['paired_vs_base']['hi95']:+.3f}] months {t['months_paired']}")
        print(f"rocket-day trades: {t['rocket_trades']}")
        print(f"improves {t['improves']}\nprofitable {t['profitable']}\nslip0/2 {t['slip_reference']['slip0'].get('mean_pct')} / {t['slip_reference']['slip2'].get('mean_pct')}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "h3_exits_money_1011.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
