"""H2 -- fewer trades by the bot's own scores (h2-selectivity-1010-spec.md). Offline, trader app.

54 candidates = pool {real_entries, all_candidates} x score {ml_proba, candidate_score,
ranker_final_score} x train quantile {0.50, 0.75, 0.90} x max_per_day {none, 3, 10}; exits P-1 trail
for all; base = P-1 trail on every bot entry. Choose on days < split, open the test half ONCE for the
winner and the base (k = 1). Verdicts by money_validator.

  (from trader/) ../pyembed/python.exe experiment_h2.py
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
POOLS = ("real_entries", "all_candidates")
KEYS = ("ml_proba", "candidate_score", "ranker_final_score")
QS = (0.50, 0.75, 0.90)
CAPS = (None, 3, 10)
MIN_TRAIN_TRADES = 100
OUT = TC.RUNTIME / "experiments"


def candidate_grid() -> list:
    return [(p, k, q, c) for p in POOLS for k in KEYS for q in QS for c in CAPS]


def name_of(p, k, q, c) -> str:
    return f"{p}|{k}|q{q:.2f}|cap{c if c is not None else 'none'}"


def train_quantile(rows: list, key: str, q: float, split: str) -> float | None:
    """q-quantile of `key` over rows dated before `split` only (spec §7.3)."""
    v = sorted(float(r[key]) for r in rows
               if isinstance(r.get(key), (int, float)) and r["ts"].strftime("%Y-%m-%d") < split)
    if not v:
        return None
    return v[min(len(v) - 1, int(q * len(v)))]


def select_winner(train_results: dict):
    """Largest train paired mean among candidates with >= MIN_TRAIN_TRADES and paired lo95 > 0; None = STOP."""
    ok = {n: r for n, r in train_results.items()
          if r["trades"] >= MIN_TRAIN_TRADES and r["paired"].get("lo95") is not None and r["paired"]["lo95"] > 0}
    if not ok:
        return None
    return max(ok, key=lambda n: ok[n]["paired"]["mean_pct"])


def _policy(pool, key, thr, cap, cooldown):
    return replace(TP.CHAMPION, name=name_of(pool, key, 0, cap),
                   admission=TP.Admission(kind=pool, min_score=thr, score_key=key, max_per_day=cap),
                   exits=TP.Exits(kind="p1_trail"),
                   risk=TP.Risk(max_open=TP.CHAMPION.risk.max_open, one_per_symbol=True,
                                cooldown_bars=cooldown if pool == "all_candidates" else 0))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    TC.bot_readonly()
    import config as BOT                                   # read-only: COOLDOWN_BARS
    trades, blocked = R.load_events(include_blocked=True)
    grid = R._grid()
    R.attach_grid(trades, grid)
    R.attach_grid(blocked, grid)
    pools = {"real_entries": [c for c in trades if c["ie"] is not None],
             "all_candidates": sorted([c for c in trades + blocked if c["ie"] is not None], key=lambda c: c["ts"])}
    print(f"pools: real_entries {len(pools['real_entries'])}, all_candidates {len(pools['all_candidates'])}; "
          f"load {time.time() - t0:.0f}s", flush=True)
    fm, fl, cache = FM.model("taker", 1), R._floor(), {}
    cooldown = int(getattr(BOT, "COOLDOWN_BARS", 19))
    base = replace(TP.CHAMPION, name="base_p1_real", exits=TP.Exits(kind="p1_trail"))

    def go(pol, rows, lo, hi):
        return R.run(pol, fm, grid, rows, floor_of=fl, day_from=lo, day_to=hi, plan_cache=cache)

    base_tr = go(base, pools["real_entries"], None, SPLIT)
    print(f"base train: {base_tr['agg']['mean_pct']:+.3f}%/d, trades {len(base_tr['trades'])}", flush=True)
    train, thresholds = {}, {}
    for (pool, key, q, cap) in candidate_grid():
        thr = train_quantile(pools[pool], key, q, SPLIT)
        n = name_of(pool, key, q, cap)
        thresholds[n] = thr
        if thr is None:
            continue
        r = go(_policy(pool, key, thr, cap, cooldown), pools[pool], None, SPLIT)
        pr = DR.paired(r["series"], base_tr["series"])
        train[n] = {"threshold": thr, "trades": len(r["trades"]), "agg": r["agg"], "paired": pr,
                    "net_per_trade": R.decompose(r["trades"]).get("net_mean_pct")}
        print(f"  {n:46s} thr {thr:8.3f} trades {len(r['trades']):5d} R {r['agg'].get('mean_pct', 0):+.3f} "
              f"paired {pr.get('mean_pct', 0):+.3f} [{pr.get('lo95', 0):+.3f},{pr.get('hi95', 0):+.3f}]", flush=True)
    winner = select_winner(train)
    report = {"spec": "h2-selectivity-1010-spec.md", "split": SPLIT, "costs": "taker + 8 bps/side",
              "base_train": base_tr["agg"], "train": train, "winner": winner}
    if winner is None:
        report["verdict"] = "STOP: no candidate passes on train; the test half was not opened"
        print("\n" + report["verdict"])
    else:
        pool, key, qs, caps = winner.split("|")
        cap = None if caps == "capnone" else int(caps[3:])
        pol = _policy(pool, key, train[winner]["threshold"], cap, cooldown)
        print(f"\nWINNER on train: {winner}  (paired {train[winner]['paired']})", flush=True)
        w_te = go(pol, pools[pool], SPLIT, None)
        b_te = go(base, pools["real_entries"], SPLIT, None)
        ch_te = R.run(TP.CHAMPION, fm, grid, trades, floor_of=fl, day_from=SPLIT)
        pr = DR.paired(w_te["series"], b_te["series"])
        diff = [(d, v) for d, v in dict(w_te["series"]).items()]
        bmap = dict(b_te["series"])
        months = DR.month_means([(d, v - bmap[d]) for d, v in diff if d in bmap])
        imp = MV.improves(pr, w_te["agg"].get("max_dd_pct", 0), b_te["agg"].get("max_dd_pct", 0), months, len(w_te["trades"]))
        tci = MV.trade_ci([t["ret_pct"] for t in w_te["trades"]])
        prof = MV.profitable(w_te["agg"], tci)
        ref = {}
        for lvl in (0, 2):
            rr = R.run(pol, FM.model("taker", lvl), grid, pools[pool], floor_of=fl, day_from=SPLIT, plan_cache=cache)
            ref[f"slip{lvl}"] = rr["agg"]
        report.update(test={"winner": w_te["agg"], "base": b_te["agg"], "champion": ch_te["agg"], "paired_vs_base": pr,
                            "paired_vs_champion": DR.paired(w_te["series"], ch_te["series"]),
                            "months_paired": months, "trades": len(w_te["trades"]), "trade_ci": tci,
                            "decomposition": R.decompose(w_te["trades"]), "improves": imp, "profitable": prof,
                            "slip_reference": ref})
        MV.log_trial({"family": "H2", "name": winner, "policy_epoch": pol.epoch(), "policy": asdict(pol),
                      "k_opened_on_test": 1, "improves": imp["verdict"], "profitable": prof["verdict"],
                      "paired_test": pr})
        print(f"TEST winner: R {w_te['agg']['mean_pct']:+.3f}%/d [{w_te['agg']['lo95']:+.3f},{w_te['agg']['hi95']:+.3f}] "
              f"cmp {w_te['agg']['compound_pct']:+.1f}% DD {w_te['agg']['max_dd_pct']:.1f}% trades {len(w_te['trades'])} "
              f"net/trade {tci['mean']:+.3f} [{tci['lo95']:+.3f},{tci['hi95']:+.3f}]")
        print(f"TEST base:   R {b_te['agg']['mean_pct']:+.3f}%/d  DD {b_te['agg']['max_dd_pct']:.1f}%   champion R {ch_te['agg']['mean_pct']:+.3f}%/d")
        print(f"paired vs base {pr['mean_pct']:+.3f} [{pr['lo95']:+.3f},{pr['hi95']:+.3f}]  months {months}")
        print(f"improves: {imp}\nprofitable: {prof}\nslip 0 / 2: {ref['slip0'].get('mean_pct')} / {ref['slip2'].get('mean_pct')} %/d")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "h2_selectivity_1010.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved {p}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
