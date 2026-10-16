"""Weakness exits -> hold on trail, narrowed by a condition chosen on Mar-Jun and judged on Jul-Oct.

WHY (2026-10-05, weak-exits-1005-spec.md follow-up 2, operator's choice)
The unconditional replay gave rocket-days +1.10 pp [+0.62, +1.63] but missed the
all-trade margin by 0.005. A narrowing condition must not be picked on the data
that judges it, so: choose on 2026-03..06, judge once on 2026-07..10.

FIXED BEFORE ANY RESULT OF EITHER HALF WAS SEEN
  candidates (12) = category set {ALL weakness, RSI divergence only}
                    x condition at the weakness bar {none, profit >= +0.5%, profit >= +1%,
                      coin in the day's top-10, day return >= +3%, day return >= +5%}
  train  trades whose day < 2026-07-01; test  trades whose day >= 2026-07-01
  rule   among candidates passing exit_validator.decide on TRAIN (rocket lower95 > 0,
         all lower95 >= -0.10 pp, >= 60% months >= 0), choose the largest rocket-day mean;
         none passes -> STOP, the test half is not opened.
  verdict  the chosen candidate on TEST with the same decide(); other candidates on test
         are printed for reference only.
Engine, trades and the hold-on-trail variant are _backtest_weak_exits.py's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config                          # noqa: E402
import exit_validator as EV            # noqa: E402
import _backtest_weak_exits as W       # noqa: E402

SPLIT_DAY = "2026-07-01"
OUT = HERE.parent / ".runtime" / "weak_exits_split.json"
CATSETS = {"ALL": None, "RSI_DIV": {"rsi_divergence"}}
CONDS = {"none": lambda f: True, "profit>=0.5": lambda f: f["profit"] >= 0.5, "profit>=1": lambda f: f["profit"] >= 1.0,
         "rank<=10": lambda f: f["rank"] <= 10, "dayret>=3": lambda f: f["dayret"] >= 0.03, "dayret>=5": lambda f: f["dayret"] >= 0.05}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    grid = EV.load_grid(int(getattr(config, "ATR_PERIOD", 14)))
    trades = W.load_trades_with_reason(grid, EV.SINCE_DEFAULT)
    rockets, winners = EV.day_labels()
    feats, pairs = {}, {}
    for i, t in enumerate(trades):
        c = W.category(t["reason"])
        if c is None:
            continue
        x, y = W.at_bar(grid, t), W.hold_on_trail(grid, t, EV.floor_of(t["mode"], config))
        if x is None or y is None or t["ix"] >= grid.N:
            continue
        pairs[i] = (x, y)
        feats[i] = {"cat": c, "profit": x, "rank": int(grid.RANK[t["ix"], t["j"]]), "dayret": float(grid.RET[t["ix"], t["j"]])}

    def rows_for(catset, cond, half):
        out = []
        for i, t in enumerate(trades):
            if (t["day"] < SPLIT_DAY) != (half == "train"):
                continue
            f = feats.get(i)
            use = f is not None and (catset is None or f["cat"] in catset) and CONDS[cond](f)
            x, y = pairs[i] if use else (0.0, 0.0)
            out.append((t, x, y))
        return out

    cands = [(cs, cd) for cs in CATSETS for cd in CONDS]
    train = {}
    for cs, cd in cands:
        train[(cs, cd)] = W.stats(rows_for(CATSETS[cs], cd, "train"), rockets, winners)
    print(f"trades: train {sum(1 for t in trades if t['day'] < SPLIT_DAY)}, test {sum(1 for t in trades if t['day'] >= SPLIT_DAY)}")
    print("\nTRAIN (2026-03..06):")
    for (cs, cd), r in train.items():
        print(f"  {cs:8s} {cd:12s} {r['verdict']:13s} changed {r['changed_trades']:4d}  rocket {r['rocket_day_diff_pp']}  all {r['all_trades_diff_pp']}  months {r['month_agreement']:.0%}/{r['months_scored']}")
    passing = [k for k, r in train.items() if r["verdict"] == "accept"]
    report = {"split_day": SPLIT_DAY, "train": {f"{k[0]}|{k[1]}": v for k, v in train.items()}, "passing_on_train": [f"{a}|{b}" for a, b in passing]}
    if not passing:
        report["verdict"] = "STOP: no candidate passes on train; the test half was not opened"
        print("\n" + report["verdict"])
    else:
        chosen = max(passing, key=lambda k: train[k]["rocket_day_diff_pp"]["mean"])
        test = W.stats(rows_for(CATSETS[chosen[0]], chosen[1], "test"), rockets, winners)
        report.update(chosen=f"{chosen[0]}|{chosen[1]}", test=test, verdict=test["verdict"])
        print(f"\nCHOSEN on train: {chosen}  (train rocket {train[chosen]['rocket_day_diff_pp']}, all {train[chosen]['all_trades_diff_pp']})")
        print(f"TEST (2026-07..10) verdict={test['verdict']} changed {test['changed_trades']}\n  rocket {test['rocket_day_diff_pp']}\n  all {test['all_trades_diff_pp']}\n"
              f"  winners {test['winner_day_diff_pp']}\n  months agree {test['month_agreement']:.0%} of {test['months_scored']}: {test['rocket_months']}")
        ref = {}
        for cs, cd in cands:
            r = W.stats(rows_for(CATSETS[cs], cd, "test"), rockets, winners)
            ref[f"{cs}|{cd}"] = r
        report["test_reference_only"] = ref
        print("\nTEST, all candidates (reference only, NOT a selection):")
        for k, r in ref.items():
            print(f"  {k:22s} {r['verdict']:13s} rocket {r['rocket_day_diff_pp']['mean']:+.3f} [{r['rocket_day_diff_pp']['lo95']:+.3f},{r['rocket_day_diff_pp']['hi95']:+.3f}]  "
                  f"all {r['all_trades_diff_pp']['mean']:+.3f} [{r['all_trades_diff_pp']['lo95']:+.3f},{r['all_trades_diff_pp']['hi95']:+.3f}]")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
