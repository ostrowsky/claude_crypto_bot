"""E8 -- derivatives positioning as an entry filter, judged in money (e8-positioning-money-1012-spec.md).

Bot entries since 2026-08-21, each joined to the LAST positioning snapshot of its coin at or before the
entry and at most 2.5 h old; exits X4 (P-1 trail, 8% floor) for every arm; base = X4 on all entries
with positioning data. 13 pre-registered rules, thresholds = train quantiles. Choose on train
(R, >= 60 trades, R above the 95th percentile of 50 random thinnings of the same intensity);
open the test half ONCE. Main verdict beats_random; improves and profitable beside it.

  (from trader/) ../pyembed/python.exe experiment_e8.py --train-only    # selection only, test not opened
  (from trader/) ../pyembed/python.exe experiment_e8.py                 # the 2026-10-18 run
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
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

SINCE = "2026-08-21"
SPLIT = "2026-09-26"
MAX_AGE = timedelta(hours=2.5)
MIN_TRAIN_TRADES = 60
N_RANDOM = 50
OUT = TC.RUNTIME / "experiments"
X4 = replace(TP.CHAMPION, name="x4_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))
RULES = ([{"name": f"{f}_above_q{q}", "feature": f, "direction": "above", "q": q}
          for f in ("oi_4h", "taker", "taker_trend", "top") for q in (0.50, 0.75)]
         + [{"name": f"{f}_below_q{q}", "feature": f, "direction": "below", "q": q}
            for f in ("retail", "funding_bp") for q in (0.50, 0.25)]
         + [{"name": "flow_longs_opening", "feature": "flow", "direction": "eq", "q": None}])


def _dt(ts):
    if isinstance(ts, datetime):
        return ts
    d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _day(c):
    return c["ts"].strftime("%Y-%m-%d")


# ---------------------------------------------------------------- positioning

def load_positioning(path: Path) -> dict:
    """{sym: [(ts, row), ...] sorted}. Broken lines and a partial tail are skipped (the file is rewritten)."""
    out = {}
    with open(path, "rb") as fh:
        data = fh.read()
    for raw in data.split(b"\n"):
        if not raw.strip():
            continue
        try:
            r = json.loads(raw.decode("utf-8", "replace"))
            t = _dt(r["ts"])
            s = r["sym"]
        except Exception:
            continue
        out.setdefault(s, []).append((t, r))
    for s in out:
        out[s].sort(key=lambda x: x[0])
    return out


def asof(index: dict, sym: str, ts, max_age: timedelta = MAX_AGE):
    """Last snapshot of `sym` at or before `ts`, at most `max_age` old; None otherwise (no look-ahead)."""
    rows = index.get(sym)
    if not rows:
        return None
    times = [t for t, _ in rows]
    i = bisect.bisect_right(times, ts) - 1
    if i < 0 or ts - times[i] > max_age:
        return None
    return rows[i][1]


def attach_positioning(cands: list, index: dict):
    kept, missing = [], 0
    for c in cands:
        row = asof(index, c["sym"], c["ts"])
        if row is None:
            missing += 1
            continue
        c["pos"] = row
        kept.append(c)
    return kept, missing


# ---------------------------------------------------------------- rules

def train_threshold(rows: list, feature: str, q: float, split: str):
    v = sorted(float(r["pos"][feature]) for r in rows
               if r["ts"].strftime("%Y-%m-%d") < split and isinstance(r["pos"].get(feature), (int, float)))
    if not v:
        return None
    return v[min(len(v) - 1, int(q * len(v)))]


def passes(pos: dict, rule, thr) -> bool:
    feature, direction = rule
    v = pos.get(feature)
    if direction == "eq":
        return v == thr
    if not isinstance(v, (int, float)) or thr is None:
        return False
    return v >= thr if direction == "above" else v <= thr


# ---------------------------------------------------------------- AUC reference (spec §5, added 2026-10-13)

def trade_label(grid, c, fm, floor_of=None):
    """Net % of the X4 trade the E1 engine makes when it admits `c` (same formula as the engine)."""
    plan = TP.exit_plan(X4, grid, c, floor_of)
    if plan is None or not c.get("price"):
        return None
    _, exit_raw, reason, _ = plan
    fill = fm.buy(c["price"], 0.0, None)
    px = exit_raw if reason == "tp" else fm.sell(exit_raw, 0.0, None)
    fee_exit = fm.maker_fee if reason == "tp" else fm.taker_fee
    return (px * (1 - fee_exit) * (1 - fm.taker_fee) / fill - 1) * 100


def feature_auc(rows: list, feature: str, direction: str) -> dict:
    """AUC of the rule's feature (oriented by its direction) against y > 0; diagnostic only."""
    import experiment_e2 as E2
    xs, ys = [], []
    for r in rows:
        v = r["pos"].get(feature)
        if r.get("y") is None:
            continue
        if direction == "eq":
            x = 1.0 if v == "longs_opening" else 0.0
        elif isinstance(v, (int, float)):
            x = float(v) if direction == "above" else -float(v)
        else:
            continue
        xs.append(x)
        ys.append(1.0 if r["y"] > 0 else 0.0)
    return {"n": len(xs), "base_rate": round(sum(ys) / len(ys), 4) if ys else None,
            "auc": E2.auc(ys, xs) if xs else None}


def auc_report(rows: list, split: str, winner=None) -> dict:
    """Train: every rule. Test: the winner only (one hypothesis opened on test)."""
    tr = [r for r in rows if r["ts"].strftime("%Y-%m-%d") < split]
    te = [r for r in rows if r["ts"].strftime("%Y-%m-%d") >= split]
    out = {"train": {r["name"]: feature_auc(tr, r["feature"], r["direction"]) for r in RULES}, "test": {}}
    if winner is not None:
        w = next(r for r in RULES if r["name"] == winner)
        out["test"][winner] = feature_auc(te, w["feature"], w["direction"])
    return out


def random_percentile(value: float, control: list) -> float:
    return sum(1 for c in control if c < value) / len(control) if control else 0.0


def beats_random(value: float, control: list) -> bool:
    return random_percentile(value, control) >= 0.95


def random_policies(frac: float) -> list:
    return [replace(X4, name=f"random{s}", admission=TP.Admission(kind="real_entries", random_frac=frac, random_seed=s))
            for s in range(N_RANDOM)]


def select_winner(train_results: dict):
    ok = {n: r for n, r in train_results.items()
          if r["trades"] >= MIN_TRAIN_TRADES and r["beats_random_train"] and r.get("R") is not None}
    if not ok:
        return None
    return max(ok, key=lambda n: ok[n]["R"])


def pipeline(validate, select, test):
    res = validate()
    w = select({k: v for k, v in res.items() if not k.startswith("_")})
    return {"train": res, "winner": w, "test": test(w) if w is not None else None}


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true", help="selection only; the test half is not opened")
    a = ap.parse_args(argv)
    t0 = time.time()
    idx = load_positioning(TC.POSITIONING)
    trades, _ = R.load_events(since=SINCE)
    grid = R._grid()
    R.attach_grid(trades, grid)
    pool = [c for c in trades if c["ie"] is not None]
    kept, missing = attach_positioning(pool, idx)
    fm, fl, cache = FM.model("taker", 1), R._floor(), {}
    last_full = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    for c in kept:
        c["y"] = trade_label(grid, c, fm, fl)
    train = [c for c in kept if _day(c) < SPLIT]
    test = [c for c in kept if SPLIT <= _day(c) <= last_full]
    print(f"entries {len(pool)}, with positioning {len(kept)} (missing {missing}, {missing / max(1, len(pool)):.0%}); "
          f"train {len(train)}, test {len(test)} [{time.time() - t0:.0f}s]", flush=True)
    report = {"spec": "e8-positioning-money-1012-spec.md", "split": SPLIT, "entries": len(pool), "with_positioning": len(kept),
              "missing_positioning": missing, "train_rows": len(train), "test_rows": len(test)}

    def go(pol, rows):
        return R.run(pol, fm, grid, rows, floor_of=fl, plan_cache=cache)

    def mark(rule, thr, rows):
        key = "e8_" + rule["name"]
        n = 0
        for c in rows:
            ok = passes(c["pos"], (rule["feature"], rule["direction"]), thr)
            c[key] = 1.0 if ok else 0.0
            n += ok
        return key, n / max(1, len(rows))

    def rule_policy(key):
        return replace(X4, name=key, admission=TP.Admission(kind="real_entries", min_score=0.5, score_key=key))

    base_tr = go(X4, train)
    report["base_train"] = base_tr["agg"]
    print(f"base X4 train: R {base_tr['agg'].get('mean_pct')} trades {len(base_tr['trades'])}", flush=True)

    def validate():
        res = {}
        for rule in RULES:
            thr = "longs_opening" if rule["direction"] == "eq" else train_threshold(kept, rule["feature"], rule["q"], SPLIT)
            key, frac = mark(rule, thr, train)
            r = go(rule_policy(key), train)
            ctrl = [go(p, train)["agg"].get("mean_pct") for p in random_policies(frac)]
            ctrl = [v for v in ctrl if v is not None]
            Rv = r["agg"].get("mean_pct")
            res[rule["name"]] = {"threshold": thr, "pass_rate": frac, "trades": len(r["trades"]), "R": Rv, "agg": r["agg"],
                                 "net_per_trade": R.decompose(r["trades"]).get("net_mean_pct"),
                                 "random_R_median": sorted(ctrl)[len(ctrl) // 2] if ctrl else None,
                                 "random_percentile": random_percentile(Rv, ctrl) if Rv is not None else None,
                                 "beats_random_train": bool(Rv is not None and beats_random(Rv, ctrl))}
            x = res[rule["name"]]
            print(f"  {rule['name']:26s} pass {frac:.2f} trades {x['trades']:4d} R {Rv if Rv is not None else float('nan'):+.3f} "
                  f"random med {x['random_R_median'] if x['random_R_median'] is not None else float('nan'):+.3f} pct {x['random_percentile']}",
                  flush=True)
        return res

    def test_fn(winner):
        rule = next(r for r in RULES if r["name"] == winner)
        thr = report["train"][winner]["threshold"]
        key, frac = mark(rule, thr, test)
        w, b = go(rule_policy(key), test), go(X4, test)
        ctrl = [go(p, test)["agg"].get("mean_pct") for p in random_policies(frac)]
        ctrl = [v for v in ctrl if v is not None]
        Rv = w["agg"].get("mean_pct")
        tci = MV.trade_ci([t["ret_pct"] for t in w["trades"]])
        pr = DR.paired(w["series"], b["series"])
        out = {"rule": rule, "threshold": thr, "pass_rate_test": frac, "winner": w["agg"], "base": b["agg"],
               "trades": len(w["trades"]), "trade_ci": tci, "paired_vs_base": pr,
               "random_R_median": sorted(ctrl)[len(ctrl) // 2] if ctrl else None, "random_percentile": random_percentile(Rv, ctrl),
               "beats_random": beats_random(Rv, ctrl),
               "improves": bool(pr.get("lo95") is not None and pr["lo95"] > 0 and beats_random(Rv, ctrl)),
               "profitable": MV.profitable(w["agg"], tci)["verdict"],
               "slip_reference": {f"slip{lv}": R.run(rule_policy(key), FM.model("taker", lv), grid, test, floor_of=fl)["agg"]
                                  for lv in (0, 2)}}
        MV.log_trial({"family": "E8", "name": winner, "k_opened_on_test": 1, "beats_random": out["beats_random"],
                      "improves": out["improves"], "profitable": out["profitable"], "test_R": w["agg"]})
        return out

    report["train"] = {}
    if a.train_only:
        report["train"] = validate()
        w = select_winner(report["train"])
        report.update(winner=w, test=None, note="train-only run: the test half was NOT opened",
                      auc_reference=auc_report(train, SPLIT, winner=None))
        print(f"\nTRAIN-ONLY: winner {w} (test not opened)")
    else:
        def val():
            report["train"] = validate()
            return report["train"]
        res = pipeline(val, select_winner, test_fn)
        report.update(winner=res["winner"], test=res["test"],
                      auc_reference=auc_report(train + test, SPLIT, winner=res["winner"]))
        if res["winner"] is None:
            report["verdict"] = "STOP: no rule beats random on train; the test half was not opened"
            print("\n" + report["verdict"])
        else:
            t = res["test"]
            print(f"\nWINNER {res['winner']}: test R {t['winner'].get('mean_pct')} [{t['winner'].get('lo95')},{t['winner'].get('hi95')}] "
                  f"trades {t['trades']} net/trade {t['trade_ci']} | random median {t['random_R_median']} pct {t['random_percentile']} "
                  f"beats_random {t['beats_random']} improves {t['improves']} profitable {t['profitable']}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / ("e8_positioning_train_only.json" if a.train_only else "e8_positioning_1018.json")
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
