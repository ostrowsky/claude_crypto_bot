"""QB -- data, train and one-time test of the quarterly basis carry (quarterly-basis-qb-1029-spec.md §2, §5, §6).

  fetch   1h klines of every BTC/ETH USD-M quarterly contract 2023Q1..2026Q4 from the public fapi.binance.com
          -> runtime/quarterly/<SYMBOL>_1h.npz (append-only)
  train   4 candidates QB_T{0,5,10,15} on decisions 2023-03-01 .. 2025-12-31, entries only into contracts expiring
          before 2026; vs QB_T0 and (reference) vs FC_PASSIVE on the same days; PBO -> runtime/experiments/QB_TRAIN.json
  test    opened ONCE by the operator (marker QB_TEST_OPENED.json, k = 1)

  (from trader/) ../pyembed/python.exe -u experiment_qb.py fetch
  (from trader/) ../pyembed/python.exe -u experiment_qb.py train
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import quarterly as Q             # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN_FROM, TEST_FROM = "2023-03-01", "2026-01-01"
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
DD_FLOOR_PCT = -10.0
START_EQUITY = 1_000_000.0
DATA = TC.RUNTIME / "quarterly"
OUT = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT / "QB_TRAIN.json"
TEST_MARKER = OUT / "QB_TEST_OPENED.json"
SPEC = "quarterly-basis-qb-1029-spec.md"


def candidates() -> dict:
    return {f"QB_T{t}": t / 100 for t in (0, 5, 10, 15)}


def admit(res: dict):
    agg = res.get("agg") or {}
    years = res.get("r_by_year") or {}
    checks = {"R_lo_gt_0": agg.get("lo95") is not None and agg["lo95"] > 0,
              "no_liquidation": res.get("liquidations", 1) == 0,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE}
    if (res.get("theta") or 0) > 0:
        checks["beats_T0"] = (res.get("paired_t0") or {}).get("mean_pct") is not None and res["paired_t0"]["mean_pct"] > 0
    return all(checks.values()), checks


def select(res: dict, pbo):
    if pbo is None or pbo >= PBO_MAX:
        return None
    adm = [n for n, v in res.items() if v.get("admitted")]
    return max(adm, key=lambda n: (res[n]["agg"]["mean_pct"], n)) if adm else None


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("QB: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


# ------------------------------------------------------------------ data
def fetch(year_from: int = 2023, year_to: int = 2026) -> dict:
    import exchange_client as X
    import history_store as HS
    c = X.ExchangeClient(key="", secret="")
    DATA.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    res = {}
    for e in Q.expiries(year_from, year_to):
        for pair in Q.PAIRS:
            sym = Q.symbol_of(pair, e)
            p = DATA / f"{sym}_1h.npz"
            old = HS.load(p)
            start = int(old["t"][-1]) + Q.H if old is not None and len(old["t"]) else e - 200 * Q.D
            end = min(e, now_ms)
            rows, cur, n_req = [], start, 0
            while cur < end:
                time.sleep(0.25)
                try:
                    page = c.futures_klines_from(sym, "1h", cur, limit=1500)
                except Exception as ex:
                    res[sym] = {"error": repr(ex)[:200]}
                    break
                n_req += 1
                if not page:
                    break
                rows.extend(r for r in page if int(r[0]) + Q.H <= now_ms)
                nxt = int(page[-1][0]) + Q.H
                if nxt <= cur:
                    break
                cur = nxt
            if rows:
                new = {"t": np.array([int(r[0]) for r in rows], dtype=np.int64)}
                for k, col in (("o", 1), ("h", 2), ("l", 3), ("c", 4), ("v", 5)):
                    new[k] = np.array([float(r[col]) for r in rows])
                merged, added = HS.merge(old, new)
                HS.save(p, merged)
                res[sym] = {"added": added, "bars": len(merged["t"]), "requests": n_req}
            else:
                res.setdefault(sym, {"added": 0, "requests": n_req})
            print(sym, json.dumps(res[sym]), flush=True)
    return res


def load() -> tuple:
    import history_store as HS
    fut = {}
    for p in sorted(DATA.glob("*_1h.npz")):
        a = HS.load(p)
        if a is not None and len(a["t"]):
            fut[p.name[:-len("_1h.npz")]] = a
    spot = {}
    for pair in Q.PAIRS:
        a = HS.load(HS.path_of(pair))
        h = HS.agg_1h(a, complete_only=True)
        spot[pair] = {k: h[k] for k in ("t", "o", "h", "l", "c", "v")}
    return spot, fut


def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def _fc_passive_series(day_from, day_to):
    import carry as C
    import experiment_fc as EF
    import experiment_fs as FS
    m = FS.train_market(FS.load_market(), test_from=day_to)
    return C.simulate(m, EF.passive_config(), C.fees(1), day_from, day_to)


def run_train() -> dict:
    import carry as C
    import daily_revenue as DR
    import experiment_xsmom as XE
    import pbo as PB
    spot, fut = load()
    fees = C.fees(1)
    max_exp = Q._ms(datetime.strptime(TEST_FROM, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    exps = sorted({Q.parse_expiry(s) for s in fut})
    t0 = time.time()
    res, series = {}, {}
    for name, th in candidates().items():
        r = Q.simulate(spot, fut, th, fees, TRAIN_FROM, TEST_FROM, start_equity=START_EQUITY, expiries=exps,
                       max_expiry_ms=max_exp)
        reasons = {}
        for t in r["trades"]:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        res[name] = {"theta": th, "agg": r["agg"], "n_trades": len(r["trades"]), "exit_reasons": reasons,
                     "liquidations": r["liquidations"], "margin_stops": r["margin_stops"],
                     "r_by_year": _year_means(r["series"]), "conservation_ok": r["conservation_ok"],
                     "trade_mean_pct": float(np.mean([t["ret_pct"] for t in r["trades"]])) if r["trades"] else None,
                     "entry_basis_median": float(np.median([t["basis_ann"] for t in r["trades"]])) if r["trades"] else None}
        series[name] = r["series"]
    base = series["QB_T0"]
    fc = _fc_passive_series(TRAIN_FROM, TEST_FROM)
    for name, v in res.items():
        v["paired_t0"] = DR.paired(series[name], base) if v["theta"] > 0 else None
        v["paired_fc_passive"] = DR.paired(series[name], fc["series"])
        ok, checks = admit(v)
        v["admitted"], v["checks"] = ok, checks
        g = v["agg"]
        print(f"{name:7s} R {g['mean_pct']:+.4f} [{g['lo95']:+.4f}, {g['hi95']:+.4f}] ~{g['mean_pct'] * 365:.1f}%/yr | "
              f"dd {g['max_dd_pct']:+.2f}% | trades {v['n_trades']} {v['exit_reasons']} | liq {v['liquidations']} | "
              f"years { {y: round(x, 4) for y, x in v['r_by_year'].items()} } | basis med {v['entry_basis_median']} | "
              f"vs FC {v['paired_fc_passive']['mean_pct']:+.4f} [{v['paired_fc_passive']['lo95']:+.4f}, "
              f"{v['paired_fc_passive']['hi95']:+.4f}] | {'ADMIT' if ok else '-'}", flush=True)
    names = list(res)
    days = sorted(set.intersection(*[set(dict(series[n])) for n in names]))
    R = np.array([[dict(series[n])[d] for n in names] for d in days])
    pbo = float(PB.cscv_pbo(R, PBO_SPLITS))
    winner = select(res, pbo)
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "contracts": sorted(fut),
           "fees": {"spot_taker": fees.spot_taker, "perp_taker": fees.perp_taker, "slip": fees.slip},
           "fc_passive_agg": fc["agg"], "candidates": res, "pbo": pbo, "winner": winner,
           "deflated_sharpe": XE._deflated([v for _, v in series[winner]], len(names)) if winner else None,
           "verdict": "winner -> test needs the operator" if winner else "STOP: no winner, test not opened",
           "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pbo": pbo, "winner": winner, "fc_passive": fc["agg"]["mean_pct"], "verdict": out["verdict"]}))
    return out


def last_full_day(spot: dict) -> str:
    """Last UTC day with all 24 hourly bars for every pair (spec §9)."""
    last = None
    for a in spot.values():
        t = np.asarray(a["t"], dtype=np.int64)
        days, cnt = np.unique(t // Q.D, return_counts=True)
        full = days[cnt == 24]
        d = int(full[-1]) if len(full) else None
        last = d if last is None or (d is not None and d < last) else last
    return datetime.fromtimestamp(last * Q.D / 1000, timezone.utc).strftime("%Y-%m-%d")


def decide_t0(agg: dict, liquidations: int) -> dict:
    checks = {"R_lo_gt_0": agg.get("lo95") is not None and agg["lo95"] > 0, "no_liquidation": liquidations == 0,
              "dd_ok": agg.get("max_dd_pct") is not None and agg["max_dd_pct"] >= DD_FLOOR_PCT}
    return {"accepted": all(checks.values()), "checks": checks}


def run_test_t0() -> dict:
    """§9: QB_T0 once on 2026; marker first, then the run; all contracts allowed."""
    import carry as C
    import daily_revenue as DR
    import money_validator as MV
    spot, fut = load()
    end = last_full_day(spot)
    day_to = datetime.fromtimestamp((Q._ms(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)) + Q.D) / 1000,
                                    timezone.utc).strftime("%Y-%m-%d")
    open_test(TEST_MARKER, {"spec": SPEC, "rule": "QB_T0", "window": [TEST_FROM, day_to], "k": 1})
    r = Q.simulate(spot, fut, 0.0, C.fees(1), TEST_FROM, day_to, start_equity=START_EQUITY,
                   expiries=sorted({Q.parse_expiry(s) for s in fut}))
    v = decide_t0(r["agg"], r["liquidations"])
    try:
        fc = _fc_passive_series(TEST_FROM, day_to)
        ref = {"fc_passive_agg": fc["agg"], "paired_fc_passive": DR.paired(r["series"], fc["series"])}
    except Exception as ex:
        ref = {"fc_passive_error": repr(ex)[:300]}
    reasons = {}
    for t in r["trades"]:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    out = {"spec": SPEC, "stage": "test", "rule": "QB_T0", "window": [TEST_FROM, day_to], "verdict": v, "agg": r["agg"],
           "liquidations": r["liquidations"], "margin_stops": r["margin_stops"], "exit_reasons": reasons,
           "trades": [{k: t[k] for k in ("pair", "symbol", "reason", "basis_ann", "ret_pct")} for t in r["trades"]],
           "conservation_ok": r["conservation_ok"], "reference": ref, "utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "QB_TEST.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    MV.log_trial({"family": "QB", "name": "QB_T0", "k_opened_on_test": 1, **v, "test_R": r["agg"]})
    g = r["agg"]
    print(f"QB_T0 TEST {TEST_FROM}..{day_to}: R {g['mean_pct']:+.4f} [{g['lo95']:+.4f}, {g['hi95']:+.4f}] "
          f"~{g['mean_pct'] * 365:.2f}%/yr | dd {g['max_dd_pct']:+.2f}% | trades {reasons} | liq {r['liquidations']} | "
          f"{'ACCEPTED' if v['accepted'] else 'REJECTED'}", flush=True)
    print(json.dumps(ref, default=str)[:600])
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "fetch":
        fetch()
        return 0
    if cmd == "train":
        run_train()
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("QB: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        run_test_t0()
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
