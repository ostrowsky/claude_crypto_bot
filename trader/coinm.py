"""G3a -- carry on COIN-M: spot coin as the margin of an inverse perp short (coinm-carry-g3a-1105-spec.md).

  simulate_pair  coin + N * (1/f_t - 1/f_e), valued at spot; N = the coin's value, so the notional is the whole capital
                 (FC_PASSIVE at L = 1 holds half). The short receives rate * N / f in coin at rate > 0. Once a day the
                 short's profit is booked and N is reset to coin * f if that changes it by >= 0.5%.
  compare        FC_PASSIVE on the same days and costs (carry.fees(1)), and the COIN-M vs USD-M funding rate itself
  fetch          the archive files of BTCUSD_PERP and ETHUSD_PERP into runtime/coinm/ (operator's permission, 2026-11-05)
  test           the ONE 2026 look (spec section 8, operator 2026-11-18): 2026-01..06, paired lo > 0, k = 1; a marker refuses
                 a second look

  (from trader/) ../pyembed/python.exe -u coinm.py fetch | run | test
"""
from __future__ import annotations

import bisect
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import archive_bh as A            # noqa: E402
import trader_config as TC        # noqa: E402

SPEC = "coinm-carry-g3a-1105-spec.md"
STORE = TC.RUNTIME / "coinm"
OUT = TC.RUNTIME / "experiments" / "G3A_TRAIN.json"
PAIRS = {"BTCUSDT": "BTCUSD_PERP", "ETHUSDT": "ETHUSD_PERP"}
BAND = 0.005
BAR = 15 * 60 * 1000
DAY = 86_400_000
WINDOW = ("2023-03-01", "2026-01-01")
BLOCK = 20
TEST_WINDOW = ("2026-01-01", "2026-07-01")      # the funding archive on disk ends 2026-06 (spec 8.1)
TEST_ALPHA = 0.05                              # k = 1
TEST_OUT = TC.RUNTIME / "experiments" / "G3A_TEST.json"
TEST_MARKER = TC.RUNTIME / "experiments" / "G3A_TEST_OPENED.json"
MARGIN_THIN = 0.01
DD_FLOOR_PCT = -10.0                           # as experiment_fc.DD_FLOOR_PCT
COVERAGE_MIN = 0.95


@dataclass(frozen=True)
class Fees:
    spot: float
    perp: float


parse_cm_klines = A.parse_klines
parse_cm_funding = A.parse_funding


def simulate_pair(marks, funding, fees: Fees, capital: float = 1.0) -> dict:
    t0, f0, s0 = marks[0]
    costs = capital * fees.spot
    coin = capital * (1 - fees.spot) / s0
    cost_coin = coin * fees.perp
    costs += cost_coin * s0
    coin -= cost_coin
    n = coin * f0
    fe = f0
    n0 = n
    ev = sorted((int(t), float(r), float(p)) for t, r, p in funding)
    ts = [e[0] for e in ev]
    series = [(int(t0), coin * s0)]
    fund_usd, rebalances, min_ratio = 0.0, 0, 1.0
    for j in range(1, len(marks)):
        tj, fj, sj = marks[j]
        for k in range(bisect.bisect_right(ts, int(marks[j - 1][0])), bisect.bisect_right(ts, int(tj))):
            _, rate, px = ev[k]
            got = rate * n / px
            coin += got
            fund_usd += got * sj
        total = coin + n * (1 / fj - 1 / fe)
        if n > 0:
            min_ratio = min(min_ratio, total / (n / fj))
        last = j == len(marks) - 1
        if not last:
            target = total * fj
            if n > 0 and abs(target - n) / n >= BAND:
                c = abs(target - n) * fees.perp / fj
                coin = total - c
                costs += c * sj
                n, fe = target, fj
                rebalances += 1
                total = coin
            series.append((int(tj), total * sj))
        else:
            c_perp = n / fj * fees.perp
            usd = (total - c_perp) * sj
            c_spot = usd * fees.spot
            costs += c_perp * sj + c_spot
            series.append((int(tj), usd - c_spot))
    return {"series": series, "funding_usd": fund_usd, "costs_usd": costs, "min_margin_ratio": min_ratio,
            "rebalances": rebalances, "n0": n0}


# ------------------------------------------------------------------ data
def fetch(log=print) -> dict:
    STORE.mkdir(parents=True, exist_ok=True)
    ex = ThreadPoolExecutor(8)
    tot = {"kline_months": 0, "funding_months": 0, "errors": 0}
    kl = ("o", "h", "l", "c", "v")
    for perp in PAIRS.values():
        pk = f"data/futures/cm/monthly/klines/{perp}/15m/"
        pf = f"data/futures/cm/monthly/fundingRate/{perp}/"
        mk = A.months_in_range(A.list_all(pk, delimiter=False), "2020-01", "2026-12")
        mf = A.months_in_range(A.list_all(pf, delimiter=False), "2020-01", "2026-12")
        a = A.fetch_series(pk, perp, mk, STORE / f"{perp}_15m.npz", parse_cm_klines, kl, ex)
        b = A.fetch_series(pf, perp, mf, STORE / f"{perp}_funding.npz", parse_cm_funding, ("rate", "mark"), ex)
        tot["kline_months"] += a.get("new", 0)
        tot["funding_months"] += b.get("new", 0)
        tot["errors"] += a.get("errors", 0) + b.get("errors", 0)
        log(f"{perp}: klines {mk[0]}..{mk[-1]} ({len(mk)}), funding {mf[0]}..{mf[-1]} ({len(mf)})")
    ex.shutdown()
    log(json.dumps(tot))
    return tot


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def load_pair(spot_sym: str, perp: str, lo: str, hi: str):
    """Marks at the close of each day's first 15m bar, on days in [lo, hi); funding events up to the last mark only."""
    import history_store as HS
    with np.load(STORE / f"{perp}_15m.npz") as z:
        tp, cp = z["t"].astype(np.int64), z["c"].astype(float)
    with np.load(STORE / f"{perp}_funding.npz") as z:
        ft, fr = z["t"].astype(np.int64), z["rate"].astype(float)
    sp = HS.load(HS.path_of(spot_sym))
    ts, cs = np.asarray(sp["t"], dtype=np.int64), np.asarray(sp["c"], dtype=float)
    marks = []
    for d in range(_ms(lo), _ms(hi), DAY):
        i, j = int(np.searchsorted(tp, d)), int(np.searchsorted(ts, d))
        if i < len(tp) and j < len(ts) and int(tp[i]) == d and int(ts[j]) == d and np.isfinite(cp[i]) and np.isfinite(cs[j]):
            marks.append((d + BAR, float(cp[i]), float(cs[j])))
    last = marks[-1][0]
    ev = []
    for t, r in zip(ft, fr):
        if marks[0][0] < t <= last and np.isfinite(r):
            k = int(np.searchsorted(tp, int(t) - int(t) % BAR))
            if k < len(tp) and np.isfinite(cp[k]):
                ev.append((int(t), float(r), float(cp[k])))
    return marks, ev, (ft, fr)


def _month_means(series) -> dict:
    by = {}
    for d, v in series:
        by.setdefault(d[:7], []).append(v)
    return {m: float(np.mean(v)) for m, v in sorted(by.items())}


def _year_means(series) -> dict:
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: float(np.mean(v)) for y, v in sorted(by.items())}


def compute(lo: str, hi: str, alpha: float = 0.05) -> dict:
    """COIN-M carry and FC_PASSIVE on the same days [lo, hi), paired at 1 - alpha, block BLOCK. Writes nothing."""
    import carry as C
    import daily_revenue as DR
    import experiment_fc as EF
    import funding_store as FS
    cf = C.fees(1)
    fees = Fees(cf.spot_taker + cf.slip, cf.perp_taker + cf.slip)
    res, eq = {}, None
    rate_rows = {}
    for spot, perp in PAIRS.items():
        marks, ev, (ft, fr) = load_pair(spot, perp, lo, hi)
        r = simulate_pair(marks, ev, fees, capital=0.5)
        res[perp] = {k: r[k] for k in ("funding_usd", "costs_usd", "min_margin_ratio", "rebalances", "n0")}
        res[perp].update(n_marks=len(marks), n_funding=len(ev), first=_day(marks[0][0]), last=_day(marks[-1][0]))
        s = dict(r["series"])
        eq = s if eq is None else {t: eq[t] + s[t] for t in eq if t in s}
        fu = FS.load_funding(FS.funding_path(spot))
        d_cm, d_um = {}, {}
        for t, x in zip(ft, fr):
            if _ms(lo) <= t < _ms(hi):
                d_cm[_day(int(t))] = d_cm.get(_day(int(t)), 0.0) + float(x) * 100
        for t, x in zip(fu["t"], fu["rate"]):
            if _ms(lo) <= t < _ms(hi):
                d_um[_day(int(t))] = d_um.get(_day(int(t)), 0.0) + float(x) * 100
        rate_rows[perp] = {"cm": sorted(d_cm.items()), "um": sorted(d_um.items())}
    ts = sorted(eq)
    series = [(_day(ts[0]), (eq[ts[0]] / 1.0 - 1) * 100)] + [(_day(b), (eq[b] / eq[a] - 1) * 100) for a, b in zip(ts, ts[1:])]
    fc = EF.evaluate(EF.load_market(), EF.passive_config(), cf, lo, hi, 0)
    paired = DR.paired(series, fc["_series"], alpha=alpha, block=BLOCK)
    agg = DR.aggregate(series, alpha=alpha, block=BLOCK)
    fa = DR.aggregate(fc["_series"], alpha=alpha, block=BLOCK)
    ds, df = dict(series), dict(fc["_series"])
    py = _year_means([(d, ds[d] - df[d]) for d in ds if d in df])
    # the rate itself, % per day, COIN-M - USD-M on common days, average of the two pairs
    rates = {}
    cm_all, um_all = {}, {}
    for perp, x in rate_rows.items():
        c, u = dict(x["cm"]), dict(x["um"])
        pr = DR.paired(sorted(c.items()), sorted(u.items()), alpha=alpha, block=BLOCK)
        rates[perp] = {"cm_mean": float(np.mean(list(c.values()))), "um_mean": float(np.mean(list(u.values()))), "paired": pr}
        for d in set(c) & set(u):
            cm_all.setdefault(d, []).append(c[d])
            um_all.setdefault(d, []).append(u[d])
    worst = min(series, key=lambda x: x[1]) if series else None
    return {"spec": SPEC, "window": [lo, hi], "fees": fees.__dict__, "pairs": res, "agg": agg, "fc_passive": fa,
            "paired_vs_fc": paired, "paired_by_year": py, "r_by_year": _year_means(series), "fc_by_year": _year_means(fc["_series"]),
            "paired_by_month": _month_means([(d, ds[d] - df[d]) for d in ds if d in df]), "worst_day": worst,
            "fc_margin_stops": fc["margin_stops"], "fc_liquidations": fc["liquidations"], "rates": rates}


def run() -> dict:
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    out = {"stage": "train", **compute(*WINDOW)}
    out["window"] = WINDOW
    res, py, paired = out["pairs"], out["paired_by_year"], out["paired_vs_fc"]
    thin = min(v["min_margin_ratio"] for v in res.values()) < 0.01
    pos = sum(1 for v in py.values() if v > 0)
    out["passes"] = bool(paired["lo95"] > 0 and pos >= 2 and not thin)
    out["verdict"] = "propose a single 2026 look to the operator" if out["passes"] else "no gain shown: STOP"
    out["provenance"] = PV.block()
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pairs", "agg", "fc_passive", "paired_vs_fc", "paired_by_year", "r_by_year", "fc_by_year",
                                          "fc_margin_stops", "rates", "passes", "verdict")}, indent=1, default=str))
    return out


# ------------------------------------------------------------------ the one 2026 look (spec section 8)
def coverage_ok(n_marks: int, n_days: int, last_funding_ms, hi: str) -> bool:
    return (n_days > 0 and n_marks >= COVERAGE_MIN * n_days and last_funding_ms is not None
            and int(last_funding_ms) >= _ms(hi) - DAY)


def data_coverage(lo: str, hi: str) -> dict:
    """Timestamps only, no prices or results: days with a perp and a spot mark, and the last funding event in the window."""
    import history_store as HS
    n_days = int((_ms(hi) - _ms(lo)) // DAY)
    out = {"n_days": n_days, "pairs": {}}
    for spot, perp in PAIRS.items():
        with np.load(STORE / f"{perp}_15m.npz") as z:
            tp = set(int(t) for t in z["t"])
        with np.load(STORE / f"{perp}_funding.npz") as z:
            ft = [int(t) for t in z["t"] if _ms(lo) <= int(t) < _ms(hi)]
        ts = set(int(t) for t in HS.load(HS.path_of(spot))["t"])
        n_marks = sum(1 for d in range(_ms(lo), _ms(hi), DAY) if d in tp and d in ts)
        last = max(ft) if ft else None
        out["pairs"][perp] = {"n_marks": n_marks, "last_funding": _day(last) if last else None, "n_funding": len(ft),
                              "ok": coverage_ok(n_marks, n_days, last, hi)}
    out["ok"] = all(v["ok"] for v in out["pairs"].values())
    return out


def require_train(path) -> None:
    p = Path(path)
    if not p.exists():
        raise RuntimeError("G3a: no train result at %s; the 2026 look needs a passed train" % p)
    if not json.loads(p.read_text(encoding="utf-8")).get("passes"):
        raise RuntimeError("G3a: the train result did not pass section 4; no 2026 look")


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("G3a: the 2026 look was already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def decide_test(paired: dict, min_margin, agg: dict) -> dict:
    lo = (paired or {}).get("lo95")
    dd = (agg or {}).get("max_dd_pct")
    v = {"gain_shown": lo is not None and lo > 0, "margin_ok": min_margin is not None and min_margin >= MARGIN_THIN,
         "dd_ok": dd is not None and dd >= DD_FLOOR_PCT}
    v["outcome"] = "accepted" if all(v.values()) else "not_shown"
    return v


def run_test(train_path=OUT, out_path=TEST_OUT, marker=TEST_MARKER, compute_fn=None, coverage_fn=None, log_fn=None,
             check_clean: bool = True) -> dict:
    import provenance as PV
    if check_clean:
        PV.require_clean()
    require_train(train_path)
    lo, hi = TEST_WINDOW
    cov = (coverage_fn or data_coverage)(lo, hi)
    if not cov.get("ok"):
        raise RuntimeError("G3a: data do not cover %s..%s; the look is not spent: %s" % (lo, hi, json.dumps(cov, default=str)))
    open_test(marker, {"spec": SPEC, "window": [lo, hi], "alpha": TEST_ALPHA, "k": 1})
    t0 = time.time()
    out = (compute_fn or compute)(lo, hi, TEST_ALPHA)
    min_margin = min(v["min_margin_ratio"] for v in out["pairs"].values())
    v = decide_test(out["paired_vs_fc"], min_margin, out["agg"])
    out.update(stage="test", window=[lo, hi], alpha=TEST_ALPHA, k_opened_on_test=1, coverage=cov, decision=v,
               min_margin_ratio=min_margin, minutes=round((time.time() - t0) / 60, 1))
    if check_clean:
        out["provenance"] = PV.block()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if log_fn is None:
        import money_validator as MV
        log_fn = MV.log_trial
    log_fn({"family": "G3A", "name": "COINM_CARRY", "k_opened_on_test": 1, "alpha": TEST_ALPHA, **v,
            "test_R": out["agg"], "paired_vs_fc": out["paired_vs_fc"]})
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fetch"]:
        fetch(log=lambda s: print(s, flush=True))
    elif argv[:1] == ["run"]:
        run()
    elif argv[:1] == ["test"]:
        r = run_test()
        print(json.dumps({k: r.get(k) for k in ("window", "alpha", "coverage", "pairs", "agg", "fc_passive", "paired_vs_fc",
                                                "paired_by_month", "worst_day", "rates", "min_margin_ratio", "decision")},
                         indent=1, default=str))
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
