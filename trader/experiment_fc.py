"""FC -- train / one-time test of funding carry (carry-funding-money-1021-spec.md §5-§8).

  panel   perp daily panel + funding from runtime/funding and runtime/perp_history (funding_store.py backfill);
          pairs whose perp/spot price ratio is off by > 5% are dropped and logged -> runtime/experiments/fc_market.npz
  train   8 candidates (W 3/14 d, K 5/10, theta_in 10/25 %/yr) and FC_PASSIVE (BTC+ETH) on decision days
          2023-03-01 .. 2025-12-31 (market cut before 2026: no 2026 bar or funding event is read)
          -> runtime/experiments/FC_TRAIN.json
  test    opened ONCE for the admitted hypotheses (winner and/or FC_PASSIVE), alpha = 0.05 / k, --operator-approved

  (from trader/) ../pyembed/python.exe -u experiment_fc.py panel
  (from trader/) ../pyembed/python.exe -u experiment_fc.py train
  (from trader/) ../pyembed/python.exe -u experiment_fc.py test --operator-approved
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import carry as C                 # noqa: E402
import experiment_xsmom as XE     # noqa: E402
import trader_config as TC        # noqa: E402
import xsmom as XS                # noqa: E402

TRAIN_FROM = "2023-03-01"
TEST_FROM = "2026-01-01"
N_RANDOM = 50
MIN_TRADES = 50
PBO_MAX = 0.5
PBO_SPLITS = 8
DSR_MIN = 0.90
N_TRIALS = 8
YEARS_AGREE = 2
DD_FLOOR_PCT = -10.0
RATIO_TOL = 0.05
START_EQUITY = 1_000_000.0
OUT_DIR = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT_DIR / "FC_TRAIN.json"
TEST_OUT = OUT_DIR / "FC_TEST.json"
TEST_MARKER = OUT_DIR / "FC_TEST_OPENED.json"
MARKET_CACHE = OUT_DIR / "fc_market.npz"
SPEC = "carry-funding-money-1021-spec.md"


# ------------------------------------------------------------------ candidates (spec §5)
def candidates() -> dict:
    out = {}
    for W in (3, 14):
        for K in (5, 10):
            for T in (10, 25):
                c = C.CarryConfig(W=W, K=K, theta_in=T / 100)
                out[c.name] = c
    return out


def passive_config() -> C.CarryConfig:
    return C.CarryConfig(W=3, K=2, theta_in=0.0, passive=True)


# ------------------------------------------------------------------ admission, selection, verdicts (spec §7-§8)
def _p95(xs):
    return XE._p95(xs)


def admit(res: dict):
    p95 = _p95(res.get("random_means") or [])
    years = res.get("r_by_year") or {}
    checks = {"mean_gt_0": (res.get("mean_pct") or 0) > 0,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE,
              "beats_random_p95": p95 is not None and res.get("mean_pct") is not None and res["mean_pct"] > p95,
              "beats_passive": (res.get("paired_passive") or {}).get("mean_pct") is not None and res["paired_passive"]["mean_pct"] > 0,
              "no_liquidation": res.get("liquidations", 1) == 0,
              "enough_trades": (res.get("n_trades") or 0) >= MIN_TRADES}
    return all(checks.values()), checks


def admit_passive(res: dict) -> bool:
    years = res.get("r_by_year") or {}
    return ((res.get("mean_pct") or 0) > 0 and res.get("liquidations", 1) == 0
            and sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE)


def row_open(pbo) -> bool:
    return pbo is not None and pbo < PBO_MAX


def select_winner(res: dict, pbo):
    if not row_open(pbo):
        return None
    adm = [n for n, v in res.items() if v.get("admitted")]
    return max(adm, key=lambda n: (res[n]["mean_pct"], n)) if adm else None


def dsr_ok(dsr) -> bool:
    return dsr is not None and dsr >= DSR_MIN


def hypotheses(winner, passive_ok: bool) -> list:
    return ([winner] if winner else []) + (["FC_PASSIVE"] if passive_ok else [])


def alpha_for(k: int) -> float:
    return 0.05 / k


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("FC: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def pipeline(train_fn, test_fn, marker=TEST_MARKER) -> dict:
    names = list(train_fn() or [])
    if not names:
        return {"stop": True, "hypotheses": []}
    open_test(marker, {"spec": SPEC, "hypotheses": names})
    alpha = alpha_for(len(names))
    return {"stop": False, "hypotheses": names, "alpha": alpha, "test": test_fn(names, alpha)}


def verdict(kind: str, day_agg: dict, random_means=None, liquidations: int = 0) -> dict:
    v = {"profitable": day_agg.get("lo95") is not None and day_agg["lo95"] > 0,
         "safe": liquidations == 0 and (day_agg.get("max_dd_pct") is not None and day_agg["max_dd_pct"] >= DD_FLOOR_PCT)}
    if kind == "fc":
        p95 = _p95(random_means or [])
        v["random_p95"] = p95
        v["beats_random"] = p95 is not None and day_agg.get("mean_pct") is not None and day_agg["mean_pct"] > p95
        ok = v["profitable"] and v["safe"] and v["beats_random"]
    else:
        ok = v["profitable"] and v["safe"]
    v["outcome"] = "accepted" if ok else "rejected"
    return v


def train_panel(panel):
    return panel.cut(TEST_FROM)


# ------------------------------------------------------------------ market (perp panel + funding)
def build_market() -> dict:
    import funding_store as FS
    import history_store as HS
    spot = XE.load_panel()
    mapping = json.loads(FS.MAPPING.read_text(encoding="utf-8"))
    daily, mult, funding = {}, {}, {}
    for s, mp in sorted(mapping.items()):
        a = HS.load(FS.perp_path(s))
        f = FS.load_funding(FS.funding_path(s))
        if a is None or not len(a["t"]) or f is None or not len(f["t"]):
            continue
        daily[s], mult[s], funding[s] = C.daily_perp(a), mp["mult"], f
    perp = C.PerpPanel.from_daily(daily, mult=mult)
    dropped = {}
    for s in list(perp.syms):
        js, jp = spot.col.get(s), perp.col[s]
        if js is None:
            dropped[s] = "no spot"
            continue
        lo, hi = max(spot.d0, perp.d0), min(spot.d0 + spot.n, perp.d0 + perp.n)
        if hi <= lo:
            dropped[s] = "no common day"
            continue
        sc = spot.close[lo - spot.d0:hi - spot.d0, js]
        pc = perp.close[lo - perp.d0:hi - perp.d0, jp]
        ok = np.isfinite(sc) & np.isfinite(pc) & (sc > 0)
        if not ok.any():
            dropped[s] = "no common close"
            continue
        ratio = float(np.median(pc[ok] / sc[ok]))
        if abs(ratio - 1) > RATIO_TOL:
            dropped[s] = "price ratio %.3f" % ratio
    keep = [s for s in perp.syms if s not in dropped]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jj = [perp.col[s] for s in keep]
    store = {"days": perp.days, "syms": np.array(keep), "close": perp.close[:, jj], "first_close": perp.first_close[:, jj],
             "qv": perp.qv[:, jj], "complete": perp.complete[:, jj], "has_first": perp.has_first[:, jj],
             "has_bar": perp.has_bar[:, jj], "high": perp.high[:, jj]}
    for s in keep:
        store[f"f_t__{s}"], store[f"f_r__{s}"] = funding[s]["t"], funding[s]["rate"]
    np.savez_compressed(MARKET_CACHE, **store)
    return {"pairs": len(keep), "dropped": dropped, "file": str(MARKET_CACHE),
            "perp_from": XS.day_str(perp.d0), "perp_to": XS.day_str(perp.d0 + perp.n - 1)}


def load_market() -> C.Market:
    if not MARKET_CACHE.exists():
        raise RuntimeError("FC: %s missing; run `experiment_fc.py panel` first" % MARKET_CACHE)
    z = np.load(MARKET_CACHE, allow_pickle=False)
    syms = [str(s) for s in z["syms"]]
    perp = C.PerpPanel(z["days"], syms, z["close"], z["first_close"], z["qv"], z["complete"], z["has_first"],
                       z["has_bar"], z["high"])
    funding = {s: {"t": z[f"f_t__{s}"], "rate": z[f"f_r__{s}"]} for s in syms}
    return C.Market(spot=XE.load_panel(), perp=perp, funding=funding)


# ------------------------------------------------------------------ evaluation
def evaluate(market, cfg, fm, day_from, day_to, n_random, passive_series=None, alpha: float = 0.05,
             lev_override=None) -> dict:
    import daily_revenue as DR
    import money_validator as MV
    run_cfg = replace(cfg, lev=lev_override) if lev_override else cfg
    r = C.simulate(market, run_cfg, fm, day_from, day_to, start_equity=START_EQUITY)
    rand = (C.random_control(market, run_cfg, fm, day_from, day_to, r["moves"], n_random, start_equity=START_EQUITY)
            if (n_random and not cfg.passive) else [])
    by_year = {}
    for d, v in r["series"]:
        by_year.setdefault(d[:4], []).append(v)
    vals = [v for _, v in r["series"]]
    reasons = {}
    for t in r["trades"]:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    out = {"name": cfg.name, "config": asdict(run_cfg), "window": [day_from, day_to], "mean_pct": r["agg"].get("mean_pct"),
           "agg": r["agg"], "n_trades": len(r["trades"]), "trade_ci": MV.trade_ci([t["ret_pct"] for t in r["trades"]]),
           "exit_reasons": reasons, "liquidations": r["liquidations"], "margin_stops": r["margin_stops"],
           "funding_income": r["funding_income"], "turnover_per_day": r["turnover_per_day"],
           "exposure_share": r["exposure_share"], "random_means": rand, "random_p95": _p95(rand),
           "r_by_year": {y: sum(v) / len(v) for y, v in sorted(by_year.items())},
           "sd_r": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "conservation_ok": r["conservation_ok"],
           "_series": r["series"]}
    if passive_series is not None:
        out["paired_passive"] = DR.paired(r["series"], passive_series, alpha=alpha)
    return out


def _slim(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


def run_train(out_path=TRAIN_OUT, n_random=N_RANDOM) -> dict:
    import fill_model  # noqa: F401  (same environment as the other experiments)
    import pbo as PB
    full = load_market()
    market = full.cut(TEST_FROM)
    fm = C.fees(1)
    t0 = time.time()
    pas = evaluate(market, passive_config(), fm, TRAIN_FROM, TEST_FROM, 0)
    pas_ok = admit_passive(pas)
    print("FC_PASSIVE      R %+.4f [%+.4f, %+.4f] | sd %.4f | trades %d | liq %d | stops %d | funding %.0f | %s"
          % (pas["agg"].get("mean_pct") or 0, pas["agg"].get("lo95") or 0, pas["agg"].get("hi95") or 0, pas["sd_r"] or 0,
             pas["n_trades"], pas["liquidations"], pas["margin_stops"], pas["funding_income"], "ADMIT" if pas_ok else "-"),
          flush=True)
    res, series = {}, {}
    cands = candidates()
    for i, (name, cfg) in enumerate(cands.items(), 1):
        e = evaluate(market, cfg, fm, TRAIN_FROM, TEST_FROM, n_random, passive_series=pas["_series"])
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        series[name] = dict(e["_series"])
        res[name] = _slim(e)
        g = e["agg"]
        print("%d/%d %-15s R %+.4f [%+.4f, %+.4f] | sd %.4f | vs passive %+.4f | rnd p95 %+.4f | trades %d | liq %d | stops %d | %s | %.0f min"
              % (i, len(cands), name, g.get("mean_pct") or 0, g.get("lo95") or 0, g.get("hi95") or 0, e["sd_r"] or 0,
                 e["paired_passive"].get("mean_pct") or 0, e["random_p95"] or 0, e["n_trades"], e["liquidations"],
                 e["margin_stops"], "ADMIT" if ok else "-", (time.time() - t0) / 60), flush=True)
    names = list(cands)
    days = sorted(set.intersection(*[set(series[n]) for n in names]))
    R = np.array([[series[n][d] for n in names] for d in days])
    pbo = float(PB.cscv_pbo(R, PBO_SPLITS)) if len(days) >= PBO_SPLITS * 2 else None
    best = select_winner(res, pbo)
    dsr = XE._deflated(list(series[best].values()), N_TRIALS) if best else None
    winner = best if (best and dsr_ok((dsr or {}).get("dsr"))) else None
    hyps = hypotheses(winner, pas_ok)
    test_days = max(1, (full.spot.d0 + full.spot.n) - XS.day_num(TEST_FROM))
    power = {n: {"sd_r": v.get("sd_r"), "halfwidth_r_test": (1.96 * v["sd_r"] / math.sqrt(test_days)) if v.get("sd_r") else None}
             for n, v in list(res.items()) + [("FC_PASSIVE", pas)]}
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "start_equity": START_EQUITY,
           "pairs_with_perp": len(full.perp.syms), "fees": asdict(fm), "passive": _slim(pas), "passive_admitted": pas_ok,
           "candidates": res, "pbo": pbo, "best": best, "deflated_sharpe": dsr, "winner": winner, "hypotheses": hyps,
           "stop": not hyps, "power_test_days_estimate": test_days, "power": power,
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def run_test(train_path=TRAIN_OUT, out_path=TEST_OUT, marker=TEST_MARKER, n_random=N_RANDOM) -> dict:
    import money_validator as MV
    XE.require_train(train_path)
    train = json.loads(Path(train_path).read_text(encoding="utf-8"))
    market = load_market()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    for name, last in (("spot", XS.day_str(market.spot.d0 + market.spot.n - 1)),
                       ("perp", XS.day_str(market.perp.d0 + market.perp.n - 1))):
        if last < yesterday:
            raise RuntimeError("FC: the %s panel ends %s, before %s; refresh history and rebuild panels first" % (name, last, yesterday))
    cands = candidates()

    def test_fn(names, alpha):
        fm = C.fees(1)
        pas = evaluate(market, passive_config(), fm, TEST_FROM, today, 0, alpha=alpha)
        out = {}
        for name in names:
            if name == "FC_PASSIVE":
                e, v = pas, verdict("passive", pas["agg"], liquidations=pas["liquidations"])
                cfg = passive_config()
            else:
                cfg = cands[name]
                e = evaluate(market, cfg, fm, TEST_FROM, today, n_random, passive_series=pas["_series"], alpha=alpha)
                v = verdict("fc", e["agg"], e["random_means"], e["liquidations"])
            refs = {tag: C.simulate(market, cfg, C.fees(lvl), TEST_FROM, today, start_equity=START_EQUITY)["agg"]
                    for tag, lvl in (("fees_level0", 0), ("fees_level2", 2))}
            refs["lev2"] = _slim(evaluate(market, cfg, fm, TEST_FROM, today, 0, lev_override=2.0))
            out[name] = {"alpha": alpha, "verdict": v, "result": _slim(e), "reference": refs}
            MV.log_trial({"family": "FC", "name": name, "k_opened_on_test": len(names), **v, "test_R": e["agg"]})
        return out

    res = pipeline(lambda: train.get("hypotheses") or [], test_fn, marker)
    res.update({"spec": SPEC, "stage": "test", "window": [TEST_FROM, today], "utc": datetime.now(timezone.utc).isoformat()})
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "panel":
        print(json.dumps(build_market(), indent=1, default=str))
        return 0
    if cmd == "train":
        r = run_train()
        print(json.dumps({k: r.get(k) for k in ("passive_admitted", "pbo", "best", "deflated_sharpe", "winner", "hypotheses",
                                                "stop")}, indent=1, default=str))
        print("admitted:", [n for n, v in r["candidates"].items() if v["admitted"]])
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("FC: the test half is opened once and only on the operator's decision; add --operator-approved")
            return 2
        r = run_test()
        print(json.dumps({k: r.get(k) for k in ("stop", "hypotheses", "alpha")}, indent=1, default=str))
        for n, v in (r.get("test") or {}).items():
            print(n, json.dumps(v["verdict"], default=str))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
