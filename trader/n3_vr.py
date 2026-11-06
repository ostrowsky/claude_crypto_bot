"""N3 stage 1 -- do USDT perps mean-revert on 1-24 h? Lo-MacKinlay variance ratio of 15m closes (n3-1106-spec.md).

  fetch    1d of every USDT perp (the rank) and 15m of any top-20 symbol the stores lack
  explore  2023-01..2025-12: N3_VR_EXPLORE.json, sigma and rho for the power calculation
  bh       2020-07..2022-12: refuses without a ledger entry; one look, then the GO rule

  (from trader/) ../pyembed/python.exe -u n3_vr.py fetch | explore | bh
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import archive_bh as A            # noqa: E402
import n_common as NC             # noqa: E402
import trader_config as TC        # noqa: E402

SPEC = "n3-1106-spec.md"
BAR = 15 * 60 * 1000
DAY = 86_400_000
MIN_RETURNS = 1440
MAIN = (4, 8, 16, 32, 96)
REF = (2, 192, 288)
TOP = 20
ALPHA = 0.05 / 5
DELTA = 0.02
BLOCK = 3
EXPLORE = ("2023-01", "2025-12")
DECISIVE = ("2020-07", "2022-12")
EXP = TC.RUNTIME / "experiments"
EXTRA = TC.RUNTIME / "usdc_n3"
MARKER = EXP / "N3_VR_BH_OPENED.json"
LEDGER = HERE.parent / "docs" / "audit" / "alpha_ledger.json"
LEDGER_ID = "N3-VR-BH"


def returns(t, c, v):
    t, c, v = np.asarray(t, dtype=np.int64), np.asarray(c, dtype=float), np.asarray(v, dtype=float)
    good = (v > 0) & (c > 0) & np.isfinite(c)
    ok = (np.diff(t) == BAR) & good[1:] & good[:-1]
    r = np.zeros(len(t) - 1)
    r[ok] = np.log(c[1:][ok] / c[:-1][ok])
    return r, ok


def vr(r, ok, q: int, robust: bool = True) -> dict:
    r, ok = np.asarray(r, dtype=float), np.asarray(ok, dtype=bool)
    n = int(ok.sum())
    mu = float(r[ok].mean())
    d = np.where(ok, r - mu, 0.0)
    ss = float((d ** 2).sum())
    sa = ss / (n - 1)
    cs = np.r_[0.0, np.cumsum(np.where(ok, r, 0.0))]
    cok = np.r_[0, np.cumsum(ok)]
    full = (cok[q:] - cok[:-q]) == q
    w = int(full.sum())
    s = (cs[q:] - cs[:-q])[full]
    m = q * w * (1 - q / n)
    value = float(((s - q * mu) ** 2).sum() / m / sa)
    if not robust:
        return {"vr": value, "z_star": None, "n": n, "windows": w}
    d2 = d ** 2
    theta = 0.0
    for j in range(1, q):
        pair = float((d2[j:] * d2[:-j]).sum())
        theta += (2 * (q - j) / q) ** 2 * n * pair / ss ** 2
    z = n ** 0.5 * (value - 1) / theta ** 0.5 if theta > 0 else float("nan")
    return {"vr": value, "z_star": float(z), "n": n, "windows": w}


def symbol_month(t, c, v, horizons):
    r, ok = returns(t, c, v)
    if int(ok.sum()) < MIN_RETURNS:
        return None
    out = {q: vr(r, ok, q) for q in horizons}
    out["sigma"] = float(r[ok].std(ddof=1))
    out["gap_share"] = 1 - int(ok.sum()) / max(1, len(ok))
    return out


def pool_months(rows: dict) -> list:
    return [(m, float(np.mean(list(rows[m].values())))) for m in sorted(rows) if rows[m]]


def go(pool_ci: dict, years: dict, horizons=MAIN) -> dict:
    for a, b in zip(horizons, horizons[1:]):
        if all(pool_ci[q]["mean"] < 0 and pool_ci[q]["hi"] is not None and pool_ci[q]["hi"] < 0
               and sum(1 for v in years[q].values() if v < 0) >= 4 for q in (a, b)):
            return {"go": True, "pair": (a, b)}
    return {"go": False, "pair": None}


def terciles(sigma: dict) -> dict:
    s = sorted(sigma, key=lambda k: sigma[k])
    return {k: i * 3 // len(s) for i, k in enumerate(s)}


def sigma_before(t, c, v, start_ms: int) -> float:
    t = np.asarray(t, dtype=np.int64)
    m = (t >= start_ms - 30 * DAY) & (t < start_ms)
    if m.sum() < 100:
        return float("nan")
    r, ok = returns(t[m], np.asarray(c)[m], np.asarray(v)[m])
    return float(r[ok].std(ddof=1)) if ok.sum() > 50 else float("nan")


def bounce_bias(c: float, sigma: float, q: int) -> float:
    return -(2 * c ** 2 / sigma ** 2) * (1 - 1 / q)


def wild(r, ok, rng) -> np.ndarray:
    """Control: random signs keep |r| and its clusters and remove any serial dependence of the signs."""
    return np.asarray(r, dtype=float) * rng.choice((-1.0, 1.0), size=len(r))


# ------------------------------------------------------------------ data
def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _perp_to_spot() -> dict:
    import funding_store as FS
    return {v.get("perp"): k for k, v in json.loads(FS.MAPPING.read_text(encoding="utf-8")).items()}


def _paths(perp: str, p2s: dict, era: str) -> list:
    extra = EXTRA / f"{perp}_{era}_15m.npz"
    if era == "bh":
        return [A.PERP / f"{perp}_15m.npz", extra]
    spot = p2s.get(perp)
    return ([TC.RUNTIME / "perp_history" / f"{spot}_15m.npz"] if spot else []) + [extra]


def _has_month(path: Path, month: str) -> bool:
    if not path.exists():
        return False
    first, last = NC.month_days(month)
    with np.load(path) as z:
        t = z["t"]
    return bool(((t >= _ms(first)) & (t < _ms(last) + DAY)).any())


def universes(vol, first_month: str, last_month: str) -> dict:
    ex = NC.pegged()
    return {m: vol.top(NC.month_days(m)[0], TOP, exclude=ex) for m in NC.months_between(first_month, last_month)}


def fetch(log=print) -> dict:
    NC.fetch_usdt_1d(log=log)
    vol = NC.load_volumes()
    p2s = _perp_to_spot()
    ex = ThreadPoolExecutor(8)
    out = {}
    for era, (lo, hi), key in (("bh", ("2020-01", "2022-12"), DECISIVE), ("recent", ("2023-01", "2025-12"), EXPLORE)):
        need = {}
        for m, syms in universes(vol, *key).items():
            for s in syms:
                need.setdefault(s, []).append(m)
        missing = {s: ms for s, ms in need.items() if not any(_has_month(p, ms[0]) for p in _paths(s, p2s, era))}
        out[era] = {"symbols_in_universe": len(need), "missing_in_store": sorted(ascii(s) for s in missing)}
        for s in missing:
            NC.fetch_months(f"data/futures/um/monthly/klines/{s}/15m/", EXTRA / f"{s}_{era}_15m.npz", A.parse_klines, ("o", "h", "l", "c", "v"), ex, first=lo, last=hi)
    ex.shutdown()
    log(json.dumps(out))
    return out


def _load(perp: str, p2s: dict, era: str, lo_ms: int, hi_ms: int):
    """Bars in [lo_ms, hi_ms) only: later data of the same file is never read into the analysis."""
    best = None
    for p in _paths(perp, p2s, era):
        if p.exists():
            with np.load(p) as z:
                t = z["t"].astype(np.int64)
                k = (t >= lo_ms) & (t < hi_ms)
                a = {"t": t[k], "c": z["c"][k].astype(float), "v": z["v"][k].astype(float)}
            if best is None or len(a["t"]) > len(best["t"]):
                best = a
    return best


def _bounce_c() -> dict:
    cal = json.loads((EXP / "COST_CAL.json").read_text(encoding="utf-8"))["tables"]["live"]
    out = {}
    for b in ("B1", "B2"):
        out[b] = (cal[b]["buy"]["1000"]["median"] + cal[b]["sell"]["1000"]["median"]) / 2 / 1e4
    return out


def study(first: str, last: str, era: str, bounce_mult: float) -> dict:
    vol = NC.load_volumes()
    p2s = _perp_to_spot()
    uni = universes(vol, first, last)
    lo_ms, hi_ms = _ms(NC.month_days(first)[0]) - 31 * DAY, _ms(NC.month_days(last)[1]) + DAY
    horizons = MAIN + REF
    rows = {q: {} for q in horizons}
    zs = {q: [] for q in horizons}
    tercile_rows = {q: {0: {}, 1: {}, 2: {}} for q in MAIN}
    per_btc, per_eth = {q: [] for q in horizons}, {q: [] for q in horizons}
    dropped, no_data, gaps, used = 0, set(), [], set()
    bounce = {q: [] for q in MAIN}
    cost = _bounce_c()
    cache = {}
    for month, syms in uni.items():
        start = _ms(NC.month_days(month)[0])
        end = _ms(NC.month_days(month)[1]) + DAY
        sig = {}
        res = {}
        for s in syms:
            if s not in cache:
                cache[s] = _load(s, p2s, era, lo_ms, hi_ms)
            a = cache[s]
            if a is None:
                no_data.add(s)
                continue
            k = (a["t"] >= start) & (a["t"] < end)
            x = symbol_month(a["t"][k], a["c"][k], a["v"][k], horizons) if k.sum() > MIN_RETURNS else None
            if x is None:
                dropped += 1
                continue
            res[s] = x
            used.add(s)
            sig[s] = sigma_before(a["t"], a["c"], a["v"], start)
            gaps.append(x["gap_share"])
        for q in horizons:
            rows[q][month] = {s: x[q]["vr"] - 1 for s, x in res.items()}
            zs[q] += [x[q]["z_star"] for x in res.values()]
            if "BTCUSDT" in res:
                per_btc[q].append(res["BTCUSDT"][q]["vr"] - 1)
            if "ETHUSDT" in res:
                per_eth[q].append(res["ETHUSDT"][q]["vr"] - 1)
        for s, x in res.items():
            c = cost["B1"] if s in ("BTCUSDT", "ETHUSDT") else cost["B2"]
            for q in MAIN:
                bounce[q].append(bounce_bias(c * bounce_mult, x["sigma"], q))
        known = {s: v for s, v in sig.items() if np.isfinite(v)}
        if known:
            g = terciles(known)
            for q in MAIN:
                for s, t in g.items():
                    tercile_rows[q][t].setdefault(month, {})[s] = res[s][q]["vr"] - 1
    out = {"horizons": {}, "window": [first, last], "months": len(uni), "symbols_used": len(used), "symbol_months_dropped_below_min": dropped,
           "symbols_without_data": sorted(ascii(s) for s in no_data), "gap_share_mean": float(np.mean(gaps)) if gaps else None,
           "pool_series": {}, "universe_sample": {m: [ascii(s) for s in uni[m]] for m in list(uni)[:: max(1, len(uni) // 6)]}}
    for q in horizons:
        series = pool_months(rows[q])
        vals = [v for _, v in series]
        c = NC.ci(vals, alpha=ALPHA, block=BLOCK)
        z = np.array([x for x in zs[q] if np.isfinite(x)])
        by_year = {}
        for m, v in series:
            by_year.setdefault(m[:4], []).append(v)
        rho = float(np.corrcoef(vals[:-1], vals[1:])[0, 1]) if len(vals) > 3 else 0.0
        item = {"pool": c, "sigma_month": float(np.std(vals, ddof=1)), "rho_lag1": rho, "by_year": {y: float(np.mean(v)) for y, v in sorted(by_year.items())},
                "share_months_negative": float(np.mean(np.array(vals) < 0)), "symbol_months": int(len(z)), "z_star": {"median": float(np.median(z)), "share_below_m196": float((z < -1.96).mean()), "share_above_p196": float((z > 1.96).mean())},
                "btc": float(np.mean(per_btc[q])) if per_btc[q] else None, "eth": float(np.mean(per_eth[q])) if per_eth[q] else None, "main": q in MAIN}
        if q in MAIN:
            item["terciles"] = {str(t): NC.ci([v for _, v in pool_months(tercile_rows[q][t])], alpha=ALPHA, block=BLOCK) for t in (0, 1, 2)}
            b = float(np.mean(bounce[q])) if bounce[q] else 0.0
            item["bounce"] = {"mean_bias": b, "mult": bounce_mult, "ratio_to_effect": abs(b) / abs(c["mean"]) if c["mean"] else None}
            n, ne = NC.required_n(item["sigma_month"], DELTA, ALPHA, m=BLOCK, rho=max(0.0, rho))
            item["power"] = {"delta": DELTA, "alpha": ALPHA, "n": n, "n_eff": ne, "m": BLOCK, "sigma_from": "this window", "rho": max(0.0, rho)}
        out["horizons"][str(q)] = item
        out["pool_series"][str(q)] = series
    return out


def explore() -> dict:
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    r = study(*EXPLORE, era="recent", bounce_mult=1.0)
    out = {"spec": SPEC, "stage": "exploration", **r, "verdict": None, "traps_checked": [4, 10], "traps_not_applicable": {"1": "no control", "2": "no entry", "3": "no spot", "5": "no PBO", "6": "no kill rule", "7": "no funding at stage 1", "8": "no execution", "9": "USDT perps only"}, "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 2)}
    (EXP / "N3_VR_EXPLORE.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _print(out)
    return out


def _print(o: dict) -> None:
    for q, x in o["horizons"].items():
        c = x["pool"]
        print(f"q={q:>3} pool {c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] {c['ci_level']} {c['ci_method']} sd {x['sigma_month']:.4f} rho {x['rho_lag1']:+.2f} neg {x['share_months_negative']:.2f}"
              f" years {({y: round(v, 4) for y, v in x['by_year'].items()})} btc {x['btc']} eth {x['eth']} z<-1.96 {x['z_star']['share_below_m196']:.3f} z>1.96 {x['z_star']['share_above_p196']:.3f}")
        if x.get("main"):
            print("     terciles", {t: (round(v["mean"], 4), round(v["lo"], 4), round(v["hi"], 4)) for t, v in x["terciles"].items()}, "bounce", x["bounce"], "n_eff", round(x["power"]["n_eff"], 1))
    print({k: o[k] for k in ("window", "months", "symbols_used", "symbol_months_dropped_below_min", "symbols_without_data", "gap_share_mean")}, flush=True)


def placebo(draws: int = 40) -> dict:
    """Is the negative pool a property of the month-long estimator on fat, clustered returns? Same symbols, same months, exploration only."""
    import provenance as PV
    PV.require_clean()
    vol = NC.load_volumes()
    p2s = _perp_to_spot()
    uni = universes(vol, *EXPLORE)
    lo_ms, hi_ms = _ms(NC.month_days(EXPLORE[0])[0]), _ms(NC.month_days(EXPLORE[1])[1]) + DAY
    cache, data = {}, {}
    for month, syms in uni.items():
        start, end = _ms(NC.month_days(month)[0]), _ms(NC.month_days(month)[1]) + DAY
        for s in syms:
            if s not in cache:
                cache[s] = _load(s, p2s, "recent", lo_ms, hi_ms)
            a = cache[s]
            if a is None:
                continue
            k = (a["t"] >= start) & (a["t"] < end)
            if k.sum() <= MIN_RETURNS:
                continue
            r, ok = returns(a["t"][k], a["c"][k], a["v"][k])
            if int(ok.sum()) >= MIN_RETURNS:
                data.setdefault(month, {})[s] = (r, ok)
    rng = np.random.default_rng(29)
    pools = {q: [] for q in MAIN}
    for _ in range(draws):
        for q in MAIN:
            rows = {m: {s: vr(wild(r, ok, rng), ok, q, robust=False)["vr"] - 1 for s, (r, ok) in x.items()} for m, x in data.items()}
            pools[q].append(float(np.mean([v for _, v in pool_months(rows)])))
    ex = json.loads((EXP / "N3_VR_EXPLORE.json").read_text(encoding="utf-8"))
    out = {"spec": SPEC, "stage": "exploration_control", "window": EXPLORE, "draws": draws, "control": "random signs on the same returns, same months, same symbols", "horizons": {}, "provenance": PV.block()}
    for q in MAIN:
        v = np.array(pools[q])
        obs = ex["horizons"][str(q)]["pool"]["mean"]
        out["horizons"][str(q)] = {"observed": obs, "control_mean": float(v.mean()), "control_sd": float(v.std(ddof=1)), "control_min": float(v.min()), "control_max": float(v.max()), "observed_minus_control": obs - float(v.mean()), "share_control_at_or_below_observed": float((v <= obs).mean())}
    (EXP / "N3_VR_PLACEBO.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["horizons"], indent=0))
    return out


def bh() -> dict:
    """One look. Refuses without the ledger entry; the marker is written before the window is read."""
    import provenance as PV
    PV.require_clean()
    if MARKER.exists():
        raise RuntimeError("N3 decisive window already opened")
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    entries = led if isinstance(led, list) else led.get("looks", led.get("entries", []))
    if not any(e.get("id") == LEDGER_ID for e in entries):
        raise RuntimeError(f"no ledger entry {LEDGER_ID}")
    MARKER.write_text(json.dumps({"opened": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "head": PV.code_state()["head"]}), encoding="utf-8")
    t0 = time.time()
    r = study(*DECISIVE, era="bh", bounce_mult=3.0)
    ex = json.loads((EXP / "N3_VR_EXPLORE.json").read_text(encoding="utf-8"))
    pool = {q: {"mean": r["horizons"][str(q)]["pool"]["mean"], "hi": r["horizons"][str(q)]["pool"]["hi"]} for q in MAIN}
    years = {q: {**r["horizons"][str(q)]["by_year"], **ex["horizons"][str(q)]["by_year"]} for q in MAIN}
    decision = go(pool, years, MAIN)
    lows = {q: r["horizons"][str(q)]["pool"]["lo"] for q in MAIN}
    if decision["go"]:
        verdict = "GO: stage 2"
    elif all(lows[q] > -DELTA for q in MAIN):
        verdict = "rejected: no horizon admits an effect stronger than -0.02"
    else:
        verdict = "not shown: STOP, the grid has no mechanism at 1-24 h"
    out = {"spec": SPEC, "stage": "decisive", **r, "years": {str(q): years[q] for q in MAIN}, "go": decision, "verdict": verdict, "alpha_per_horizon": ALPHA, "traps_checked": [4, 10], "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 2)}
    (EXP / "N3_VR_BH.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _print(out)
    print(json.dumps({"go": decision, "verdict": verdict}, default=str))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fetch"]:
        fetch(log=lambda s: print(s, flush=True))
    elif argv[:1] == ["explore"]:
        explore()
    elif argv[:1] == ["placebo"]:
        placebo()
    elif argv[:1] == ["bh"]:
        bh()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
