"""N1-D -- carry with the coin as USD-M collateral in Multi-Assets mode, G3b (n1d-1106-spec.md). History only, nothing on the demo.

  simulate_pair  coin Q as collateral + short Q of the perp: Q*S + Q*(P0 - P)*fx + U. USDT balance U takes funding; U < 0 sells coin
  run            G3b-USDT and G3b-USDC against FC_PASSIVE on the same days, same costs

  (from trader/) ../pyembed/python.exe -u n1d_carry.py run
"""
from __future__ import annotations

import bisect
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import n_common as NC             # noqa: E402
import trader_config as TC        # noqa: E402

SPEC = "n1d-1106-spec.md"
EXP = TC.RUNTIME / "experiments"
OUT = EXP / "N1D_G3B_TRAIN.json"
WINDOW = ("2023-03-01", "2026-01-01")
WINDOW_USDC = ("2024-02-02", "2026-01-01")
PAIRS = {"BTCUSDT": "BTCUSDC", "ETHUSDT": "ETHUSDC"}
ALPHA = 0.025
DELTA = 0.003
BUFFER = 0.02
HAIRCUT = 0.95
EXCHANGE = 0.001 + 0.001
CM_TAKER = 0.0004
BAR = 900_000
DAY = 86_400_000


@dataclass(frozen=True)
class Fees:
    spot: float
    perp: float
    exchange: float = EXCHANGE


def simulate_pair(marks, funding, fees: Fees, capital: float = 1.0, buffer: float = BUFFER, haircut: float = HAIRCUT, mmr: float = 0.01) -> dict:
    t0, p0, s0, fx0, _ = marks[0]
    costs = capital * (1 - buffer) * fees.spot
    q_c = capital * (1 - buffer) * (1 - fees.spot) / s0
    q_s = q_c
    c_perp = q_s * p0 * fx0 * fees.perp
    costs += c_perp
    u = capital * buffer - c_perp
    ev = sorted((int(t), float(r), float(p), float(f)) for t, r, p, f in funding)
    ts = [e[0] for e in ev]
    series = [(int(t0), u + q_c * s0)]
    fund, exchanges, min_ratio = 0.0, [], float("inf")

    def ratio(pp, ss, fx):
        return (haircut * q_c * ss + u + q_s * (p0 - pp) * fx) / (mmr * q_s * pp * fx)

    min_ratio = min(min_ratio, ratio(p0, s0, fx0))
    for j in range(1, len(marks)):
        tj, pj, sj, fxj, hj = marks[j]
        prev_s = marks[j - 1][2]
        for k in range(bisect.bisect_right(ts, int(marks[j - 1][0])), bisect.bisect_right(ts, int(tj))):
            tk, rate, px, fx = ev[k]
            got = rate * q_s * px * fx
            fund += got
            u += got
            if u < 0:
                usd = -u / (1 - fees.exchange)
                q_c -= usd / prev_s
                costs += usd * fees.exchange
                u = 0.0
                exchanges.append({"t": tk, "usd": usd, "share": usd / capital})
        min_ratio = min(min_ratio, ratio(pj, sj, fxj), ratio(hj, sj * hj / pj, fxj))
        value = u + q_c * sj + q_s * (p0 - pj) * fxj
        if j == len(marks) - 1:
            c_s, c_p = q_c * sj * fees.spot, q_s * pj * fxj * fees.perp
            costs += c_s + c_p
            value -= c_s + c_p
        series.append((int(tj), value))
    return {"series": series, "funding_usd": fund, "costs_usd": costs, "exchanges": exchanges, "min_margin_ratio": min_ratio, "n_coin": q_c, "n_short": q_s}


# ------------------------------------------------------------------ data
def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _d(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def _arr(path, lo, hi, keys=("c", "h")):
    with np.load(path) as z:
        t = z["t"].astype(np.int64)
        k = (t >= lo) & (t < hi)
        return {"t": t[k], **{c: z[c][k].astype(float) for c in keys}}


def load(spot: str, version: str, lo: str, hi: str, fx=None):
    """Marks of [lo, hi) and funding between them. version: 'USDT' or 'USDC'."""
    import funding_store as FS
    import n1_usdc as U
    import n2_quote as N2
    lo_ms, hi_ms = _ms(lo), _ms(hi)
    if version == "USDT":
        perp = _arr(FS.perp_path(spot), lo_ms, hi_ms)
        with np.load(FS.funding_path(spot)) as z:
            ft, fr = z["t"].astype(np.int64), z["rate"].astype(float)
    else:
        perp = _arr(TC.RUNTIME / "n4" / "perp" / f"{PAIRS[spot]}.npz", lo_ms, hi_ms)
        with np.load(U.STORE / f"{PAIRS[spot]}_funding.npz") as z:
            ft, fr = z["t"].astype(np.int64), z["rate"].astype(float)
    sp = _arr(TC.RUNTIME / "history" / f"{spot}_15m.npz", lo_ms, hi_ms, keys=("c",))
    tp, cp, hp = perp["t"], perp["c"], perp["h"]
    ts, cs = sp["t"], sp["c"]
    marks, skipped = [], 0
    for d in range(lo_ms, hi_ms, DAY):
        i, j = int(np.searchsorted(tp, d)), int(np.searchsorted(ts, d))
        if i < len(tp) and j < len(ts) and int(tp[i]) == d and int(ts[j]) == d:
            k = int(np.searchsorted(tp, d + DAY))
            high = float(np.nanmax(hp[i:k])) if k > i else float(cp[i])
            rate = N2.fx_at(fx, d + BAR) if version == "USDC" else 1.0
            marks.append((d + BAR, float(cp[i]), float(cs[j]), rate, high))
        else:
            skipped += 1
    ev = []
    missing = 0
    for t, r in zip(ft, fr):
        if marks[0][0] < t <= marks[-1][0] and np.isfinite(r):
            k = int(np.searchsorted(tp, int(t) - int(t) % BAR))
            if k < len(tp) and np.isfinite(cp[k]):
                ev.append((int(t), float(r), float(cp[k]), N2.fx_at(fx, int(t)) if version == "USDC" else 1.0))
            else:
                missing += 1
    return marks, ev, {"skipped_days": skipped, "funding_events": len(ev), "funding_without_bar": missing}


def book(version: str, fees: Fees, lo: str, hi: str, fx=None):
    eq, info = None, {}
    for spot in PAIRS:
        marks, ev, st = load(spot, version, lo, hi, fx=fx)
        r = simulate_pair(marks, ev, fees, capital=0.5)
        s = dict(r["series"])
        eq = s if eq is None else {t: eq[t] + s[t] for t in eq if t in s}
        info[spot] = {**st, "funding_usd": r["funding_usd"], "costs_usd": r["costs_usd"], "exchanges": r["exchanges"], "min_margin_ratio": r["min_margin_ratio"], "n_coin": r["n_coin"], "n_short": r["n_short"]}
    ts = sorted(eq)
    series = [(_d(ts[0]), (eq[ts[0]] - 1) * 100)] + [(_d(b), (eq[b] / eq[a] - 1) * 100) for a, b in zip(ts, ts[1:])]
    return series, info


def _years(series) -> dict:
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: float(np.mean(v)) for y, v in sorted(by.items())}


def evaluate_candidate(name: str, series, info, fc_series, years_needed: int) -> dict:
    ds, df = dict(series), dict(fc_series)
    days = sorted(set(ds) & set(df))
    diff = [ds[d] - df[d] for d in days]
    ci = {b: NC.ci(diff, alpha=ALPHA, block=b) for b in (5, 20, 60)}
    py = _years([(d, ds[d] - df[d]) for d in days])
    ex = [x for v in info.values() for x in v["exchanges"]]
    worst_ex = max((x["share"] for x in ex), default=0.0)
    margin = min(v["min_margin_ratio"] for v in info.values())
    vals = np.array([v for _, v in series])
    sd = float(np.std(diff, ddof=1))
    rho = float(np.corrcoef(diff[:-1], diff[1:])[0, 1]) if len(diff) > 3 else 0.0
    n, ne = NC.required_n(sd, DELTA, ALPHA, m=20, rho=max(0.0, rho))
    checks = {"lo_positive": bool(ci[20]["lo"] is not None and ci[20]["lo"] > 0), "years_positive": sum(1 for v in py.values() if v > 0) >= years_needed, "no_exchange_above_1pct": worst_ex <= 0.01, "margin_ge_3x": margin >= 3.0}
    return {"name": name, "n_days": len(days), "R": NC.ci(list(vals), alpha=ALPHA), "fc_on_same_days": float(np.mean([df[d] for d in days])), "paired": ci, "paired_by_year": py, "r_by_year": _years(series), "checks": checks, "passes": all(checks.values()),
            "exchanges": len(ex), "worst_exchange_share": worst_ex, "min_margin_ratio": margin, "sd_day": float(vals.std(ddof=1)), "sharpe_yr": float(vals.mean() / vals.std(ddof=1) * 365 ** 0.5), "worst_day": float(vals.min()), "max_dd_pct": _max_dd(vals),
            "pairs": {k: {a: b for a, b in v.items() if a != "exchanges"} for k, v in info.items()}, "power_forward": {"delta": DELTA, "alpha": ALPHA, "sd_paired_day": sd, "rho": max(0.0, rho), "n": n, "n_eff": ne, "m": 20}}


def _max_dd(pct) -> float:
    eq = np.cumprod(1 + np.asarray(pct) / 100)
    return float((eq / np.maximum.accumulate(eq) - 1).min() * 100)


def _fx_stress(cap_share: float = 0.98) -> dict:
    b = json.loads((EXP / "N1_USDC_BASIS.json").read_text(encoding="utf-8"))
    return {"note": "loss on the whole book = notional share x |1 - rate|, no recovery assumed", "by_year_p999": {y: b["by_year"][y]["p999"] * cap_share * 100 for y in b["by_year"]}, "max_pct": b["distribution"]["max"] * cap_share * 100}


def run() -> dict:
    import carry as C
    import experiment_fc as EF
    import n2_quote as N2
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    cf = C.fees(1)
    fees_t = Fees(cf.spot_taker + cf.slip, cf.perp_taker + cf.slip)
    fees_c = Fees(cf.spot_taker + cf.slip, CM_TAKER + cf.slip)
    mkt = EF.load_market()
    lo, hi = WINDOW
    fc = EF.evaluate(mkt, EF.passive_config(), cf, lo, hi, 0)
    s_t, i_t = book("USDT", fees_t, lo, hi)
    a = evaluate_candidate("G3b-USDT", s_t, i_t, fc["_series"], 2)
    a["slip_sensitivity"] = {}
    for label, slip in (("0bp", 0.0), ("20bp", 0.002)):
        st, _ = book("USDT", Fees(cf.spot_taker + slip, cf.perp_taker + slip), lo, hi)
        a["slip_sensitivity"][label] = float(np.mean([v for _, v in st]))
    fx = N2._fx()
    lo2, hi2 = WINDOW_USDC
    s_c, i_c = book("USDC", fees_c, lo2, hi2, fx=fx)
    b = evaluate_candidate("G3b-USDC", s_c, i_c, fc["_series"], 2)
    b["fx_stress"] = _fx_stress()
    g3a = json.loads((EXP / "G3A_TRAIN.json").read_text(encoding="utf-8"))
    out = {"spec": SPEC, "stage": "exploration", "window": {"USDT": WINDOW, "USDC": WINDOW_USDC}, "alpha_each": ALPHA, "delta": DELTA, "candidates": {"G3b-USDT": a, "G3b-USDC": b},
           "fc_passive": {"mean_pct": fc["mean_pct"], "margin_stops": fc["margin_stops"], "liquidations": fc["liquidations"], "r_by_year": fc["r_by_year"]},
           "reference_g3a": {"R": g3a["agg"]["mean_pct"], "paired": g3a["paired_vs_fc"]["mean_pct"], "note": "COIN-M, USD basis risk"}, "costs": "carry.fees(1) on spot and USDT perp, exactly as the benchmark; USDC taker 0.04% + the same slip",
           "not_modelled": ["USDC collateral transfer", "Binance auto-exchange terms beyond the design", "haircut changes", "margin mode switch on a live FC-DEMO", "interest on USDT balance"],
           "verdict": None, "traps_checked": [4, 5, 7, 9, 10], "traps_note": {"5": "no PBO: 2 fixed candidates"}, "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 2)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("n_days", "R", "fc_on_same_days", "paired", "paired_by_year", "r_by_year", "checks", "passes", "exchanges", "worst_exchange_share", "min_margin_ratio", "sharpe_yr", "worst_day", "max_dd_pct", "power_forward")} for k, v in out["candidates"].items()}, indent=0, default=str))
    print(json.dumps({"slip": a["slip_sensitivity"], "fx_stress": b["fx_stress"], "fc": out["fc_passive"], "pairs": a["pairs"]}, default=str))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        run()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
