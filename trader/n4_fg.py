"""N4 -- G1 funding harvest around the accrual, G2 premium convergence (n4-1106-spec.md).

  forecast   the funding rate from the premium bars closed by the decision; formula of the Binance page (spec §1)
  g1_trade   the side that should receive funding, close of T-15 min to close of T+15 min
  g2_trades  short perp + long spot after the premium index clears a threshold, exit below a third of it or after 8 h
  fetch      premium, funding with intervals, 15m perps and spot the stores lack, USDC versions (spec §3)
  explore    2023-2025: N4_G1_EXPLORE.json, N4_G2_EXPLORE.json; available events on 2020-07..2022-12 without prices or outcomes
  bh g1|g2   decisive 2020-07..2022-12: refuses without a ledger entry, one look

  (from trader/) ../pyembed/python.exe -u n4_fg.py fetch | explore | bh g1 | bh g2
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import archive_bh as A            # noqa: E402
import funding_store as FS        # noqa: E402
import n_common as NC             # noqa: E402
import trader_config as TC        # noqa: E402

SPEC = "n4-1106-spec.md"
STORE = TC.RUNTIME / "n4"
EXP = TC.RUNTIME / "experiments"
BAR = 900_000
HOUR = 3_600_000
DAY = 86_400_000
I8 = 1e-4
CLAMP = 5e-4
CS = (0.001, 0.003)
ALPHA = 0.05 / 4
DELTA_G1, DELTA_G2 = 3.0, 10.0
TAKER_UM, TAKER_CM, SPOT = 0.0005, 0.0004, 0.001
MAKER_UM = 0.0002
BNB_UM, BNB_CM, BNB_SPOT = 0.00045, 0.00036, 0.00075
SLIP = {"B1": 0.0, "B2": 0.6e-4, "B3": 3.3e-4, "B4": 11.4e-4}
G2_THR_TOP, G2_THR_REST = 0.004, 0.008
G2_HOLD = 32
LISTING_MS = 3 * DAY
TOP = 100
EXPLORE = ("2023-01-01", "2025-12-31")
EXPLORE_USDC = ("2024-01-01", "2025-12-31")
DECISIVE = ("2020-07-01", "2022-12-31")
LEDGER = HERE.parent / "docs" / "audit" / "alpha_ledger.json"
NOT_MODELLED = ["queue and impact above the calibrated 1 000 USDT", "funding cap in the forecast", "USDC/USDT over the 30 min hold of G1",
                "spot and perp transfers", "historic fee grid: the current grid is applied to every window"]


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _day(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def forecast(prem: dict, t_prev: int, t_next: int, decision_ms: int, i8: float = I8):
    t, c = prem["t"], prem["c"]
    i0 = int(np.searchsorted(t, t_prev, "left"))
    i1 = int(np.searchsorted(t, decision_ms - BAR, "right"))
    if i1 <= i0:
        return None, 0
    x = c[i0:i1].astype(float)
    k = np.isfinite(x)
    x = x[k]
    if not len(x):
        return None, 0
    n = round((t_next - t_prev) / HOUR)
    w = np.ones(len(x)) if n <= 1 else np.arange(1.0, len(x) + 1)
    p = float((x * w).sum() / w.sum())
    return (p + float(np.clip(i8 - p, -CLAMP, CLAMP))) * n / 8, len(x)


def g1_trade(bars: dict, t_fund: int, rate: float, fhat: float, cost_side: float):
    t, c = bars["t"], bars["c"]
    t0 = int(t_fund) - int(t_fund) % BAR
    eo, xo = t0 - 2 * BAR, t0
    e, x = int(np.searchsorted(t, eo)), int(np.searchsorted(t, xo))
    if e >= len(t) or x >= len(t) or int(t[e]) != eo or int(t[x]) != xo or not (c[e] > 0 and c[x] > 0):
        return None
    side = -1 if fhat > 0 else 1
    price = side * (float(c[x]) / float(c[e]) - 1) * 1e4
    fund = -side * float(rate) * 1e4
    cost = 2 * float(cost_side) * 1e4
    return {"net_bp": price + fund - cost, "price_bp": price, "funding_bp": fund, "cost_bp": cost, "side": side, "entry_ms": eo + BAR}


def placebo_times(t_fund: int, all_funding) -> list:
    ft = np.asarray(all_funding, dtype=np.int64)
    i = int(np.searchsorted(ft, t_fund))
    if i > 0 and round((int(ft[i]) - int(ft[i - 1])) / HOUR) <= 1:
        return []
    if i == 0 and len(ft) > 1 and round((int(ft[1]) - int(ft[0])) / HOUR) <= 1:
        return []
    out = []
    for tp in (t_fund - 2 * HOUR, t_fund + 2 * HOUR):
        lo, hi = tp - BAR, tp + BAR
        if not ((ft >= lo) & (ft <= hi)).any():
            out.append(tp)
    return out


def spot_of(perp: str, spot_syms):
    s = perp[:-4] + "USDT" if perp.endswith("USDC") else perp
    return FS.perp_to_spot(s, spot_syms)


def g2_trades(prem: dict, perp: dict, spot: dict, thr_of_bar, funding, cost_of_bar, excluded_until: int = 0, entry_lag: int = 0, last_ms=None):
    pt, pc = prem["t"], np.asarray(prem["c"], dtype=float)
    ft, fr = (np.asarray(funding["t"], dtype=np.int64), np.asarray(funding["rate"], dtype=float)) if funding is not None and len(funding["t"]) else (np.zeros(0, dtype=np.int64), np.zeros(0))
    tp, cp, ts, cs = perp["t"], perp["c"], spot["t"], spot["c"]
    dropped = {"listing": 0, "no_entry_bar": 0, "no_exit_bar": 0, "out_of_universe": 0}
    trades = []

    def at(tt, cc, ms):
        i = int(np.searchsorted(tt, ms))
        if i < len(tt) and int(tt[i]) == ms and cc[i] > 0 and np.isfinite(cc[i]):
            return float(cc[i])
        return None

    cand = np.flatnonzero(pc > G2_THR_TOP)                # no threshold of the rule is lower; the exact one is checked per bar
    free_from = -1
    for j in cand:
        i = int(j)
        ti = int(pt[i])
        if ti <= free_from:
            continue
        thr = thr_of_bar(ti)
        if thr is None:
            if pc[i] > G2_THR_TOP:
                dropped["out_of_universe"] += 1
            continue
        if not pc[i] > thr:
            continue
        if ti < excluded_until:
            dropped["listing"] += 1
            continue
        te = ti + entry_lag * BAR
        ep, es = at(tp, cp, te), at(ts, cs, te)
        if ep is None or es is None:
            dropped["no_entry_bar"] += 1
            continue
        last = te + G2_HOLD * BAR
        lo = int(np.searchsorted(pt, te, "right"))
        hi = int(np.searchsorted(pt, last, "right"))
        below = np.flatnonzero(pc[lo:hi] < thr / 3)
        tx, reason = (int(pt[lo + int(below[0])]), "converged") if len(below) else (last, "time")
        xp, xs = at(tp, cp, tx), at(ts, cs, tx)
        if xp is None or xs is None:
            perp_end = int(tp[-1]) if len(tp) else -1
            if last_ms is not None and perp_end < tx and perp_end < last_ms and perp_end > te and at(ts, cs, perp_end) is not None:
                tx, reason = perp_end, "delisted"
                xp, xs = at(tp, cp, tx), at(ts, cs, tx)
            if xp is None or xs is None:
                dropped["no_exit_bar"] += 1
                free_from = tx
                continue
        e_close, x_close = te + BAR, tx + BAR
        price = ((ep - xp) / ep + (xs - es) / es) * 1e4
        fund = float(fr[(ft > e_close) & (ft <= x_close)].sum()) * 1e4
        cost = float(cost_of_bar(te)) * 1e4
        trades.append({"net_bp": price + fund - cost, "price_bp": price, "funding_bp": fund, "cost_bp": cost, "entry_ms": e_close, "exit_ms": x_close,
                       "reason": reason, "bars": (tx - te) // BAR, "prem_entry_bp": float(pc[i]) * 1e4})
        free_from = tx
    return trades, dropped


# ------------------------------------------------------------------ data
def _usdc_admitted() -> list:
    return json.loads((EXP / "N1_LIQUIDITY.json").read_text(encoding="utf-8"))["admitted"]


def _perp_paths(sym: str) -> list:
    out = [A.PERP / f"{sym}_15m.npz", STORE / "perp" / f"{sym}.npz", TC.RUNTIME / "usdc_n3" / f"{sym}_bh_15m.npz", TC.RUNTIME / "usdc_n3" / f"{sym}_recent_15m.npz"]
    for sp, m in _MAP.items():
        if m.get("perp") == sym:
            out.append(FS.perp_path(sp))
    return out


_MAP = json.loads(FS.MAPPING.read_text(encoding="utf-8")) if FS.MAPPING.exists() else {}


def _load_cut(paths, lo_ms: int, hi_ms: int, keys=("c",)):
    """Bars in [lo_ms, hi_ms) from every file that has them; later data never reaches the analysis."""
    parts = []
    for p in paths:
        if Path(p).exists():
            with np.load(p) as z:
                t = z["t"].astype(np.int64)
                k = (t >= lo_ms) & (t < hi_ms)
                if k.any():
                    parts.append({"t": t[k], **{c: z[c][k].astype(float) for c in keys}})
    if not parts:
        return None
    t = np.concatenate([p["t"] for p in parts])
    o = np.argsort(t, kind="stable")
    t = t[o]
    keep = np.r_[t[1:] != t[:-1], True]
    return {"t": t[keep], **{c: np.concatenate([p[c] for p in parts])[o][keep] for c in keys}}


def _spot_paths(spot: str) -> list:
    return [A.STORE / f"{spot}_15m.npz", STORE / "spot" / f"{spot}.npz", TC.RUNTIME / "history" / f"{spot}_15m.npz"]


def _funding(sym: str, lo_ms: int, hi_ms: int):
    return _load_cut([STORE / "funding" / f"{sym}.npz"], lo_ms, hi_ms, keys=("rate", "hours"))


def _premium(sym: str, lo_ms: int, hi_ms: int):
    return _load_cut([STORE / "premium" / f"{sym}.npz"], lo_ms, hi_ms)


def _spot_names() -> set:
    return {i.rstrip("/").split("/")[-1] for i in A.list_all("data/spot/monthly/klines/", delimiter=True) if i.rstrip("/").endswith("USDT")}


def _ranks(vol, lo: str, hi: str) -> dict:
    ex = NC.pegged()
    out, d = {}, datetime.strptime(lo, "%Y-%m-%d")
    end = datetime.strptime(hi, "%Y-%m-%d")
    while d <= end:
        key = d.strftime("%Y-%m-%d")
        out[key] = vol.ranks(key, exclude=ex)
        d += timedelta(days=1)
    return out


def _universe(ranks: dict) -> list:
    return sorted({s for r in ranks.values() for s, v in r.items() if v <= TOP})


def fetch(log=print) -> dict:
    import n1_usdc as U
    vol = NC.load_volumes()
    bh = _universe(_ranks(vol, *DECISIVE))
    rc = _universe(_ranks(vol, *EXPLORE))
    usdc = _usdc_admitted()
    names = _spot_names()
    tot = {"bh": len(bh), "recent": len(rc), "usdc": len(usdc), "mb": None, "errors": []}
    kl = ("o", "h", "l", "c", "v")
    ex, outer = ThreadPoolExecutor(12), ThreadPoolExecutor(4)

    def has_bars(paths, first, last):
        lo, hi = _ms(NC.month_days(first)[0]), _ms(NC.month_days(last)[1]) + DAY
        return _load_cut(paths, lo, hi) is not None

    def premium_job(a):
        s, first, last = a
        return NC.fetch_months(f"data/futures/um/monthly/premiumIndexKlines/{s}/15m/", STORE / "premium" / f"{s}.npz", A.parse_klines, ("c",), ex, first=first, last=last)

    jobs = [(s, "2020-06", "2022-12") for s in bh] + [(s, "2022-12", "2025-12") for s in rc] + [(s, "2023-12", "2025-12") for s in usdc]
    for i, _ in enumerate(outer.map(premium_job, jobs)):
        if i % 200 == 0:
            log(f"premium {i}/{len(jobs)}")

    def funding_job(a):
        s, first, last = a
        return NC.fetch_months(f"data/futures/um/monthly/fundingRate/{s}/", STORE / "funding" / f"{s}.npz", U.parse_funding_full, ("rate", "hours"), ex, first=first, last=last)

    list(outer.map(funding_job, jobs))
    log("funding done")

    missing = [(s, "2020-06", "2022-12") for s in bh if not has_bars(_perp_paths(s), "2020-06", "2022-12")]
    missing += [(s, "2022-12", "2025-12") for s in rc if not has_bars(_perp_paths(s), "2022-12", "2025-12")]
    missing += [(s, "2023-12", "2025-12") for s in usdc]

    def perp_job(a):
        s, first, last = a
        return NC.fetch_months(f"data/futures/um/monthly/klines/{s}/15m/", STORE / "perp" / f"{s}.npz", A.parse_klines, kl, ex, first=first, last=last)

    list(outer.map(perp_job, missing))
    tot["perp_fetched"] = len(missing)
    log(f"perps done {len(missing)}")

    miss_spot = {}
    for s, first, last in jobs:
        m = spot_of(s, names)
        if m and not has_bars(_spot_paths(m[0]), first, last):
            a = miss_spot.get(m[0], (first, last))
            miss_spot[m[0]] = (min(first, a[0]), max(last, a[1]))

    def spot_job(a):
        s, (first, last) = a
        return NC.fetch_months(f"data/spot/monthly/klines/{s}/15m/", STORE / "spot" / f"{s}.npz", A.parse_klines, kl, ex, first=first, last=last)

    list(outer.map(spot_job, list(miss_spot.items())))
    tot["spot_fetched"] = len(miss_spot)
    ex.shutdown()
    outer.shutdown()
    tot["mb"] = round(sum(p.stat().st_size for p in STORE.rglob("*.npz")) / 1e6, 1)
    log(ascii(tot))
    return tot


# ------------------------------------------------------------------ G1 run
def _bucket(sym: str, rank) -> str:
    import cost_calibration as CC
    base = sym[:-4] + "USDT" if sym.endswith("USDC") else sym
    return CC.bucket(base, rank)


def g1_events(symbols, ranks: dict, lo: str, hi: str, taker: float, want_prices: bool = True, rank_of=None) -> dict:
    """Every accrual of the universe in [lo, hi]: forecast, and with want_prices the trade, the placebo and the maker entry."""
    lo_ms, hi_ms = _ms(lo), _ms(hi) + DAY
    rows = {c: [] for c in CS}
    placebo = {c: [] for c in CS}
    maker = {c: [] for c in CS}
    acc = []
    stats = {"no_exact_bar": 0, "fallback": 0, "events": 0}
    rank_of = rank_of or (lambda s: s)
    for s in symbols:
        f = _funding(s, lo_ms, hi_ms)
        if f is None or len(f["t"]) < 3:
            continue
        p = _premium(s, lo_ms, hi_ms)
        if p is None:
            continue
        ft = (f["t"] - f["t"] % BAR).astype(np.int64)
        bars = None
        seen = set()
        for i in range(1, len(ft)):
            T, Tp = int(ft[i]), int(ft[i - 1])
            if not (lo_ms <= T < hi_ms):
                continue
            if T == Tp or T - Tp > 8 * HOUR + BAR:                 # a duplicate, or an accrual missing from the file
                stats["bad_interval"] = stats.get("bad_interval", 0) + 1
                continue
            day = _day(T)
            rank = ranks.get(day, {}).get(rank_of(s))
            if rank is None or rank > TOP:
                continue
            stats["events"] += 1
            fhat, n = forecast(p, Tp, T, T - BAR)
            fb = False
            if fhat is None:
                fhat, fb = float(f["rate"][i - 1]), True
                stats["fallback"] += 1
            rate = float(f["rate"][i])
            acc.append((fhat, rate, round((T - Tp) / HOUR), day[:4]))
            if not want_prices:
                for c in CS:
                    if abs(fhat) >= c:
                        rows[c].append({"day": day})
                continue
            if bars is None and abs(fhat) >= min(CS):
                bars = _load_cut(_perp_paths(s), lo_ms - DAY, hi_ms + DAY, keys=("c", "h", "l"))
            bucket = _bucket(s, rank)
            side_cost = taker + SLIP[bucket]
            for c in CS:
                if abs(fhat) >= c:
                    tr = g1_trade(bars, T, rate, fhat, side_cost) if bars is not None else None
                    if tr is None:
                        stats["no_exact_bar"] += 1
                        continue
                    rows[c].append({**tr, "sym": s, "day": _day(tr["entry_ms"]), "bucket": bucket, "rank": rank, "fhat": fhat, "rate": rate, "interval": round((T - Tp) / HOUR), "fallback": fb})
            # placebo: the condition is computed at T' - 15 min on the premium of the interval running then (spec §2)
            for tq in placebo_times(T, ft):
                if tq in seen:                                     # T + 2 h and the next T - 2 h coincide on a 4 h interval
                    continue
                seen.add(tq)
                j = int(np.searchsorted(ft, tq, "right"))
                if j == 0 or j >= len(ft):
                    continue
                fh2, _ = forecast(p, int(ft[j - 1]), int(ft[j]), tq - BAR)
                if fh2 is None:
                    continue
                if abs(fh2) >= min(CS) and bars is None:
                    bars = _load_cut(_perp_paths(s), lo_ms - DAY, hi_ms + DAY, keys=("c", "h", "l"))
                for c in CS:
                    if abs(fh2) >= c and bars is not None:
                        tr = g1_trade(bars, tq, 0.0, fh2, side_cost)
                        if tr is not None:
                            placebo[c].append({**tr, "day": _day(tr["entry_ms"]), "sym": s, "bucket": bucket})
            # maker entry: decided at T - 30 min, limit at that close, filled through it in the next bar
            fm, _ = forecast(p, Tp, T, T - 2 * BAR)
            if fm is not None and abs(fm) >= min(CS):
                if bars is None:
                    bars = _load_cut(_perp_paths(s), lo_ms - DAY, hi_ms + DAY, keys=("c", "h", "l"))
                if bars is not None:
                    tt = bars["t"]
                    a = int(np.searchsorted(tt, T - 3 * BAR))
                    if a + 1 < len(tt) and int(tt[a]) == T - 3 * BAR and int(tt[a + 1]) == T - 2 * BAR:
                        limit = float(bars["c"][a])
                        side = -1 if fm > 0 else 1
                        hit = bars["l"][a + 1] < limit if side == 1 else bars["h"][a + 1] > limit
                        x = int(np.searchsorted(tt, T))
                        if hit and x < len(tt) and int(tt[x]) == T and limit > 0:
                            price = side * (float(bars["c"][x]) / limit - 1) * 1e4
                            fund = -side * rate * 1e4
                            cost = ((0.0 if taker == TAKER_CM else MAKER_UM) + taker + SLIP[bucket]) * 1e4
                            for c in CS:
                                if abs(fm) >= c:
                                    maker[c].append({"net_bp": price + fund - cost, "day": _day(T), "sym": s})
    return {"rows": rows, "placebo": placebo, "maker": maker, "accuracy": acc, "stats": stats}


def _accuracy(acc: list) -> dict:
    if not acc:
        return {}

    def one(rows):
        f = np.array([a[0] for a in rows])
        r = np.array([a[1] for a in rows])
        return {"n": len(rows), "corr": float(np.corrcoef(f, r)[0, 1]) if len(rows) > 2 and f.std() > 0 and r.std() > 0 else None, "sign_match": float(np.mean(np.sign(f) == np.sign(r))), "mean_abs_err_bp": float(np.mean(np.abs(f - r)) * 1e4)}
    out = {"all": one(acc), "by_interval": {}, "by_year": {}}
    for n in sorted({a[2] for a in acc}):
        out["by_interval"][str(n)] = one([a for a in acc if a[2] == n])
    for y in sorted({a[3] for a in acc}):
        out["by_year"][y] = one([a for a in acc if a[3] == y])
    big = [a for a in acc if abs(a[0]) >= 0.001]
    if big:
        out["at_or_above_0.10pct"] = one(big)
    return out


def _summary(rows: list, delta: float, alpha: float = ALPHA) -> dict:
    if not rows:
        return {"n": 0}
    v = np.array([x["net_bp"] for x in rows])
    ci = NC.cluster_ci(list(v), [x["day"] for x in rows], alpha=alpha)
    by_y, by_b, by_s, by_d = {}, {}, {}, {}
    for x in rows:
        by_y.setdefault(x["day"][:4], []).append(x["net_bp"])
        by_b.setdefault(x.get("bucket", "?"), []).append(x["net_bp"])
        by_s.setdefault(x.get("sym", "?"), []).append(x["net_bp"])
        by_d[x["day"]] = by_d.get(x["day"], 0.0) + x["net_bp"]
    days = sorted(by_d, key=lambda d: -by_d[d])
    top5 = set(days[:5])
    rest = [x["net_bp"] for x in rows if x["day"] not in top5]
    syms = sorted(by_s, key=lambda s: -sum(by_s[s]))
    m = len(rows) / len(by_d)
    sigma = float(v.std(ddof=1)) if len(v) > 1 else None
    rho = NC.icc(list(v), [x["day"] for x in rows]) if len(by_d) > 1 else 0.0
    n, ne = NC.required_n(sigma, delta, alpha, m=m, rho=max(0.0, rho)) if sigma else (None, None)
    return {"n": len(rows), "ci": ci, "sigma_bp": sigma, "median_bp": float(np.median(v)), "pos_share": float((v > 0).mean()), "n_days": len(by_d), "m_per_day": m, "rho_day": float(rho),
            "power": {"delta_bp": delta, "alpha": alpha, "n": n, "n_eff": ne},
            "by_year": {k: {"n": len(a), "mean": float(np.mean(a))} for k, a in sorted(by_y.items())},
            "by_bucket": {k: {"n": len(a), "mean": float(np.mean(a))} for k, a in sorted(by_b.items())},
            "without_5_best_days": float(np.mean(rest)) if rest else None, "top_symbols": [(ascii(s), len(by_s[s]), round(sum(by_s[s]), 1)) for s in syms[:5]], "bottom_symbols": [(ascii(s), len(by_s[s]), round(sum(by_s[s]), 1)) for s in syms[-5:]],
            "means": {k: float(np.mean([x[k] for x in rows if k in x])) for k in ("price_bp", "funding_bp", "cost_bp", "bars", "prem_entry_bp", "fx_bp") if any(k in x for x in rows)}}


def _resensitive_g1(rows: list, taker_new: float, taker_old: float, slip_mult: float) -> float:
    if not rows:
        return float("nan")
    out = []
    for x in rows:
        s = SLIP[x["bucket"]]
        out.append(x["net_bp"] + 2 * ((taker_old + s) - (taker_new + s * slip_mult)) * 1e4)
    return float(np.mean(out))


def _resensitive_g2(rows: list, slip_mult: float, bnb: bool, perp_taker: float) -> float:
    if not rows:
        return float("nan")
    out = []
    for x in rows:
        s = SLIP[x["bucket"]]
        base = (SPOT + perp_taker + 2 * s) * 2
        new = ((BNB_SPOT if bnb else SPOT) + (BNB_UM if bnb and perp_taker == TAKER_UM else BNB_CM if bnb else perp_taker) + 2 * s * slip_mult) * 2
        out.append(x["net_bp"] + (base - new) * 1e4)
    return float(np.mean(out))


def _g1_report(ev: dict, taker: float) -> dict:
    out = {}
    for c in CS:
        rows = ev["rows"][c]
        s = _summary(rows, DELTA_G1)
        pl = ev["placebo"][c]
        s["placebo"] = {"n": len(pl), "mean_net_bp": float(np.mean([x["net_bp"] for x in pl])) if pl else None, "mean_price_bp": float(np.mean([x["price_bp"] for x in pl])) if pl else None}
        if rows and pl:
            s["effect_of_accrual_bp"] = s["ci"]["mean"] - s["placebo"]["mean_net_bp"]
        mk = ev["maker"][c]
        s["maker_entry"] = {"n": len(mk), "mean_net_bp": float(np.mean([x["net_bp"] for x in mk])) if mk else None}
        s["sensitivity"] = {"bnb": _resensitive_g1(rows, BNB_UM if taker == TAKER_UM else BNB_CM, taker, 1.0), "slip_0": _resensitive_g1(rows, taker, taker, 0.0), "slip_2.5": _resensitive_g1(rows, taker, taker, 2.5)}
        out[str(c)] = s
    return out


# ------------------------------------------------------------------ G2 run
def _g2_setup(sym: str, ranks: dict, vol, names: set, lo_ms: int, hi_ms: int, rank_of=None):
    m = spot_of(sym, names)
    if not m:
        return None, "no_spot_pair"
    spot_sym, mult = m
    prem = _premium(sym, lo_ms, hi_ms)
    if prem is None or not len(prem["t"]):
        return None, "no_premium"
    perp = _load_cut(_perp_paths(sym), lo_ms, hi_ms)
    spot = _load_cut(_spot_paths(spot_sym), lo_ms, hi_ms)
    if perp is None or spot is None:
        return None, "no_bars"
    import funding_premium as FP
    if not FP.pair_ok(perp, spot, mult):
        return None, "pair_not_ok"
    return {"prem": prem, "perp": perp, "spot": spot, "mult": mult, "spot_sym": spot_sym}, None


def g2_events(symbols, ranks: dict, vol, lo: str, hi: str, perp_taker: float, lag: int = 0, rank_of=None, seed: int = 29, want_placebo: bool = True, fx=None) -> dict:
    lo_ms, hi_ms = _ms(lo), _ms(hi) + DAY
    names = _SPOTS()
    rows, pl, tally = [], [], {"listing": 0, "no_entry_bar": 0, "no_exit_bar": 0, "out_of_universe": 0}
    skipped = {}
    rank_of = rank_of or (lambda s: s)
    rng = np.random.default_rng(seed)
    for s in symbols:
        x, why = _g2_setup(s, ranks, vol, names, lo_ms, hi_ms)
        if x is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        fund = _funding(s, lo_ms, hi_ms)
        fs = {"t": (fund["t"] - fund["t"] % BAR).astype(np.int64), "rate": fund["rate"]} if fund is not None else None
        key = rank_of(s)
        # launch of this perp and first spot bar of the whole store, not of the cut window
        first_p = _ms(vol.first[s]) if s in vol.first else _first_bar(_perp_paths(s))
        excl = max(first_p or 0, _first_bar(_spot_paths(x["spot_sym"])) or 0) + LISTING_MS

        def thr(ms, k=key):
            r = ranks.get(_day(ms), {}).get(k)
            if r is None or r > TOP:
                return None
            return G2_THR_TOP if r <= 20 else G2_THR_REST

        def bucket_of(ms, k=key, sym=s):
            r = ranks.get(_day(ms), {}).get(k, 100)
            return _bucket(sym, r)

        def cost(ms):
            return (SPOT + perp_taker + 2 * SLIP[bucket_of(ms)]) * 2

        tr, dropped = g2_trades(x["prem"], x["perp"], x["spot"], thr, fs, cost, excluded_until=excl, entry_lag=lag, last_ms=hi_ms - BAR)
        for k, v in dropped.items():
            tally[k] += v
        for t in tr:
            t.update({"sym": s, "day": _day(t["entry_ms"]), "bucket": bucket_of(t["entry_ms"] - BAR), "first_week": t["entry_ms"] - excl + LISTING_MS < 7 * DAY})
            if fx is not None:
                import n2_quote as N2
                a, b = N2.fx_at(fx, t["entry_ms"]), N2.fx_at(fx, t["exit_ms"])
                t["fx_bp"] = (b / a - 1) * 1e4
                t["net_bp"] += t["fx_bp"]
        rows += tr
        if want_placebo and tr:
            prem = x["prem"]
            pv = prem["c"]
            pool = {}
            for th in (G2_THR_TOP, G2_THR_REST):
                pool[th] = prem["t"][(pv >= 0) & (pv <= th / 2)]
            for t in tr:
                th = thr(t["entry_ms"] - BAR) or G2_THR_REST
                cand = pool[th]
                cand = cand[(cand >= excl)]
                if not len(cand):
                    continue
                t0 = int(rng.choice(cand))
                k = (prem["t"] >= t0) & (prem["t"] <= t0 + G2_HOLD * BAR)
                forced = {"t": prem["t"][k], "c": np.where(prem["t"][k] == t0, max(th, G2_THR_TOP) * 1.0001, pv[k])}
                q, _ = g2_trades(forced, x["perp"], x["spot"], lambda ms, h=th: h, fs, cost, excluded_until=0, entry_lag=0, last_ms=hi_ms - BAR)
                if q:
                    q0 = q[0]
                    if q0["entry_ms"] == t0 + BAR:
                        pl.append({**q0, "day": _day(q0["entry_ms"]), "sym": s, "bucket": bucket_of(t0)})
    return {"rows": rows, "placebo": pl, "dropped": tally, "skipped": skipped}


def _first_bar(paths):
    first = None
    for p in paths:
        if Path(p).exists():
            with np.load(p) as z:
                t = z["t"]
                if len(t):
                    first = int(t[0]) if first is None else min(first, int(t[0]))
    return first


_SPOT_CACHE = {}


def _SPOTS() -> set:
    if "x" not in _SPOT_CACHE:
        _SPOT_CACHE["x"] = _spot_names()
    return _SPOT_CACHE["x"]


def _g2_report(ev: dict, perp_taker: float) -> dict:
    rows = ev["rows"]
    s = _summary(rows, DELTA_G2)
    pl = ev["placebo"]
    s["placebo"] = {"n": len(pl), "mean_net_bp": float(np.mean([x["net_bp"] for x in pl])) if pl else None, "mean_price_bp": float(np.mean([x["price_bp"] for x in pl])) if pl else None}
    s["dropped"] = ev["dropped"]
    s["skipped_symbols"] = ev["skipped"]
    s["reasons"] = {r: {"n": sum(1 for x in rows if x["reason"] == r), "mean": float(np.mean([x["net_bp"] for x in rows if x["reason"] == r]))} for r in {x["reason"] for x in rows}}
    s["first_week_share"] = float(np.mean([x["first_week"] for x in rows])) if rows else None
    s["mean_excl_first_week"] = float(np.mean([x["net_bp"] for x in rows if not x["first_week"]])) if rows and any(not x["first_week"] for x in rows) else None
    s["sensitivity"] = {"bnb": _resensitive_g2(rows, 1.0, True, perp_taker), "slip_0": _resensitive_g2(rows, 0.0, False, perp_taker), "slip_2.5": _resensitive_g2(rows, 2.5, False, perp_taker)}
    return s


def _available_g2(symbols, ranks: dict, vol, lo: str, hi: str) -> int:
    """Signals on the window from the premium alone: no price, no outcome (an upper bound: spot and pair checks need prices)."""
    lo_ms, hi_ms = _ms(lo), _ms(hi) + DAY
    n = 0
    for s in symbols:
        p = _premium(s, lo_ms, hi_ms)
        if p is None or not len(p["t"]):
            continue
        first = vol.first.get(s)
        excl = (_ms(first) if first else 0) + LISTING_MS

        def thr(ms, k=s):
            r = ranks.get(_day(ms), {}).get(k)
            if r is None or r > TOP:
                return None
            return G2_THR_TOP if r <= 20 else G2_THR_REST

        fake = {"t": p["t"], "c": np.ones(len(p["t"]))}
        tr, _ = g2_trades(p, fake, fake, thr, None, lambda _: 0.0, excluded_until=excl, last_ms=hi_ms - BAR)
        n += len(tr)
    return n


def explore() -> dict:
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    vol = NC.load_volumes()
    ranks = _ranks(vol, *EXPLORE)
    uni = _universe(ranks)
    usdc = _usdc_admitted()
    base = lambda s: s[:-4] + "USDT"
    ev = g1_events(uni, ranks, *EXPLORE, TAKER_UM)
    print("g1 events", ev["stats"], {c: len(v) for c, v in ev["rows"].items()}, round(time.time() - t0), flush=True)
    rep = _g1_report(ev, TAKER_UM)
    ev_c = g1_events(usdc, ranks, *EXPLORE_USDC, TAKER_CM, rank_of=base)
    rep_c = _g1_report(ev_c, TAKER_CM)
    bh_ranks = _ranks(vol, *DECISIVE)
    bh_uni = _universe(bh_ranks)
    ev_b = g1_events(bh_uni, bh_ranks, *DECISIVE, TAKER_UM, want_prices=False)
    avail = {str(c): len(v) for c, v in ev_b["rows"].items()}
    for c in CS:
        p = rep[str(c)]["power"]
        p["available_decisive"] = avail[str(c)]
        p["window_opens"] = bool(p["n_eff"] is not None and avail[str(c)] >= p["n_eff"])
    out = {"spec": SPEC, "stage": "exploration", "window": EXPLORE, "candidates": rep, "usdc_reference": {"window": EXPLORE_USDC, "candidates": rep_c, "accuracy": _accuracy(ev_c["accuracy"])},
           "forecast_accuracy": _accuracy(ev["accuracy"]), "events_stats": ev["stats"], "universe": len(uni), "fee_grid": "current grid on the whole window",
           "not_modelled": NOT_MODELLED, "verdict": None, "traps_checked": [1, 2, 4, 5, 6, 7, 10], "traps_note": {"5": "no PBO: 2 fixed candidates", "6": "kill rule is about the signal only"}, "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 1)}
    (EXP / "N4_G1_EXPLORE.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: {x: rep[k][x] for x in ("n", "ci", "sigma_bp", "m_per_day", "rho_day", "power", "by_year", "by_bucket", "without_5_best_days", "means", "placebo", "maker_entry", "sensitivity", "top_symbols")} for k in rep}, indent=0, default=str), flush=True)
    print("accuracy", json.dumps(out["forecast_accuracy"], default=str), flush=True)
    print("usdc", json.dumps({k: {x: rep_c[k].get(x) for x in ("n", "ci", "by_year")} for k in rep_c}, default=str), flush=True)

    ev2 = g2_events(uni, ranks, vol, *EXPLORE, TAKER_UM)
    g2 = _g2_report(ev2, TAKER_UM)
    ev2_lag = g2_events(uni, ranks, vol, *EXPLORE, TAKER_UM, lag=1, want_placebo=False)
    g2["next_bar_entry"] = {"n": len(ev2_lag["rows"]), "ci": NC.cluster_ci([x["net_bp"] for x in ev2_lag["rows"]], [x["day"] for x in ev2_lag["rows"]], alpha=ALPHA) if ev2_lag["rows"] else None}
    fx = None
    import n2_quote as N2
    fx = N2._fx()
    ev2c = g2_events(usdc, ranks, vol, *EXPLORE_USDC, TAKER_CM, rank_of=base, fx=fx)
    g2c = _g2_report(ev2c, TAKER_CM)
    avail2 = _available_g2(bh_uni, bh_ranks, vol, *DECISIVE)
    g2["power"]["available_decisive_upper_bound"] = avail2
    g2["power"]["window_opens"] = bool(g2["power"]["n_eff"] is not None and avail2 >= g2["power"]["n_eff"])
    out2 = {"spec": SPEC, "stage": "exploration", "window": EXPLORE, "candidates": {"USDT": g2, "USDC": {**g2c, "window": EXPLORE_USDC, "decisive": "no USDC perp before 2024-01: not testable"}}, "universe": len(uni), "fee_grid": "current grid on the whole window", "not_modelled": NOT_MODELLED,
            "verdict": None, "traps_checked": [1, 2, 3, 4, 5, 6, 7, 9, 10], "traps_note": {"5": "no PBO: 2 fixed candidates", "3": "listing filter 72 h, first week share reported", "9": "USDC version carries the USDC/USDT change"}, "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 1)}
    (EXP / "N4_G2_EXPLORE.json").write_text(json.dumps(out2, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    keys = ("n", "ci", "sigma_bp", "m_per_day", "rho_day", "power", "by_year", "by_bucket", "without_5_best_days", "means", "placebo", "reasons", "first_week_share", "mean_excl_first_week", "sensitivity", "dropped", "skipped_symbols", "next_bar_entry", "top_symbols", "bottom_symbols")
    print(json.dumps({"USDT": {x: g2.get(x) for x in keys}, "USDC": {x: g2c.get(x) for x in keys}}, indent=0, default=str), flush=True)
    return out


def _word(ci: dict, delta: float) -> str:
    if ci["lo"] is not None and ci["lo"] > 0:
        return "accepted"
    if ci["hi"] is not None and ci["hi"] < delta:
        return "rejected: upper bound below delta"
    return "rejected: lower bound <= 0 (kill rule)"


def _open(which: str) -> None:
    import provenance as PV
    marker = EXP / f"N4_{which}_BH_OPENED.json"
    if marker.exists():
        raise RuntimeError(f"{which} decisive window already opened")
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    if not any(e.get("id") == f"N4-{which}-BH" for e in led["looks"]):
        raise RuntimeError(f"no ledger entry N4-{which}-BH")
    marker.write_text(json.dumps({"opened": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "head": PV.code_state()["head"]}), encoding="utf-8")


def bh(which: str) -> dict:
    import provenance as PV
    PV.require_clean()
    which = which.upper()
    ex_name = {"G1": "N4_G1_EXPLORE.json", "G2": "N4_G2_EXPLORE.json"}[which]
    if not (EXP / ex_name).exists():
        raise RuntimeError("explore first")
    _open(which)
    t0 = time.time()
    vol = NC.load_volumes()
    ranks = _ranks(vol, *DECISIVE)
    uni = _universe(ranks)
    if which == "G1":
        ev = g1_events(uni, ranks, *DECISIVE, TAKER_UM)
        rep = _g1_report(ev, TAKER_UM)
        for c in rep:
            rep[c]["verdict"] = _word(rep[c]["ci"], DELTA_G1) if rep[c]["n"] else "no events"
        out = {"spec": SPEC, "stage": "decisive", "window": DECISIVE, "candidates": rep, "forecast_accuracy": _accuracy(ev["accuracy"]), "events_stats": ev["stats"], "provenance": PV.block()}
    else:
        ev = g2_events(uni, ranks, vol, *DECISIVE, TAKER_UM)
        rep = _g2_report(ev, TAKER_UM)
        rep["verdict"] = _word(rep["ci"], DELTA_G2) if rep["n"] else "no events"
        lag = g2_events(uni, ranks, vol, *DECISIVE, TAKER_UM, lag=1, want_placebo=False)
        rep["next_bar_entry"] = {"n": len(lag["rows"]), "ci": NC.cluster_ci([x["net_bp"] for x in lag["rows"]], [x["day"] for x in lag["rows"]], alpha=ALPHA) if lag["rows"] else None}
        out = {"spec": SPEC, "stage": "decisive", "window": DECISIVE, "candidates": {"USDT": rep}, "provenance": PV.block()}
    out.update({"fee_grid": "current grid on the whole window", "not_modelled": NOT_MODELLED, "traps_checked": [1, 2, 3, 4, 5, 6, 7, 9, 10], "minutes": round((time.time() - t0) / 60, 1)})
    (EXP / f"N4_{which}_BH.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["candidates"], indent=0, default=str)[:6000])
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fetch"]:
        fetch(log=lambda s: print(s, flush=True))
    elif argv[:1] == ["explore"]:
        explore()
    elif argv[:1] == ["bh"]:
        bh(argv[1])
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
