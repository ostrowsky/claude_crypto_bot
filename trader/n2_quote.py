"""N2 -- passive quote on a USDC perp with zero maker fee, immediate taker hedge on the USDT perp (n2-1106-spec.md).

  schedule   decision times of the quote from the USDT tape: band 0.5 bp, refresh 1 s, at most 5 changes a second
  simulate   one pair, one latency, one theta: fills strictly through the limit, two wallets, every flow accounted
  pilot      2025-12-15..17: fills, sigma, rho; STOP if the pool earns <= 0 at L = 250/250 and theta = 3 bp
  explore    2025-12-10..31, the exploration window of the design
  fetch      aggTrades of the window (archive, no operator decision below 2 GB for the experiment)

  (from trader/) ../pyembed/python.exe -u n2_quote.py fetch pilot|explore   then   pilot | explore
"""
from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import archive_bh as A            # noqa: E402
import n_common as NC             # noqa: E402
import trader_config as TC        # noqa: E402

SPEC = "n2-1106-spec.md"
STORE = TC.RUNTIME / "tape_n2"
EXP = TC.RUNTIME / "experiments"
USDC = TC.RUNTIME / "usdc"
PAIRS = {"BTC": ("BTCUSDC", "BTCUSDT", 0.0), "ETH": ("ETHUSDC", "ETHUSDT", 0.0), "SOL": ("SOLUSDC", "SOLUSDT", 0.59e-4)}
THETAS = (1e-4, 3e-4)
LAGS = (250, 1000)
TAKER_USDT = 0.0005
TAKER_USDC = 0.0004
BNB_USDT = 0.00045
BNB_USDC = 0.00036
ALPHA = 0.025
DELTA = 1.0
WINDOWS = {"pilot": ("2025-12-15", "2025-12-17"), "explore": ("2025-12-10", "2025-12-31")}
MARKOUTS = (1000, 5000, 60000)
DAY = 86_400_000
MIN_GAP = 200


@dataclass(frozen=True)
class Cfg:
    theta: float
    l_q: int
    l_h: int
    h: float
    tick: float
    q: float = 1000.0
    g: float = 20000.0
    maker: float = 0.0
    taker_usdt: float = TAKER_USDT
    taker_usdc: float = TAKER_USDC
    slip: float = 0.0
    strict: bool = True
    slip_usdc: float | None = None


def parse_agg(text) -> dict:
    src = io.StringIO(text) if isinstance(text, str) else text
    df = pd.read_csv(src, header=None, usecols=[0, 1, 2, 5, 6], names=["a", "p", "q", "t", "m"], dtype=str, keep_default_na=False)
    df = df[df["a"].str.fullmatch(r"\d+")]
    t = df["t"].astype("int64").to_numpy()
    t = np.where(t > 10 ** 14, t // 1000, t)
    a = df["a"].astype("int64").to_numpy()
    p = df["p"].astype(float).to_numpy()
    q = df["q"].astype(float).to_numpy()
    m = df["m"].str.lower().eq("true").to_numpy()
    o = np.lexsort((a, t))
    t, a, p, q, m = t[o], a[o], p[o], q[o], m[o]
    keep = np.ones(len(a), dtype=bool)
    if len(a) > 1:
        _, first = np.unique(a, return_index=True)
        keep[:] = False
        keep[first] = True
    return {"t": t[keep], "p": p[keep], "q": q[keep], "m": m[keep]}


def schedule(t_u, p_u, band: float = 0.5e-4, refresh: int = 1000, min_gap: int = MIN_GAP):
    t_u, p_u = np.asarray(t_u, dtype=np.int64), np.asarray(p_u, dtype=float)
    end = int(t_u[-1])
    d, F = [int(t_u[0])], [float(p_u[0])]
    while True:
        cur, fq = d[-1], F[-1]
        i0 = int(np.searchsorted(t_u, cur + min_gap, "left"))
        i1 = int(np.searchsorted(t_u, cur + refresh, "right"))
        hit = None
        # short tapes in the window are the rule: look in small steps so a busy second stays cheap
        for a in range(i0, i1, 64):
            seg = p_u[a:min(a + 64, i1)]
            w = np.flatnonzero(np.abs(seg - fq) > band * fq)
            if len(w):
                hit = a + int(w[0])
                break
        if hit is not None:
            nd, nf = int(t_u[hit]), float(p_u[hit])
        else:
            nd = cur + refresh
            if nd > end:
                break
            nf = float(p_u[int(np.searchsorted(t_u, nd, "right")) - 1])
        d.append(nd)
        F.append(nf)
    return np.array(d, dtype=np.int64), np.array(F, dtype=float)


def _round(x, tick):
    return np.round(x, max(0, int(np.ceil(-np.log10(tick)))) + 3)


def quotes(F, cfg: Cfg):
    F = np.asarray(F, dtype=float)
    w = cfg.h + cfg.theta
    bid = _round(np.floor(F * (1 - w) / cfg.tick + 1e-9) * cfg.tick, cfg.tick)
    ask = _round(np.ceil(F * (1 + w) / cfg.tick - 1e-9) * cfg.tick, cfg.tick)
    return bid, ask


def fx_at(fx, ms: int) -> float:
    if fx is None:
        return 1.0
    i = int(np.searchsorted(fx["t"], ms - 60_000, "right")) - 1
    return float(fx["c"][max(0, i)])


def _day(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def simulate(usdc: dict, usdt: dict, cfg: Cfg, fx=None, funding_c=None, funding_u=None, sched=None) -> dict:
    tc, pc, qc, mc = usdc["t"], usdc["p"], usdc["q"], usdc["m"]
    tu, pu = usdt["t"], usdt["p"]
    d, F = sched if sched is not None else schedule(tu, pu)
    e = d + cfg.l_q
    bid_s, ask_s = quotes(F, cfg)
    k = np.searchsorted(e, tc, "right") - 1
    live = k >= 0
    kk = np.maximum(k, 0)
    cross_b = pc < bid_s[kk] if cfg.strict else pc <= bid_s[kk]
    cross_a = pc > ask_s[kk] if cfg.strict else pc >= ask_s[kk]
    cand = {1: np.flatnonzero(live & mc & cross_b), -1: np.flatnonzero(live & ~mc & cross_a)}

    off_until = {1: -1, -1: -1}
    disabled = {1: False, -1: False}
    override = {1: None, -1: None}                # (start index time, end time, price, F)
    last_change = {1: -1, -1: -1}
    W_c = W_c_val = W_t = 0.0
    Q = pos_usd = 0.0
    spread = fees = slip_cost = fund = 0.0
    fills = []
    ev = []
    for name, f in (("c", funding_c), ("u", funding_u)):
        if f is not None:
            ev += [(int(x), name, float(r)) for x, r in zip(f["t"], f["rate"])]
    ev.sort()
    ev_i = 0
    days = {}
    end_t = max(int(tc[-1]) if len(tc) else 0, 0)
    day_next = (int(tc[0]) // DAY + 1) * DAY if len(tc) else None
    eq_prev = 0.0

    def p_last(ts, ps, at):
        i = int(np.searchsorted(ts, at, "right")) - 1
        return float(ps[max(0, i)])

    def equity(at):
        rate = fx_at(fx, at)
        return (W_c + Q * p_last(tc, pc, at)) * rate + W_t - Q * p_last(tu, pu, at)

    def close_day(at):
        nonlocal eq_prev
        eq = equity(at)
        days.setdefault(_day(at - 1), {"pnl": 0.0, "notional": 0.0, "n": 0})["pnl"] += eq - eq_prev
        eq_prev = eq

    def flush(until):
        nonlocal ev_i, W_c, W_c_val, W_t, fund, day_next
        while True:
            t_ev = ev[ev_i][0] if ev_i < len(ev) else None
            if t_ev is not None and t_ev <= until and (day_next is None or t_ev < day_next):
                _, name, r = ev[ev_i]
                ev_i += 1
                if name == "c":
                    flow = -Q * p_last(tc, pc, t_ev) * r
                    rate = fx_at(fx, t_ev)
                    W_c += flow
                    W_c_val += flow * rate
                    fund += flow * rate
                else:
                    flow = Q * p_last(tu, pu, t_ev) * r
                    W_t += flow
                    fund += flow
                continue
            if day_next is not None and day_next <= until:
                close_day(day_next)
                day_next += DAY
                continue
            return

    def next_fill(side, frm):
        if disabled[side]:
            return None
        i0 = max(frm, int(np.searchsorted(tc, off_until[side], "left")))
        ov = override[side]
        if ov is not None:
            end_ov, px = ov[1], ov[2]
            i_end = int(np.searchsorted(tc, end_ov, "left"))
            if i0 < i_end:
                seg_p, seg_m = pc[i0:i_end], mc[i0:i_end]
                if side == 1:
                    hit = np.flatnonzero(seg_m & (seg_p < px if cfg.strict else seg_p <= px))
                else:
                    hit = np.flatnonzero(~seg_m & (seg_p > px if cfg.strict else seg_p >= px))
                if len(hit):
                    return i0 + int(hit[0]), px, ov[3]
                i0 = i_end
            else:
                i0 = max(i0, i_end)
        c = cand[side]
        j = int(np.searchsorted(c, i0, "left"))
        if j >= len(c):
            return None
        i = int(c[j])
        return i, float(bid_s[k[i]] if side == 1 else ask_s[k[i]]), float(F[k[i]])

    def replace_side(side, tau):
        prev_e = int(np.searchsorted(e, tau, "right")) - 1
        last = max(last_change[side], int(e[prev_e]) if prev_e >= 0 else -1)
        R = max(tau + cfg.l_q, last + MIN_GAP)
        Fr = p_last(tu, pu, R - cfg.l_q)
        b, a = quotes(np.array([Fr]), cfg)
        px = float(b[0] if side == 1 else a[0])
        nxt = int(np.searchsorted(e, R + MIN_GAP, "left"))
        until = int(e[nxt]) if nxt < len(e) else 2 ** 62
        off_until[side] = R
        override[side] = (R, until, px, Fr)
        last_change[side] = R

    cur = 0
    while True:
        nb, na = next_fill(1, cur), next_fill(-1, cur)
        if nb is None and na is None:
            break
        if na is None or (nb is not None and nb[0] <= na[0]):
            i, x, fq, s = nb[0], nb[1], nb[2], 1
        else:
            i, x, fq, s = na[0], na[1], na[2], -1
        tau = int(tc[i])
        flush(tau)
        room = cfg.g - s * pos_usd
        notional = min(cfg.q, float(pc[i] * qc[i]), room)
        if notional <= 0:
            disabled[s] = True
            continue
        qty = notional / x
        rate = fx_at(fx, tau)
        h_i = int(np.searchsorted(tu, tau + cfg.l_h, "left"))
        H = float(pu[min(h_i, len(pu) - 1)])
        H_eff = H * (1 - s * cfg.slip)
        flow_c = -s * qty * x - cfg.maker * qty * x
        W_c += flow_c
        W_c_val += flow_c * rate
        W_t += s * qty * H_eff - cfg.taker_usdt * qty * H_eff
        Q += s * qty
        pos_usd += s * notional
        spread += s * qty * (H - x * rate)
        fees += cfg.maker * qty * x * rate + cfg.taker_usdt * qty * H_eff
        slip_cost += qty * H * cfg.slip
        fills.append({"t": tau, "s": s, "x": x, "notional": notional, "H": H, "F_quote": fq, "fx": rate})
        dd = days.setdefault(_day(tau), {"pnl": 0.0, "notional": 0.0, "n": 0})
        dd["notional"] += notional
        dd["n"] += 1
        cur = i + 1
        replace_side(s, tau)
        if s * pos_usd >= cfg.g - 1e-9:
            disabled[s] = True
        if disabled[-s] and -s * pos_usd < cfg.g - 1e-9:
            disabled[-s] = False
            replace_side(-s, tau)
    end_t = max(end_t, int(fills[-1]["t"]) if fills else 0)
    flush(end_t)
    before = W_c_val + W_t
    p_c, p_u = p_last(tc, pc, end_t), float(pu[min(int(np.searchsorted(tu, end_t, "left")), len(pu) - 1)])
    rate = fx_at(fx, end_t)
    slip_c = cfg.slip if cfg.slip_usdc is None else cfg.slip_usdc
    if Q != 0:
        s = -1 if Q > 0 else 1                   # s = what we do on the USDC leg
        qty = abs(Q)
        flow = -s * qty * p_c * (1 + s * slip_c)
        flow -= cfg.taker_usdc * qty * p_c
        W_c += flow
        W_c_val += flow * rate
        W_t += s * qty * p_u * (1 - s * cfg.slip) - cfg.taker_usdt * qty * p_u
        Q = 0.0
    unwind = (W_c_val + W_t) - before
    total = W_c * rate + W_t
    fx_term = W_c * rate - W_c_val
    # final day mark includes the unwind
    eq = total
    days.setdefault(_day(end_t), {"pnl": 0.0, "notional": 0.0, "n": 0})["pnl"] += eq - eq_prev
    return {"fills": fills, "total": total, "spread": spread, "fees": fees, "slip": slip_cost, "funding": fund, "fx": fx_term, "unwind": unwind,
            "days": days, "w_usdc": W_c, "w_usdt": W_t, "decisions": int(len(d))}


# ------------------------------------------------------------------ data
def _days(lo: str, hi: str) -> list:
    a, b = datetime.strptime(lo, "%Y-%m-%d"), datetime.strptime(hi, "%Y-%m-%d")
    return [(a + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((b - a).days + 1)]


def fetch(window: str, log=print) -> dict:
    lo, hi = WINDOWS[window]
    STORE.mkdir(parents=True, exist_ok=True)
    tot = {"files": 0, "new": 0, "mb": 0.0, "missing": []}
    for key, (a, b, _) in PAIRS.items():
        for s in (a, b):
            for d in _days(lo, hi):
                out = STORE / f"{s}-{d}.npz"
                tot["files"] += 1
                if out.exists():
                    continue
                raw = A.http_get(f"https://{A.FILES}/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip")
                if raw is None:
                    tot["missing"].append(f"{s}-{d}")
                    continue
                tot["new"] += 1
                tot["mb"] += len(raw) / 1e6
                x = parse_agg(A._zip_text(raw))
                np.savez(out, t=x["t"], p=x["p"], q=x["q"], m=x["m"])
            log(f"{s} done {round(tot['mb'])} MB")
    log(json.dumps(tot))
    return tot


def load_tape(sym: str, lo: str, hi: str) -> dict:
    parts = []
    for d in _days(lo, hi):
        with np.load(STORE / f"{sym}-{d}.npz") as z:
            parts.append({k: z[k] for k in ("t", "p", "q", "m")})
    return {k: np.concatenate([p[k] for p in parts]) for k in ("t", "p", "q", "m")}


def tape_integrity(a: dict) -> dict:
    d = np.diff(a["t"])
    return {"n": int(len(a["t"])), "sorted": bool((d >= 0).all()), "unit_ms": bool(a["t"].max() < 10 ** 14), "buyer_maker_share": float(a["m"].mean())}


def _funding(sym: str, lo: str, hi: str):
    with np.load(USDC / f"{sym}_funding.npz") as z:
        t, r = z["t"], z["rate"]
    a, b = _ms(lo), _ms(hi) + DAY
    k = (t >= a) & (t < b)
    return {"t": t[k], "rate": r[k]}


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fx() -> dict:
    with np.load(USDC / "USDCUSDT_1m.npz") as z:
        return {"t": z["t"], "c": z["c"]}


def per_fill(r: dict, usdt: dict, cfg: Cfg) -> dict:
    """Arrays of one run for markouts, the captured spread and the minute clusters."""
    if not r["fills"]:
        return {}
    t = np.array([f["t"] for f in r["fills"]], dtype=np.int64)
    s = np.array([f["s"] for f in r["fills"]], dtype=float)
    x = np.array([f["x"] for f in r["fills"]])
    nt = np.array([f["notional"] for f in r["fills"]])
    H = np.array([f["H"] for f in r["fills"]])
    rate = np.array([f["fx"] for f in r["fills"]])
    tu, pu = usdt["t"], usdt["p"]

    def at(ms):
        return pu[np.maximum(np.searchsorted(tu, ms, "right") - 1, 0)]

    F0 = at(t)
    out = {"captured_bp": float(np.average(s * (F0 - x * rate) / F0, weights=nt) * 1e4), "markouts_bp": {}}
    for dt in MARKOUTS:
        out["markouts_bp"][str(dt // 1000) + "s"] = float(np.average(s * (at(t + dt) - F0) / F0, weights=nt) * 1e4)
    out["share_fast_F_2bp"] = float(np.mean(np.abs(F0 - at(t - 1000)) / F0 > 2e-4))
    qty = nt / x
    v = (s * qty * (H - x * rate) - qty * H * cfg.slip - cfg.taker_usdt * qty * H * (1 - s * cfg.slip) - cfg.maker * qty * x * rate) / nt * 1e4
    minute = t // 60_000
    u, inv, cnt = np.unique(minute, return_inverse=True, return_counts=True)
    m_bar = float(cnt.mean())
    grand = float(v.mean())
    means = np.bincount(inv, weights=v) / cnt
    k = len(u)
    n = len(v)
    msb = float((cnt * (means - grand) ** 2).sum() / max(1, k - 1))
    msw = float(((v - means[inv]) ** 2).sum() / max(1, n - k))
    m0 = (n - float((cnt ** 2).sum()) / n) / max(1, k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw) if msb + (m0 - 1) * msw > 0 else 0.0
    out.update({"sigma_bp": float(v.std(ddof=1)) if n > 1 else None, "mean_v_bp": grand, "m_per_minute": m_bar, "rho_minute": float(icc), "n_minutes": int(k)})
    return out


def run_window(window: str) -> dict:
    lo, hi = WINDOWS[window]
    t0 = time.time()
    inv = {r["symbol"]: r for r in json.loads((EXP / "N1_INVENTORY.json").read_text(encoding="utf-8"))["rows"]}
    fx = _fx()
    res = {}
    integ = {}
    for key, (a, b, slip) in PAIRS.items():
        usdc, usdt = load_tape(a, lo, hi), load_tape(b, lo, hi)
        integ[a], integ[b] = tape_integrity(usdc), tape_integrity(usdt)
        fc, fu = _funding(a, lo, hi), _funding(b, lo, hi)
        sched = schedule(usdt["t"], usdt["p"])
        print(key, "tape", len(usdc["t"]), len(usdt["t"]), "decisions", len(sched[0]), round(time.time() - t0), "s", flush=True)
        tick = inv[a]["tick"]
        h = TAKER_USDT + slip
        for theta in THETAS:
            for lq in LAGS:
                for lh in LAGS:
                    cfg = Cfg(theta=theta, l_q=lq, l_h=lh, h=h, tick=tick, slip=slip)
                    r = simulate(usdc, usdt, cfg, fx=fx, funding_c=fc, funding_u=fu, sched=sched)
                    pf = per_fill(r, usdt, cfg)
                    item = {k: r[k] for k in ("total", "spread", "fees", "slip", "funding", "fx", "unwind", "days")}
                    item.update({"n_fills": len(r["fills"]), "notional": float(sum(f["notional"] for f in r["fills"])), **pf, "theta_bp": theta * 1e4, "L_q": lq, "L_h": lh, "h_bp": h * 1e4})
                    res.setdefault(key, {})[f"{int(theta * 1e4)}|{lq}|{lh}"] = item
                    if lq == 1000 and lh == 1000:
                        ceq = Cfg(theta=theta, l_q=lq, l_h=lh, h=h, tick=tick, slip=slip, strict=False)
                        re = simulate(usdc, usdt, ceq, fx=fx, funding_c=fc, funding_u=fu, sched=sched)
                        res[key][f"{int(theta * 1e4)}|{lq}|{lh}"]["equal_price_sensitivity"] = {"n_fills": len(re["fills"]), "total": re["total"], "notional": float(sum(f["notional"] for f in re["fills"]))}
                    print(f"  {key} theta {theta * 1e4:.0f} L {lq}/{lh}: fills {len(r['fills'])} net {r['total']:.2f} spread {r['spread']:.2f} fees {r['fees']:.2f} fx {r['fx']:.2f} fund {r['funding']:.2f} unwind {r['unwind']:.2f} {round(time.time() - t0)}s", flush=True)
    pooled = {}
    for theta in THETAS:
        for lq in LAGS:
            for lh in LAGS:
                c = f"{int(theta * 1e4)}|{lq}|{lh}"
                days = sorted({d for k in res for d in res[k][c]["days"]})
                num = [sum(res[k][c]["days"].get(d, {}).get("pnl", 0.0) for k in res) for d in days]
                den = [sum(res[k][c]["days"].get(d, {}).get("notional", 0.0) for k in res) for d in days]
                ci = NC.ratio_ci(num, den, alpha=ALPHA, block=1, scale=1e4)
                cap = 2 * 20000.0 * len(res)
                nf = sum(res[k][c]["n_fills"] for k in res)
                wv = sum(res[k][c].get("captured_bp", 0) * res[k][c]["notional"] for k in res)
                tot_not = sum(res[k][c]["notional"] for k in res) or 1
                pooled[c] = {"mean_bp": ci["mean"], "ci": ci, "n_fills": nf, "fills_per_day": nf / max(1, len(days)), "book_R_pct_per_day": float(np.mean(num)) / cap * 100 if num else None,
                             "total": sum(res[k][c]["total"] for k in res), "funding": sum(res[k][c]["funding"] for k in res), "fx": sum(res[k][c]["fx"] for k in res),
                             "unwind": sum(res[k][c]["unwind"] for k in res), "captured_bp": wv / tot_not,
                             "markout_1s_bp": sum(res[k][c].get("markouts_bp", {}).get("1s", 0) * res[k][c]["notional"] for k in res) / tot_not,
                             "by_pair": {k: {"mean_bp": res[k][c]["total"] / res[k][c]["notional"] * 1e4 if res[k][c]["notional"] else None, "n_fills": res[k][c]["n_fills"]} for k in res}}
    return {"pairs": res, "pool": pooled, "integrity": integ, "minutes": round((time.time() - t0) / 60, 2)}


def report(stage: str, window: str) -> dict:
    import provenance as PV
    PV.require_clean()
    r = run_window(window)
    main_c = {c: r["pool"][c] for c in ("1|1000|1000", "3|1000|1000")}
    pw = {}
    days = max(1, len(_days(*WINDOWS[window])))
    for c in main_c:
        sig, m, rho = [], [], []
        for k in r["pairs"]:
            x = r["pairs"][k][c]
            if x.get("sigma_bp"):
                sig.append((x["sigma_bp"], x["n_fills"]))
                m.append((x["m_per_minute"], x["n_fills"]))
                rho.append((x["rho_minute"], x["n_fills"]))
        if sig:
            w = np.array([b for _, b in sig], dtype=float)
            s_ = float(np.average([a for a, _ in sig], weights=w))
            m_ = float(np.average([a for a, _ in m], weights=w))
            rho_ = max(0.0, float(np.average([a for a, _ in rho], weights=w)))
            n, ne = NC.required_n(s_, DELTA, ALPHA, m=m_, rho=rho_)
            pw[c] = {"delta_bp": DELTA, "alpha": ALPHA, "sigma_bp": s_, "m": m_, "rho": rho_, "n": n, "n_eff": ne, "fills_in_30d_at_this_rate": main_c[c]["n_fills"] / days * 30}
    stop_cell = r["pool"]["3|250|250"]["mean_bp"]
    out = {"spec": SPEC, "stage": stage, "window": WINDOWS[window], **r, "main_candidates": main_c, "power": pw, "pilot_stop_cell_bp": stop_cell,
           "pilot_stop": bool(stop_cell is None or stop_cell <= 0), "verdict": None if stage == "pilot" else "exploration: no verdict",
           "not_modelled": ["queue position (strictly-through proxy)", "own order impact", "post-only rejections", "exchange downtime", "partial hedge", "historical tickSize", "funding on hedge between fill and hedge"],
           "fee_grid": "current grid applied to the window (USDC maker 0 since 2025-12-10)", "traps_checked": [2, 4, 8, 9, 10], "provenance": PV.block()}
    name = {"pilot": "N2_PILOT.json", "exploration": "N2_EXPLORE.json"}[stage]
    (EXP / name).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for c, v in r["pool"].items():
        print(c, {k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in ("mean_bp", "n_fills", "fills_per_day", "book_R_pct_per_day", "total", "funding", "fx", "unwind", "captured_bp", "markout_1s_bp")}, v["by_pair"])
    print(json.dumps({"power": pw, "stop_cell": stop_cell, "stop": out["pilot_stop"], "minutes": r["minutes"]}, default=str, indent=0))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fetch"]:
        fetch(argv[1], log=lambda s: print(s, flush=True))
    elif argv[:1] == ["pilot"]:
        report("pilot", "pilot")
    elif argv[:1] == ["explore"]:
        report("exploration", "explore")
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
