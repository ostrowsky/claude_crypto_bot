"""G1 funding harvest, G2 premium convergence, G3b FC at L = 2 -- stage 0 on data already on disk
(funding-premium-g1g2-1105-spec.md). Exploration window 2023-03..2025-12 only; nothing is downloaded.

  g1_trade   the side that receives funding, from the close 15 min before the event to the close 15 min after it
  causal_ok  the causal selection row: the PREVIOUS event already had the same sign and cleared the threshold
  g2_trades  short perp + long spot on equal notional after the perp's premium to spot clears a threshold; exit one bar
             after it falls below a third of it, or after 32 bars; funding over the hold; one position per coin
  thr_of_bar / cost_of_bar take a bar's open time in ms

  (from trader/) ../pyembed/python.exe -u funding_premium.py run
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

import trader_config as TC        # noqa: E402

SPEC = "funding-premium-g1g2-1105-spec.md"
OUT = TC.RUNTIME / "experiments" / "FUNDING_PREMIUM_STAGE0.json"
BAR = 15 * 60 * 1000
DAY = 86_400_000
THRESHOLDS = (0.001, 0.003)
ALPHA = 0.05 / 3
G2_HOLD_BARS, G2_MIN_SIGNAL = 32, 0.004
G2_THR_TOP20, G2_THR_TOP100 = 0.004, 0.008
G2_LISTING_DAYS = 7
SPOT_BNB_TAKER = 0.00075
PERP_TAKER = 0.0005
PERP_MAKER = 0.0002


# ------------------------------------------------------------------ G1
def causal_ok(rate: float, prev_rate, thr: float) -> bool:
    if prev_rate is None or abs(rate) < thr or abs(prev_rate) < thr:
        return False
    return (rate > 0) == (prev_rate > 0)


def g1_trade(bars: dict, t_fund: int, rate: float, cost_side: float, prev_rate=None):
    t, c = bars["t"], bars["c"]
    t0 = int(t_fund) - int(t_fund) % BAR
    e_open, x_open = t0 - 2 * BAR, t0
    e = int(np.searchsorted(t, e_open))
    x = int(np.searchsorted(t, x_open))
    if e >= len(t) or x >= len(t) or int(t[e]) != e_open or int(t[x]) != x_open:
        return None
    if not (np.isfinite(c[e]) and np.isfinite(c[x])) or c[e] <= 0:
        return None
    move = (float(c[x]) / float(c[e]) - 1) * 100
    price = -move if rate > 0 else move
    r = abs(float(rate)) * 100
    cost = 2 * float(cost_side) * 100
    return {"ret_pct": price + r - cost, "price_pct": price, "rate_pct": r, "cost_pct": cost,
            "entry_close_ms": e_open + BAR, "prev_rate": prev_rate}


# ------------------------------------------------------------------ G2
def g2_trades(perp: dict, spot: dict, mult: float, thr_of_bar, funding, cost_of_bar, excluded_until_ms: int = 0) -> list:
    tp, ts = np.asarray(perp["t"], dtype=np.int64), np.asarray(spot["t"], dtype=np.int64)
    t, ip, isp = np.intersect1d(tp, ts, return_indices=True)
    if not len(t):
        return []
    cp, cs = np.asarray(perp["c"], dtype=float)[ip] * float(mult), np.asarray(spot["c"], dtype=float)[isp]
    with np.errstate(divide="ignore", invalid="ignore"):
        prem = cp / cs - 1
    if funding is not None and len(funding["t"]):
        ft, fr = np.asarray(funding["t"], dtype=np.int64), np.asarray(funding["rate"], dtype=float)
    else:
        ft, fr = np.zeros(0, dtype=np.int64), np.zeros(0)
    out, i, n = [], 0, len(t)
    cands = np.flatnonzero(prem > G2_MIN_SIGNAL)
    k = 0
    while k < len(cands):
        i = int(cands[k])
        k += 1
        if int(t[i]) < excluded_until_ms or i + 1 >= n:
            continue
        thr = thr_of_bar(int(t[i]))
        if thr is None or not prem[i] > thr:
            continue
        e = i + 1
        last = e + G2_HOLD_BARS
        x, reason = last, "time"
        below = np.flatnonzero(prem[e + 1:min(last, n)] < thr / 3)
        if len(below):
            x, reason = e + 1 + int(below[0]) + 1, "converged"
        if x >= n:                                        # the data ends before the exit: no result yet
            break
        if not (np.isfinite(cp[e]) and np.isfinite(cp[x]) and np.isfinite(cs[e]) and np.isfinite(cs[x])):
            continue
        price = ((cp[e] - cp[x]) / cp[e] + (cs[x] - cs[e]) / cs[e]) * 100
        e_close, x_close = int(t[e]) + BAR, int(t[x]) + BAR
        fund = float(np.sum(fr[(ft > e_close) & (ft <= x_close)])) * 100
        cost = float(cost_of_bar(int(t[e]))) * 100
        out.append({"ret_pct": float(price) + fund - cost, "price_pct": float(price), "funding_pct": fund, "cost_pct": cost,
                    "entry_close_ms": e_close, "reason": reason, "prem_entry": float(prem[e]) * 100,
                    "prem_exit": float(prem[x]) * 100, "bars": x - e})
        while k < len(cands) and int(cands[k]) <= x:      # one position per coin: a new signal only after the exit
            k += 1
    return out


# ------------------------------------------------------------------ stage 0
def _year(ms) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y")


def _stats(rows, alpha: float = ALPHA) -> dict:
    import la_short_train as LS
    if not rows:
        return {"n": 0}
    ci = LS.day_cluster_ci(rows, alpha=alpha)
    r = np.array([x["ret_pct"] for x in rows])
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else None
    half = (ci["hi"] - ci["lo"]) / 2
    by_b, by_y = {}, {}
    for x in rows:
        by_b.setdefault(x["bucket"], []).append(x["ret_pct"])
        by_y.setdefault(_year(x["entry_close_ms"]), []).append(x["ret_pct"])
    no_b4 = [x["ret_pct"] for x in rows if x["bucket"] != "B4"]
    out = {**ci, "sd": sd, "median": float(np.median(r)), "pos_share": float(np.mean(r > 0)),
           "mde_80": half * (1 + 0.8416 / 2.394),
           "by_bucket": {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(by_b.items())},
           "by_year": {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(by_y.items())},
           "mean_without_B4": float(np.mean(no_b4)) if no_b4 else None, "n_without_B4": len(no_b4)}
    for key in ("price_pct", "rate_pct", "funding_pct", "cost_pct", "prem_entry", "prem_exit", "bars", "ret_maker_pct"):
        vals = [x[key] for x in rows if key in x]
        if vals:
            out["mean_" + key] = float(np.mean(vals))
    pos = [v["mean"] for v in out["by_year"].values() if v["mean"] > 0]
    out["passes"] = bool(ci["lo"] > 0 and (out["mean_without_B4"] or 0) > 0 and len(pos) >= 2)
    return out


def run() -> dict:
    import cost_calibration as CC
    import experiment_fc as EF
    import carry as C
    import daily_revenue as DR
    import funding_store as FS
    import history_store as HS
    import hype_short as H
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    p = H.build_panel("recent", *H.EXPLORE)
    a, b = H._window(p, *H.EXPLORE)
    table = json.loads(CC.OUT.read_text(encoding="utf-8"))["tables"]["live"]
    slip = {k: (table[k]["buy"]["1000"]["median"] + table[k]["sell"]["1000"]["median"]) / 2 / 1e4 for k in table}
    rank_cache = {}

    def ranks(i):
        if i not in rank_cache:
            turn = np.nansum(p.qvol[max(0, i - H.TURN_DAYS):i], axis=0)
            order = np.argsort(-turn, kind="stable")
            r = np.empty(len(order), dtype=np.int64)
            r[order] = np.arange(1, len(order) + 1)
            rank_cache[i] = r
        return rank_cache[i]

    day0 = p.days[0]
    lo_ms, hi_ms = H._day(H.EXPLORE[0]) * DAY, H._day(H.EXPLORE[1]) * DAY
    mapping = json.loads(FS.MAPPING.read_text(encoding="utf-8"))
    g1 = {f"{thr}|{row}": [] for thr in THRESHOLDS for row in ("A", "B")}
    g2, intervals = [], {}
    for col, sym in enumerate(p.syms):
        with np.load(FS.perp_path(sym)) as z:
            bars = {"t": z["t"].astype(np.int64), "c": z["c"].astype(float)}
        f = FS.load_funding(FS.funding_path(sym))
        if f is not None and len(f["t"]):
            ft, fr = f["t"].astype(np.int64), f["rate"].astype(float)
            w = np.flatnonzero((ft >= lo_ms) & (ft < hi_ms) & (np.abs(fr) >= THRESHOLDS[0]))
            for k in w:
                t, rate = int(ft[k]), float(fr[k])
                i = t // DAY - day0
                if not (0 <= i < len(p.days)) or p.first[col] > i - H.AGE_MIN:
                    continue
                rank = int(ranks(i)[col])
                if rank > H.TOP_N:
                    continue
                bk = CC.bucket(sym, rank)
                side_cost = PERP_TAKER + slip[bk]
                tr = g1_trade(bars, t, rate, side_cost)
                if tr is None:
                    continue
                prev = float(fr[k - 1]) if k > 0 else None
                hours = round((t - int(ft[k - 1])) / 3_600_000) if k > 0 else None
                maker = tr["ret_pct"] + (side_cost - PERP_MAKER) * 100
                row = {**tr, "bucket": bk, "sym": sym, "interval_h": hours, "ret_maker_pct": maker}
                for thr in THRESHOLDS:
                    if abs(rate) >= thr:
                        g1[f"{thr}|A"].append(row)
                        if causal_ok(rate, prev, thr):
                            g1[f"{thr}|B"].append(row)
        # G2
        sp = HS.load(HS.path_of(sym))
        if sp is None or not len(sp["t"]):
            continue
        spot = {"t": np.asarray(sp["t"], dtype=np.int64), "c": np.asarray(sp["c"], dtype=float)}
        m = 1 / float((mapping.get(sym) or {}).get("mult") or 1)
        mask_p = (bars["t"] >= lo_ms) & (bars["t"] < hi_ms)
        excl = max(int(bars["t"][0]), int(spot["t"][0])) + G2_LISTING_DAYS * DAY

        def thr_of(tb, col=col):
            i = tb // DAY - day0
            if not (0 <= i < len(p.days)):
                return None
            r = int(ranks(i)[col])
            return G2_THR_TOP20 if r <= 20 else (G2_THR_TOP100 if r <= H.TOP_N else None)

        def cost_of(tb, col=col, sym=sym):
            i = tb // DAY - day0
            s = slip[CC.bucket(sym, int(ranks(i)[col]))]
            return 2 * (SPOT_BNB_TAKER + PERP_TAKER) + 4 * s

        for x in g2_trades({"t": bars["t"][mask_p], "c": bars["c"][mask_p]}, spot, m, thr_of, f, cost_of, excl):
            i = x["entry_close_ms"] // DAY - day0
            x["bucket"] = CC.bucket(sym, int(ranks(min(i, len(p.days) - 1))[col]))
            x["sym"] = sym
            g2.append(x)
    out = {"spec": SPEC, "stage": "exploration, stage 0", "window": H.EXPLORE, "alpha": ALPHA, "g1": {}}
    for key, rows in g1.items():
        s = _stats(rows)
        for hv in (1, 2, 4, 8):
            vv = [x["ret_pct"] for x in rows if x["interval_h"] == hv]
            if vv:
                s.setdefault("by_interval_h", {})[str(hv)] = {"n": len(vv), "mean": float(np.mean(vv))}
        out["g1"][key] = s
        print(f"G1 thr {key}: n {s.get('n')} mean {s.get('mean'):+.3f} [{s.get('lo'):+.3f}, {s.get('hi'):+.3f}] | price "
              f"{s.get('mean_price_pct'):+.3f} rate {s.get('mean_rate_pct'):+.3f} cost {s.get('mean_cost_pct'):.3f} | maker "
              f"{s.get('mean_ret_maker_pct'):+.3f} | sd {s.get('sd'):.2f} | no B4 {s.get('mean_without_B4'):+.3f} (n "
              f"{s.get('n_without_B4')}) | {json.dumps(s.get('by_bucket'))} | {json.dumps(s.get('by_year'))} | "
              f"{json.dumps(s.get('by_interval_h'))} | {'PASS' if s['passes'] else '-'}", flush=True)
    s = _stats(g2)
    reasons = {}
    for x in g2:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    s["reasons"] = reasons
    out["g2"] = s
    print(f"G2: n {s.get('n')} mean {s.get('mean'):+.3f} [{s.get('lo'):+.3f}, {s.get('hi'):+.3f}] | price "
          f"{s.get('mean_price_pct'):+.3f} funding {s.get('mean_funding_pct'):+.3f} cost {s.get('mean_cost_pct'):.3f} | prem "
          f"{s.get('mean_prem_entry'):+.3f} -> {s.get('mean_prem_exit'):+.3f} | bars {s.get('mean_bars'):.1f} | sd "
          f"{s.get('sd'):.2f} | no B4 {s.get('mean_without_B4')} | {json.dumps(s.get('by_bucket'))} | "
          f"{json.dumps(s.get('by_year'))} | {reasons} | {'PASS' if s['passes'] else '-'}", flush=True)
    # G3b
    market = EF.load_market()
    fees = C.fees(1)
    l1 = EF.evaluate(market, EF.passive_config(), fees, EF.TRAIN_FROM, EF.TEST_FROM, 0)
    l2 = EF.evaluate(market, EF.passive_config(), fees, EF.TRAIN_FROM, EF.TEST_FROM, 0, lev_override=2)
    pr = DR.paired(l2["_series"], l1["_series"], block=20)
    out["g3b"] = {"l1": DR.aggregate(l1["_series"], block=20), "l2": DR.aggregate(l2["_series"], block=20), "paired_l2_l1": pr,
                  "l1_margin_stops": l1["margin_stops"], "l2_margin_stops": l2["margin_stops"],
                  "l1_liq": l1["liquidations"], "l2_liq": l2["liquidations"], "r_by_year_l2": l2["r_by_year"]}
    print("G3b", json.dumps(out["g3b"], default=str)[:900], flush=True)
    out["provenance"] = PV.block()
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
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
