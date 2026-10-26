"""DailyRevenue v1 aggregation (daily-revenue-metric-1010-spec.md §4) -- shared by the demo
account metric (phase D) and the E1 replay, so replay and fact are aggregated by ONE code path.

aggregate(series) takes [(day, R_pct)] for the days that count (FULL + BOT_DOWN) and returns
mean with a 5-day block-bootstrap 95% CI, compounded return, max drawdown on the equity
curve, share of positive days, worst day. paired(a, b) does the same for a - b on common days.
"""
from __future__ import annotations

import io
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BLOCK = 5
BOOT = 2000
MIN_DAYS_DIRECTION = 2 * BLOCK     # fewer days: the block bootstrap degenerates (1 day -> CI [x, x])


def _block_boot_means(x: list, rnd: random.Random, block: int = BLOCK, boot: int = BOOT) -> list:
    n = len(x)
    if n == 0:
        return []
    b = max(1, min(block, n))
    starts = n - b + 1
    out = []
    for _ in range(boot):
        s, cnt = 0.0, 0
        while cnt < n:
            i = rnd.randrange(starts)
            for v in x[i:i + b]:
                if cnt >= n:
                    break
                s += v
                cnt += 1
        out.append(s / n)
    out.sort()
    return out


def aggregate(series, seed: int = 29, excluded: int = 0, alpha: float = 0.05) -> dict:
    """lo95/hi95 hold the (1 - alpha) interval; alpha = 0.05/k for a Bonferroni-corrected family."""
    series = sorted(series)
    x = [float(r) for _, r in series]
    n = len(x)
    if n == 0:
        return {"n_days": 0, "excluded": excluded}
    m = sum(x) / n
    bs = _block_boot_means(x, random.Random(seed))
    eq, peak, mdd = 1.0, 1.0, 0.0
    for v in x:
        eq *= 1 + v / 100
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return {"n_days": n, "excluded": excluded, "from": series[0][0], "to": series[-1][0],
            "mean_pct": round(m, 4), "lo95": round(bs[int(alpha / 2 * len(bs))], 4),
            "hi95": round(bs[min(len(bs) - 1, int((1 - alpha / 2) * len(bs)))], 4),
            "sd_pct": round(math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)), 4) if n > 1 else None,
            "compound_pct": round((eq - 1) * 100, 3), "max_dd_pct": round(mdd * 100, 3),
            "pos_share": round(sum(1 for v in x if v > 0) / n, 3), "worst_pct": round(min(x), 4)}


def paired(a, b, seed: int = 29, alpha: float = 0.05) -> dict:
    """a - b on the days both have."""
    da, db = dict(a), dict(b)
    common = sorted(set(da) & set(db))
    return aggregate([(d, da[d] - db[d]) for d in common], seed, alpha=alpha)


def month_means(series) -> dict:
    by = {}
    for d, v in series:
        by.setdefault(d[:7], []).append(v)
    return {m: sum(v) / len(v) for m, v in sorted(by.items())}


def verdict_text(agg: dict) -> str:
    """The wording rule of §4: a direction only when the CI excludes zero."""
    if not agg.get("n_days"):
        return "нет данных"
    if agg["n_days"] < MIN_DAYS_DIRECTION:
        return "рано судить"
    if agg["lo95"] > 0:
        return "выше нуля"
    if agg["hi95"] < 0:
        return "ниже нуля"
    return "рано судить"


# ---------------------------------------------------------------------------
# Phase D: the demo account's own day (demo-ledger-d-1013-spec.md §4)
# ---------------------------------------------------------------------------

MARK_WINDOW = timedelta(minutes=10)
SNAP_EVERY_MIN = 5
LIVE_SHARE_MIN = 0.95
RESET_PCT = 0.5
RECONCILE_PCT = 0.01
QUOTE = "USDT"
FEE_RESERVE = frozenset({"BNB"})   # held only to pay fees at the discount
STABLES = frozenset({"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "XUSD", "BFUSD", "PYUSD", "RLUSD", "USD1"})


def _ts(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(v)


def _day_dt(day) -> datetime:
    return day if isinstance(day, datetime) else datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def midnight_mark(snaps, day_dt):
    """Snapshot nearest to 00:00 UTC of day_dt within ±10 min, else None (method P)."""
    best, gap = None, None
    for s in snaps:
        g = abs(_ts(s["ts"]) - day_dt)
        if g <= MARK_WINDOW and (gap is None or g < gap):
            best, gap = s, g
    return best


def _apply_fills(qty: dict, fills) -> dict:
    """Quantities after fills: base ±qty, quote ∓quote_qty, commission − in its asset."""
    q = dict(qty)
    for f in fills:
        sym = f["symbol"]
        base = sym[: -len(QUOTE)] if sym.endswith(QUOTE) else sym
        sgn = 1.0 if f["side"] == "BUY" else -1.0
        q[base] = q.get(base, 0.0) + sgn * float(f["qty"])
        q[QUOTE] = q.get(QUOTE, 0.0) - sgn * float(f["quote_qty"])
        if f.get("commission_asset"):
            q[f["commission_asset"]] = q.get(f["commission_asset"], 0.0) - float(f.get("commission") or 0)
    return q


def compute_day(day: str, snaps, fills) -> dict:
    """One UTC day of the demo account: R by equity marks (P), reconciled against the ledger (L)."""
    d0 = _day_dt(day)
    d1 = d0 + timedelta(days=1)
    in_day = [s for s in snaps if d0 <= _ts(s["ts"]) < d1]
    share = len(in_day) / (24 * 60 / SNAP_EVERY_MIN)
    day_fills = [f for f in fills if d0 <= _ts(f["time"]) < d1]
    rec = {"day": day, "schema": "daily_revenue.demo.v1", "snapshot_share": round(share, 4),
           "fills": len(day_fills), "provenance": None, "R_pct": None, "R_policy_pct": None,
           "stable_contrib_pct": None, "fee_reserve_contrib_pct": None, "P_minus_L_pct": None}
    m0, m1 = midnight_mark(snaps, d0), midnight_mark(snaps, d1)
    if m0 is None or m1 is None:
        rec.update(status="UNKNOWN", reason="no_midnight_mark" + ("_start" if m0 is None else "_end"))
        return rec
    if m0.get("unpriced_count") or m1.get("unpriced_count"):
        rec.update(status="UNKNOWN", reason="unpriced_assets")
        return rec
    e0, e1 = float(m0["equity_usdt"]), float(m1["equity_usdt"])
    rec.update(provenance="equity_snapshots", mark_start=m0["ts"], mark_end=m1["ts"],
               equity_start=e0, equity_end=e1, R_pct=(e1 / e0 - 1) * 100)
    h0, h1 = m0.get("holdings"), m1.get("holdings")
    status = "FULL" if share >= LIVE_SHARE_MIN else "BOT_DOWN"
    if h0 is not None and h1 is not None:
        stable = sum(h["qty"] * (h1[a]["px"] - h["px"]) for a, h in h0.items() if a in STABLES and a in h1)
        rec["stable_contrib_pct"] = stable / e0 * 100
        # the BNB fee reserve (2026-10-22) is not a position: its price move is not policy revenue either
        reserve = sum(h["qty"] * (h1[a]["px"] - h["px"]) for a, h in h0.items() if a in FEE_RESERVE and a in h1)
        rec["fee_reserve_contrib_pct"] = reserve / e0 * 100
        rec["R_policy_pct"] = rec["R_pct"] - rec["stable_contrib_pct"] - rec["fee_reserve_contrib_pct"]
        q_end = _apply_fills({a: h["qty"] for a, h in h0.items()}, day_fills)
        px_end = {a: h["px"] for a, h in h1.items()}
        missing = [a for a, q in q_end.items() if abs(q) > 1e-12 and a not in px_end]
        if missing:
            rec["L_missing_price"] = sorted(missing)
        else:
            L = sum(q * px_end[a] for a, q in q_end.items())
            rec["L_equity"] = L
            rec["P_minus_L_pct"] = (e1 - L) / e0 * 100
            gap = abs(rec["P_minus_L_pct"])
            if gap > RESET_PCT:
                status = "RESET"
            elif gap > RECONCILE_PCT:
                status = "RECONCILE_FAIL"
    else:
        rec["reason"] = "snapshot_without_holdings"     # L and the stablecoin split unavailable
    rec["status"] = status
    return rec


def benchmarks(btc_open_close, universe_open_close) -> dict:
    o, c = btc_open_close if btc_open_close else (None, None)
    u = [(c2 / o2 - 1) * 100 for o2, c2 in (universe_open_close or []) if o2]
    return {"bench_btc_pct": (c / o - 1) * 100 if o else None,
            "bench_universe_pct": sum(u) / len(u) if u else None, "bench_universe_n": len(u),
            "bench_usdt_pct": 0.0}


def _read_rows(path: Path) -> list:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def append_day(path: Path, rec: dict) -> dict:
    """Append-only: a recomputed day is a new row with revision + 1."""
    path = Path(path)
    prev = [r for r in _read_rows(path) if r.get("day") == rec["day"]]
    rec = {**rec, "revision": max([r.get("revision", 0) for r in prev] + [0]) + 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def latest_by_day(path: Path) -> dict:
    out = {}
    for r in _read_rows(path):
        if r.get("revision", 0) >= out.get(r["day"], {}).get("revision", 0):
            out[r["day"]] = r
    return out


def _bench_live(day: str) -> dict:
    """Daily candles from the public API: BTC and the equal-weight universe (tradable + liquidity_unmeasured)."""
    import exchange_client as X
    import trader_config as TC
    c = X.ExchangeClient()
    t0 = int(_day_dt(day).timestamp() * 1000)

    def oc(sym):
        try:
            for k in c.klines(sym, "1d", 3, base=X.LIVE_REST):
                if int(k[0]) == t0:
                    return float(k[1]), float(k[4])
        except Exception:
            return None
        return None
    u = json.loads((TC.RUNTIME / "trading" / "universe.json").read_text(encoding="utf-8"))
    syms = [p["symbol"] for p in u.get("pairs", []) if p.get("status") in ("tradable", "liquidity_unmeasured")]
    pairs = [x for x in (oc(s) for s in syms) if x]
    b = benchmarks(oc("BTCUSDT"), pairs)
    b["bench_universe_of"] = len(syms)
    return b


def compute_main(day: str | None = None, bench: bool = True) -> dict:
    import trader_config as TC
    day = day or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    d0 = _day_dt(day)
    snaps = [s for s in _read_rows(TC.RUNTIME / "trading" / "equity_snapshots.jsonl")
             if d0 - timedelta(minutes=15) <= _ts(s["ts"]) <= d0 + timedelta(days=1, minutes=15)]
    rec = compute_day(day, snaps, _read_rows(TC.RUNTIME / "trading" / "ledger.jsonl"))
    if bench:
        rec.update(_bench_live(day))
    rec["computed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return append_day(TC.RUNTIME / "metrics" / "daily_revenue.jsonl", rec)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).resolve().parent))      # embedded python ignores the script dir
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        import audit
        import faulthandler
        faulthandler.dump_traceback_later(600, exit=True)      # watchdog
        audit.set_process("daily_revenue")
        r = compute_main(sys.argv[2] if len(sys.argv) > 2 else None, bench="--no-bench" not in sys.argv)
        print(json.dumps({k: r.get(k) for k in ("day", "status", "R_pct", "R_policy_pct", "stable_contrib_pct",
                                                "P_minus_L_pct", "snapshot_share", "bench_btc_pct", "revision")}))
