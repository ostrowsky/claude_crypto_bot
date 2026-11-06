"""N5 -- is implied volatility above realized by more than a round trip costs? Research only (n5-1106-spec.md).

  fetch  BVOL value at 00:00 UTC of each day and the spread rows of ~30-day ATM options; raw files are not kept
  run    N5_VRP.json

  (from trader/) ../pyembed/python.exe -u n5_vrp.py fetch | run
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import archive_bh as A            # noqa: E402
import trader_config as TC        # noqa: E402

SPEC = "n5-1106-spec.md"
STORE = TC.RUNTIME / "n5"
EXP = TC.RUNTIME / "experiments"
BAR = 900_000
DAY = 86_400_000
FIRST, LAST_IV, LAST_FILE = "2023-06-20", "2025-12-01", "2025-12-31"
INTERVALS = 2880
MIN_SHARE = 0.95
LAG = 30
COST_SIDE = 0.0005
KNOWN_GAPS = ("2023-09-25", "2023-10-24", "2024-06-11", "2024-06-12", "2024-06-30")
COINS = {"BTC": ("BTCBVOLUSDT", "BTCUSDT"), "ETH": ("ETHBVOLUSDT", "ETHUSDT")}


def first_minute(text: str, day_ms: int):
    for r in csv.reader(io.StringIO(text)):
        if not r or not r[0].strip().lstrip("-").isdigit():
            continue
        t = int(r[0])
        if t > 10 ** 14:
            t //= 1000
        if day_ms <= t < day_ms + 60_000:
            return float(r[-1])
        if t >= day_ms + 60_000:
            return None
    return None


def rv30(t, c, start_ms: int):
    t, c = np.asarray(t, dtype=np.int64), np.asarray(c, dtype=float)
    end = start_ms + 30 * DAY
    k = (t >= start_ms) & (t <= end)
    tt, cc = t[k], c[k]
    if len(tt) < 2:
        return None
    ok = (np.diff(tt) == BAR) & (cc[1:] > 0) & (cc[:-1] > 0)
    if int(ok.sum()) < MIN_SHARE * INTERVALS:
        return None
    r = np.log(cc[1:][ok] / cc[:-1][ok])
    return float(100 * math.sqrt(365 / 30 * float((r ** 2).sum())))


def newey_west(x, lag: int) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    v = float((d ** 2).sum()) / n
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * float((d[j:] * d[:-j]).sum()) / n
    return {"mean": float(x.mean()), "se": math.sqrt(max(v, 0.0) / n), "n": n, "lag": lag}


def leland_points(sigma: float, cost_side: float = COST_SIDE) -> float:
    if cost_side == 0:
        return 0.0
    le = math.sqrt(2 / math.pi) * (2 * cost_side) / (sigma * math.sqrt(1 / 365))
    return 100 * sigma * (math.sqrt(1 + le) - 1)


def expiry_days(symbol: str, day: str) -> int:
    code = symbol.split("-")[1]
    exp = date(2000 + int(code[:2]), int(code[2:4]), int(code[4:6]))
    return (exp - date.fromisoformat(day)).days


def spread_rows(text: str) -> list:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            if r["hour"] not in ("00", "12", "0", "12"):
                continue
            if not 25 <= expiry_days(r["symbol"], r["date"]) <= 35:
                continue
            delta = abs(float(r["delta"]))
            if not 0.40 <= delta <= 0.60:
                continue
            bid, ask, vega = float(r["best_bid_price"]), float(r["best_ask_price"]), float(r["vega"])
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        if bid > 0 and ask > bid and vega > 0:
            out.append((r["date"], r["hour"], r["symbol"], delta, (ask - bid) / vega))
    return out


def day_spread(rows: list):
    if not rows:
        return None
    return float(np.median([x[4] for x in rows]))


# ------------------------------------------------------------------ data
def _days(lo: str, hi: str) -> list:
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch(log=print) -> dict:
    STORE.mkdir(parents=True, exist_ok=True)
    tot = {}
    with ThreadPoolExecutor(6) as ex:
        for coin, (sym, under) in COINS.items():
            path = STORE / f"{coin}_iv.json"
            have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            todo = [d for d in _days(FIRST, LAST_IV) if d not in have]

            def one(d, s=sym):
                raw = A.http_get(f"https://{A.FILES}/data/option/daily/BVOLIndex/{s}/{s}-BVOLIndex-{d}.zip")
                return d, (first_minute(A._zip_text(raw), _ms(d)) if raw else None), raw is not None

            for i, (d, v, exists) in enumerate(ex.map(one, todo)):
                have[d] = {"iv": v, "file": exists}
                if i % 100 == 0:
                    path.write_text(json.dumps(have), encoding="utf-8")
                    log(f"{coin} iv {i}/{len(todo)}")
            path.write_text(json.dumps(have), encoding="utf-8")
            keys = A.list_all(f"data/option/daily/EOHSummary/{under}/", delimiter=False)
            eoh = [k for k in keys if k.endswith(".zip") and k[-14:-4] <= LAST_FILE]
            path2 = STORE / f"{coin}_spread.json"
            sp = json.loads(path2.read_text(encoding="utf-8")) if path2.exists() else {}
            todo2 = [k for k in eoh if k[-14:-4] not in sp]

            def two(k):
                raw = A.http_get(f"https://{A.FILES}/{k}")
                rows = spread_rows(A._zip_text(raw)) if raw else []
                return k[-14:-4], day_spread(rows), len(rows)

            for d, w, n in ex.map(two, todo2):
                sp[d] = {"width": w, "n": n}
            path2.write_text(json.dumps(sp), encoding="utf-8")
            tot[coin] = {"iv_days": len(have), "iv_values": sum(1 for v in have.values() if v["iv"] is not None), "eoh_days": len(sp), "eoh_with_rows": sum(1 for v in sp.values() if v["width"] is not None)}
    log(json.dumps(tot))
    return tot


def _perp(under: str):
    import funding_store as FS
    with np.load(FS.perp_path(under)) as z:
        t = z["t"].astype(np.int64)
        k = t <= _ms(LAST_FILE) + DAY - BAR
        return t[k], z["c"][k].astype(float)


def study(coin: str) -> dict:
    sym, under = COINS[coin]
    iv = json.loads((STORE / f"{coin}_iv.json").read_text(encoding="utf-8"))
    sp = json.loads((STORE / f"{coin}_spread.json").read_text(encoding="utf-8"))
    t, c = _perp(under)
    rows, no_iv, no_rv = [], [], 0
    for d in _days(FIRST, LAST_IV):
        v = iv.get(d, {}).get("iv")
        if v is None:
            no_iv.append(d)
            continue
        r = rv30(t, c, _ms(d))
        if r is None:
            no_rv += 1
            continue
        rows.append((d, v, r))
    ivs, rvs = np.array([x[1] for x in rows]), np.array([x[2] for x in rows])
    gap = ivs - rvs
    var = ivs ** 2 - rvs ** 2
    every = gap[::30]
    widths = [v["width"] for v in sp.values() if v["width"] is not None]
    spread_cost = float(np.median(widths)) if widths else None
    hedge = leland_points(float(ivs.mean()) / 100)
    total = spread_cost + hedge if spread_cost is not None else None
    by_year = {}
    for (d, _, _), g in zip(rows, gap):
        by_year.setdefault(d[:4], []).append(g)
    worst = sorted(zip(gap, [x[0] for x in rows]))[:5]
    ratio = float(ivs.mean() / rvs.mean())
    out = {"coin": coin, "n": len(rows), "window": [rows[0][0], rows[-1][0]], "iv_mean": float(ivs.mean()), "rv_mean": float(rvs.mean()), "iv_rv_ratio": ratio, "index_assumption_ok": 0.5 <= ratio <= 2.0,
           "corr": float(np.corrcoef(ivs, rvs)[0, 1]), "gap": newey_west(gap, LAG), "gap_median": float(np.median(gap)), "share_iv_above": float((gap > 0).mean()), "gap_by_year": {y: float(np.mean(v)) for y, v in sorted(by_year.items())},
           "variance_gap": newey_west(var, LAG), "non_overlap": {"n": int(len(every)), "mean": float(every.mean()), "se": float(every.std(ddof=1) / math.sqrt(len(every)))},
           "worst_windows": [(d, float(g)) for g, d in worst], "max_rv_above_iv": float(-gap.min()), "days_without_iv": no_iv, "known_gaps_confirmed": [d for d in KNOWN_GAPS if d in no_iv], "days_without_rv": no_rv,
           "cost": {"spread_median_points": spread_cost, "spread_days": len(widths), "spread_period": [min(sp), max(sp)] if sp else None, "delta_hedge_leland_points": hedge, "total_points": total}}
    out["stop"] = bool(total is None or out["gap"]["mean"] < total)
    return out


def run() -> dict:
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    res = {c: study(c) for c in COINS}
    out = {"spec": SPEC, "stage": "descriptive", "window": [FIRST, LAST_IV], "coins": res, "decision": {c: ("STOP: mean IV - RV is below a round trip" if res[c]["stop"] else "report to the operator: options need a separate decision") for c in res},
           "not_modelled": ["option exchange fee", "the spread measured on 2023-05..10 only, carried over the window", "margin and liquidation of a short option", "vega and gamma path, rehedging more often than daily", "index formula details given as images on the page"],
           "verdict": None, "traps_checked": [1, 4, 8, 10], "traps_note": {"1": "RV is strictly after IV", "4": "descriptive, no power claim", "8": "mainnet archive only"}, "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 2)}
    (EXP / "N5_VRP.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["coins"], indent=0, default=str)[:5000])
    print(out["decision"])
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fetch"]:
        fetch(log=lambda s: print(s, flush=True))
    elif argv[:1] == ["run"]:
        run()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
