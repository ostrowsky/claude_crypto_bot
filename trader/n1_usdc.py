"""N1-A/B/C -- USDC perps: inventory, funding against the USDT version, liquidity, the USDC/USDT rate (n1-1106-spec.md).

Descriptive: no verdict, no alpha, no held-out window. Nothing after 2025-12 is requested or read.

  inventory  live and demo exchangeInfo (public GET), rank of the USDT version on 2026-01-01 from perp_history
  fetch      funding and 1d of both versions, USDCUSDT 1m, into runtime/usdc/
  funding    daily sum of USDC - USDT, pool of coins, BTC and ETH apart
  liquidity  quote volume and trades, admission to N2 / N3 / N4
  basis      |1 - rate| distribution and episodes above 0.1% and 0.5%

  (from trader/) ../pyembed/python.exe -u n1_usdc.py inventory | fetch | run
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import archive_bh as A            # noqa: E402
import n_common as NC             # noqa: E402
import trader_config as TC        # noqa: E402

SPEC = "n1-1106-spec.md"
STORE = TC.RUNTIME / "usdc"
EXP = TC.RUNTIME / "experiments"
LAST_MONTH = "2025-12"
LAST_DAY = "2025-12-31"
DELTA = 0.002
BLOCK = 20
ADMIT_MEDIAN = 20e6
ADMIT_MIN_DAYS = 60
ADMIT_FIRST = "2025-10-03"
RANK_DAY = "2026-01-01"
DELISTED = ("MATICUSDC",)
DAY_MS = 86_400_000
MIN_MS = 60_000
TRAPS = [3, 4, 7, 9, 10]


def _ms_of(raw: str) -> int:
    t = int(raw)
    return t // 1000 if t > 10 ** 14 else t


def _rows(text: str):
    for r in csv.reader(io.StringIO(text)):
        if r and r[0].strip().lstrip("-").isdigit():
            yield r


def parse_daily(text: str) -> dict:
    by = {_ms_of(r[0]): (float(r[7]), int(float(r[8]))) for r in _rows(text)}
    t = sorted(by)
    return {"t": np.array(t, dtype=np.int64), "qv": np.array([by[x][0] for x in t], dtype=float),
            "count": np.array([by[x][1] for x in t], dtype=np.int64)}


def parse_funding_full(text: str) -> dict:
    by = {}
    for r in _rows(text):
        hours = float(r[1]) if len(r) >= 3 else float("nan")
        by[_ms_of(r[0])] = (float(r[-1]), hours)
    t = sorted(by)
    return {"t": np.array(t, dtype=np.int64), "rate": np.array([by[x][0] for x in t], dtype=float),
            "hours": np.array([by[x][1] for x in t], dtype=float)}


def _day(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def daily_sums(t, rate, hours) -> dict:
    """% per UTC day, complete days only: intervals add to [20, 28] h, or >= 2 events when the file has no interval."""
    by = {}
    for ti, r, h in zip(t, rate, hours):
        by.setdefault(_day(int(ti)), []).append((float(r), float(h)))
    out = {}
    for d, v in by.items():
        hs = [h for _, h in v]
        ok = len(v) >= 2 if any(math.isnan(h) for h in hs) else 20 <= sum(hs) <= 28
        if ok:
            out[d] = sum(r for r, _ in v) * 100
    return out


def pool(pairs: dict) -> list:
    by = {}
    for x in pairs.values():
        for d, v in x["usdc"].items():
            if d >= x["from"] and d in x["usdt"]:
                by.setdefault(d, []).append(v - x["usdt"][d])
    return [(d, round(float(np.mean(by[d])), 12)) for d in sorted(by)]


def months_upto(keys, last: str = LAST_MONTH) -> list:
    return A.months_in_range(keys, "2019-01", last)


def usdt_of(sym: str) -> str:
    return sym[:-4] + "USDT"


def episodes(t, x, thr: float, gap_min: int = 5) -> list:
    t = np.asarray(t, dtype=np.int64)
    dev = np.abs(1 - np.asarray(x, dtype=float))
    out = []
    for ti, d in zip(t[dev > thr], dev[dev > thr]):
        if out and (ti - out[-1]["_last"]) / MIN_MS - 1 <= gap_min:
            out[-1]["_last"] = int(ti)
            out[-1]["peak"] = max(out[-1]["peak"], float(d))
        else:
            out.append({"_first": int(ti), "_last": int(ti), "peak": float(d)})
    return [{"start": datetime.fromtimestamp(e["_first"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
             "minutes": (e["_last"] - e["_first"]) // MIN_MS + 1, "peak": e["peak"]} for e in out]


def admit(days: dict, first_day: str, launch_day: str) -> dict:
    v = [q for d, q in days.items() if first_day <= d <= LAST_DAY]
    med = float(np.median(v)) if v else 0.0
    return {"median": med, "n_days": len(v), "launched_before_window": launch_day <= first_day,
            "admitted": bool(v and med >= ADMIT_MEDIAN and len(v) >= ADMIT_MIN_DAYS and launch_day <= first_day)}


# ------------------------------------------------------------------ inventory
def _get(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trader-n1/1"}), timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _filter(sym: dict, kind: str) -> dict:
    return next((f for f in sym.get("filters", []) if f.get("filterType") == kind), {})


def rank_table(day: str = RANK_DAY) -> dict:
    """Rank of every USDT perp on `day` by quote volume over the 30 days before it (close * volume on 15m, no later bar)."""
    d1 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lo, hi = int((d1 - timedelta(days=30)).timestamp() * 1000), int(d1.timestamp() * 1000)
    age_ms = lo
    rows = {}
    for p in (TC.RUNTIME / "perp_history").glob("*_15m.npz"):
        with np.load(p) as z:
            t, c, v = z["t"].astype(np.int64), z["c"].astype(float), z["v"].astype(float)
        if not len(t) or t[0] > age_ms:
            continue
        m = (t >= lo) & (t < hi)
        if len({int(x) // DAY_MS for x in t[m]}) >= 25:
            rows[p.name[:-len("_15m.npz")]] = float(np.nansum(c[m] * v[m]))
    return {s: i + 1 for i, s in enumerate(sorted(rows, key=lambda s: -rows[s]))}


def inventory() -> dict:
    import funding_store as FS
    import provenance as PV
    PV.require_clean()
    ex = _get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    demo = _get("https://demo-fapi.binance.com/fapi/v1/exchangeInfo")
    info = _get("https://fapi.binance.com/fapi/v1/fundingInfo")
    interval = {x["symbol"]: x.get("fundingIntervalHours") for x in info}
    demo_syms = {s["symbol"]: s for s in demo["symbols"]}
    live_syms = {s["symbol"]: s for s in ex["symbols"]}
    ranks = rank_table()
    mapping = json.loads(FS.MAPPING.read_text(encoding="utf-8")) if FS.MAPPING.exists() else {}
    perp_to_spot = {v.get("perp"): k for k, v in mapping.items()}
    rows, missing = [], []
    syms = [s for s in ex["symbols"] if s.get("marginAsset") == "USDC" and s.get("contractType") == "PERPETUAL"]
    for s in sorted(syms, key=lambda x: x["symbol"]):
        u = usdt_of(s["symbol"])
        if u not in live_syms:
            missing.append(s["symbol"])
        spot = perp_to_spot.get(u)
        rows.append({"symbol": s["symbol"], "status": s["status"], "onboard": _day(s["onboardDate"]), "usdt": u, "usdt_listed": u in live_syms,
                     "usdt_rank_2026_01_01": ranks.get(spot) if spot else None, "spot_of_usdt": spot,
                     "tick": float(_filter(s, "PRICE_FILTER").get("tickSize", 0)), "min_notional": float(_filter(s, "MIN_NOTIONAL").get("notional", 0)),
                     "funding_interval_h": interval.get(s["symbol"], 8), "interval_from_fundingInfo": s["symbol"] in interval,
                     "on_demo": s["symbol"] in demo_syms, "demo_status": demo_syms.get(s["symbol"], {}).get("status")})
    out = {"spec": SPEC, "stage": "descriptive", "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rows": rows, "n": len(rows),
           "no_usdt_version": missing, "delisted_in_archive_only": list(DELISTED), "demo_usdc_count": sum(1 for r in rows if r["on_demo"]),
           "verdict": None, "traps_checked": [10], "provenance": PV.block()}
    EXP.mkdir(exist_ok=True)
    (EXP / "N1_INVENTORY.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------ data
def _fetch_file(prefix: str, name: str, parser, keys, ex, first_month: str = "2019-01") -> dict:
    """Months <= LAST_MONTH of a prefix, one npz, resumable through a months file."""
    STORE.mkdir(parents=True, exist_ok=True)
    out = STORE / f"{name}.npz"
    done_path = out.with_suffix(".months.json")
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    ks = A.list_all(prefix, delimiter=False)
    months = [m for m in A.months_in_range(ks, first_month, LAST_MONTH)]
    by_month = {}
    for k in ks:
        for m in months:
            if k.endswith(f"-{m}.zip"):
                by_month[m] = k
    todo = [m for m in months if m not in done]

    def one(m):
        raw = A.http_get(f"https://{A.FILES}/{by_month[m]}")
        return m, parser(A._zip_text(raw)) if raw else None

    parts = list(ex.map(one, todo))
    old = None
    if out.exists():
        with np.load(out) as z:
            old = {k: z[k] for k in z.files}
    arrays = ([old] if old else []) + [p for _, p in parts if p is not None]
    if arrays:
        A._save(out, arrays, keys)
    done.update({m: bool(p is not None) for m, p in parts})
    done_path.write_text(json.dumps(done), encoding="utf-8")
    return {"months": len(months), "new": len(todo), "missing": [m for m, p in parts if p is None]}


def symbol_list() -> list:
    ex = _get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    live = [s["symbol"] for s in ex["symbols"] if s.get("marginAsset") == "USDC" and s.get("contractType") == "PERPETUAL"]
    return sorted(set(live) | set(DELISTED))


def fetch(log=print) -> dict:
    ex = ThreadPoolExecutor(8)
    tot = {"symbols": 0, "months": 0, "new": 0, "missing": 0}
    for sym in symbol_list():
        for s in (sym, usdt_of(sym)):
            for kind, prefix, parser, keys in (
                    ("funding", f"data/futures/um/monthly/fundingRate/{s}/", parse_funding_full, ("rate", "hours")),
                    ("1d", f"data/futures/um/monthly/klines/{s}/1d/", parse_daily, ("qv", "count"))):
                r = _fetch_file(prefix, f"{s}_{kind}", parser, keys, ex, first_month="2023-12")
                tot["months"] += r["months"]
                tot["new"] += r["new"]
                tot["missing"] += len(r["missing"])
        tot["symbols"] += 1
        log(f"{sym} done")
    r = _fetch_file("data/spot/monthly/klines/USDCUSDT/1m/", "USDCUSDT_1m", A.parse_klines, ("o", "h", "l", "c", "v"), ex, first_month="2023-01")
    tot["usdcusdt"] = r
    ex.shutdown()
    log(json.dumps(tot))
    return tot


def _load(name: str):
    p = STORE / f"{name}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _integrity(a) -> dict:
    t = a["t"]
    d = np.diff(t)
    return {"n": int(len(t)), "monotonic": bool((d > 0).all()), "max_year": _day(int(t.max()))[:4] if len(t) else None,
            "unit_ok": bool(len(t) == 0 or t.max() < 10 ** 14)}


# ------------------------------------------------------------------ analyses
def _launch(sym: str, inv: dict) -> str:
    row = next((r for r in inv["rows"] if r["symbol"] == sym), None)
    if row:
        return row["onboard"]
    f = _load(f"{sym}_funding")
    return _day(int(f["t"][0])) if f is not None and len(f["t"]) else "9999-12-31"


def _power(sigma: float, n_obs: int, delta: float = DELTA) -> dict:
    return {"delta": delta, "sigma": sigma, "n_needed_iid": NC.required_n(sigma, delta, 0.05)[0], "n_available": n_obs,
            "note": "reference only: not a held-out window, days are autocorrelated"}


def funding_report(inv: dict, syms: list) -> dict:
    pairs, dropped, per = {}, {}, {}
    for s in syms:
        a, b = _load(f"{s}_funding"), _load(f"{usdt_of(s)}_funding")
        if a is None or b is None:
            dropped[s] = "no funding file for one version"
            continue
        launch = _launch(s, inv)
        start = (datetime.strptime(launch, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        c, u = daily_sums(a["t"], a["rate"], a["hours"]), daily_sums(b["t"], b["rate"], b["hours"])
        n_all_c = len({_day(int(x)) for x in a["t"]})
        n_all_u = len({_day(int(x)) for x in b["t"]})
        c = {d: v for d, v in c.items() if "2024-01-01" <= d <= LAST_DAY}
        u = {d: v for d, v in u.items() if "2024-01-01" <= d <= LAST_DAY}
        pairs[s] = {"usdc": c, "usdt": u, "from": start}
        series = sorted((d, c[d] - u[d]) for d in c if d >= start and d in u)
        if len(series) >= 10:
            g = NC.ci([x for _, x in series], block=BLOCK)
            per[s] = {"launch": launch, "from": start, "n_days": len(series), "usdc_mean": float(np.mean([c[d] for d, _ in series])),
                      "usdt_mean": float(np.mean([u[d] for d, _ in series])), "diff": g, "sigma": float(np.std([x for _, x in series], ddof=1)),
                      "dropped_incomplete_days": {"usdc": n_all_c - len(daily_sums(a["t"], a["rate"], a["hours"])), "usdt": n_all_u - len(daily_sums(b["t"], b["rate"], b["hours"]))},
                      "transfer_forbidden": bool(g["hi"] is not None and g["hi"] < -DELTA)}
        else:
            dropped[s] = f"{len(series)} common complete days"
    series = pool({k: v for k, v in pairs.items() if k in per})
    blocks = {b: NC.ci([x for _, x in series], block=b) for b in (5, 20, 60)}
    sigma = float(np.std([x for _, x in series], ddof=1))
    out = {"spec": SPEC, "stage": "descriptive", "window": [series[0][0], series[-1][0]], "metric": "daily funding USDC - USDT, %/day", "delta": DELTA,
           "pool": blocks[20], "pool_blocks": blocks, "pool_by_year": _year_means(series),
           "pool_transfer_forbidden": bool(blocks[20]["hi"] < -DELTA), "per_symbol": per, "dropped": dropped, "BTC": per.get("BTCUSDC"), "ETH": per.get("ETHUSDC"),
           "power": _power(sigma, len(series)), "verdict": None, "traps_checked": [4, 7, 10]}
    return out


def _year_means(series) -> dict:
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: float(np.mean(v)) for y, v in sorted(by.items())}


def liquidity_report(inv: dict, syms: list) -> dict:
    rows = {}
    for s in syms:
        a, b = _load(f"{s}_1d"), _load(f"{usdt_of(s)}_1d")
        if a is None:
            rows[s] = {"admitted": False, "reason": "no 1d file"}
            continue
        d_c = {_day(int(t)): float(q) for t, q in zip(a["t"], a["qv"])}
        d_n = {_day(int(t)): int(q) for t, q in zip(a["t"], a["count"])}
        adm = admit(d_c, ADMIT_FIRST, _launch(s, inv))
        days = [d for d in d_c if ADMIT_FIRST <= d <= LAST_DAY]
        row = {**adm, "median_trades": float(np.median([d_n[d] for d in days])) if days else None}
        if b is not None:
            d_u = {_day(int(t)): float(q) for t, q in zip(b["t"], b["qv"])}
            d_un = {_day(int(t)): int(q) for t, q in zip(b["t"], b["count"])}
            common = [d for d in days if d in d_u]
            row["usdt_median_qv"] = float(np.median([d_u[d] for d in common])) if common else None
            row["usdt_median_trades"] = float(np.median([d_un[d] for d in common])) if common else None
            row["share_of_usdt"] = row["median"] / row["usdt_median_qv"] if row["usdt_median_qv"] else None
        inv_row = next((r for r in inv["rows"] if r["symbol"] == s), {})
        row["usdt_rank_2026_01_01"] = inv_row.get("usdt_rank_2026_01_01")
        rows[s] = row
    admitted = sorted((s for s, r in rows.items() if r.get("admitted")), key=lambda s: -rows[s]["median"])
    return {"spec": SPEC, "stage": "descriptive", "window": [ADMIT_FIRST, LAST_DAY], "threshold_usdc": ADMIT_MEDIAN, "rows": rows, "admitted": admitted,
            "verdict": None, "traps_checked": [10]}


def basis_report() -> dict:
    a = _load("USDCUSDT_1m")
    t, c = a["t"], a["c"]
    m = (t >= int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)) & (t < int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
    t, c = t[m], c[m]
    dev = np.abs(1 - c)
    q = lambda x: {k: float(np.percentile(x, p)) for k, p in (("p50", 50), ("p90", 90), ("p99", 99), ("p999", 99.9))} | {"max": float(x.max()), "n": int(len(x))}
    years = {}
    for y in (2023, 2024, 2025):
        lo = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        hi = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        k = (t >= lo) & (t < hi)
        years[str(y)] = q(dev[k])
    expected = int((t.max() - t.min()) / MIN_MS) + 1
    eps = {}
    for thr in (0.001, 0.005):
        e = episodes(t, c, thr)
        mins = [x["minutes"] for x in e]
        eps[str(thr)] = {"n": len(e), "minutes_total": int(sum(mins)), "median_minutes": float(np.median(mins)) if mins else 0, "longest": sorted(e, key=lambda x: -x["minutes"])[:10],
                         "by_year": {y: sum(1 for x in e if x["start"][:4] == y) for y in ("2023", "2024", "2025")}}
    signed = 1 - c
    return {"spec": SPEC, "stage": "descriptive", "window": ["2023-01-01", LAST_DAY], "distribution": q(dev), "by_year": years, "episodes": eps,
            "mean_signed_1_minus_rate": float(signed.mean()), "missing_minutes": expected - int(len(t)), "missing_share": 1 - len(t) / expected,
            "integrity": _integrity(a), "verdict": None, "traps_checked": [9, 10]}


def run() -> dict:
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    inv = json.loads((EXP / "N1_INVENTORY.json").read_text(encoding="utf-8"))
    syms = sorted({r["symbol"] for r in inv["rows"]} | set(DELISTED))
    fr = funding_report(inv, syms)
    lq = liquidity_report(inv, syms)
    bs = basis_report()
    integ = {}
    for s in syms:
        for name in (f"{s}_funding", f"{usdt_of(s)}_funding", f"{s}_1d", f"{usdt_of(s)}_1d"):
            a = _load(name)
            if a is not None:
                integ[name] = _integrity(a)
    bad = {k: v for k, v in integ.items() if not (v["monotonic"] and v["unit_ok"]) or (v["max_year"] or "0") > "2025"}
    for name, out in (("N1_FUNDING", fr), ("N1_LIQUIDITY", lq), ("N1_USDC_BASIS", bs)):
        out["integrity_bad"] = bad
        out["provenance"] = PV.block()
        (EXP / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pool": fr["pool"], "pool_by_year": fr["pool_by_year"], "forbidden": fr["pool_transfer_forbidden"], "BTC": fr["BTC"], "ETH": fr["ETH"],
                      "power": fr["power"], "dropped": fr["dropped"], "admitted": lq["admitted"], "basis": bs["distribution"], "by_year": bs["by_year"],
                      "episodes": {k: {x: v[x] for x in ("n", "minutes_total", "median_minutes", "by_year")} for k, v in bs["episodes"].items()},
                      "longest_0.5": bs["episodes"]["0.005"]["longest"][:5], "missing": bs["missing_share"], "integrity_bad": bad, "sec": round(time.time() - t0)}, indent=1, default=str))
    return fr


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["inventory"]:
        r = inventory()
        print(json.dumps({k: r[k] for k in ("n", "no_usdt_version", "demo_usdc_count")}))
        for x in r["rows"]:
            print(x["symbol"], x["status"], x["onboard"], "rank", x["usdt_rank_2026_01_01"], "tick", x["tick"], "minN", x["min_notional"], "h", x["funding_interval_h"], "demo", x["on_demo"])
    elif argv[:1] == ["fetch"]:
        fetch(log=lambda s: print(s, flush=True))
    elif argv[:1] == ["run"]:
        run()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
