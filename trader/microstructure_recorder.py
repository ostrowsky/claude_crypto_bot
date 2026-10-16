"""A0 -- record spread, depth cost and demo-vs-live divergence (demo-connection-a-1010-spec.md).

WHY
history/*_15m*.csv holds OHLCV only: no spread, no book depth, no quote volume.
Without them the E1 replay cannot price slippage, and without comparing the demo
book with the live market nobody can say whether demo fills mean anything. CLAUDE.md
§0 rule 1: missing data is collected first. Public endpoints only -- no key.

ONE SNAPSHOT (run every MICRO_RECORD_INTERVAL_SEC by the scheduler or `loop`)
  book     bookTicker of every candidate pair on BOTH hosts: bid, ask, mid, spread bps
  divergence  (mid_demo - mid_live) / mid_live in bps, per pair present on both
  depth    a rotating slice of pairs (whole universe once an hour), both hosts:
           cost of a market BUY / SELL of 100 / 500 / 1000 / 5000 USDT, bps from mid
  24h      once an hour: quoteVolume and trade count per pair, both hosts
Output: trader/runtime/microstructure/YYYY-MM-DD.jsonl, append-only, one line per snapshot.
Weight budget: MICRO_WEIGHT_BUDGET_SHARE of the REQUEST_WEIGHT limit read from
exchangeInfo; 429 -> sleep Retry-After; 418 -> stop until the next hour.

  ../pyembed/python.exe microstructure_recorder.py snapshot
  ../pyembed/python.exe microstructure_recorder.py loop
  ../pyembed/python.exe microstructure_recorder.py summary [--days 7]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import exchange_client as X  # noqa: E402
import trader_config as TC  # noqa: E402

OUT_DIR = TC.RUNTIME / "microstructure"
STATE = OUT_DIR / "_state.json"
SIZES = (100, 500, 1000, 5000)
HOSTS = {"live": X.LIVE_REST, "demo": X.DEMO_REST}
DEPTH_LIMIT = 20
EXCHANGE_INFO_MAX_AGE_SEC = 86400
EXCHANGE_INFO_CACHE = TC.RUNTIME / "trading" / "exchange_info.json"
DEPTH_DEADLINE_SEC = 120   # whole depth pass; a stalled connection must not eat the 5-minute slot


def walk_cost(levels, notional: float, mid: float, side: str):
    """bps from mid paid by a market order of `notional` USDT walking `levels` [[price, qty], ...].
    None if the visible book is too thin."""
    if mid <= 0:
        return None
    left, cost_q, got = float(notional), 0.0, 0.0
    for p, q in levels:
        p, q = float(p), float(q)
        if p <= 0 or q <= 0:
            continue
        take_quote = min(left, p * q)
        got += take_quote / p
        cost_q += take_quote
        left -= take_quote
        if left <= 1e-9:
            break
    if left > 1e-9 or got <= 0:
        return None
    vwap = cost_q / got
    return round((vwap / mid - 1) * 1e4 if side == "buy" else (1 - vwap / mid) * 1e4, 3)


def candidates(xinfo: dict) -> list:
    import trading_universe as U
    u = U.build(xinfo, hist=lambda s: 10 ** 6)            # structural rules only
    return sorted(p["symbol"] for p in u["pairs"])


IP_CAP_SHARE = 0.80        # never push the IP total (shared with the live bot) above this share of the limit


def weight_budget(xinfo: dict) -> tuple:
    """(own budget per snapshot, IP-total cap): MICRO_WEIGHT_BUDGET_SHARE and IP_CAP_SHARE
    of the per-minute REQUEST_WEIGHT limit read from exchangeInfo."""
    share = float(TC.MICRO_WEIGHT_BUDGET_SHARE)
    limit = 6000
    for r in xinfo.get("rateLimits", []):
        if r.get("rateLimitType") == "REQUEST_WEIGHT" and r.get("interval") == "MINUTE":
            limit = int(r.get("limit", limit))
    return int(share * limit), int(IP_CAP_SHARE * limit)


def cached_exchange_info(client, path: Path | None = None, max_age: int | None = None) -> dict:
    """exchangeInfo at most once a day: a disk cache younger than max_age is used without any
    request; a missing, stale or broken cache is refetched and rewritten (spec: hang fix, req. 2)."""
    path = path or EXCHANGE_INFO_CACHE
    max_age = EXCHANGE_INFO_MAX_AGE_SEC if max_age is None else max_age
    try:
        c = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(c["fetched_at"]) < max_age and isinstance(c.get("data"), dict):
            return c["data"]
    except Exception:
        pass
    data = client.exchange_info(max_age_sec=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fetched_at": time.time(), "data": data}), encoding="utf-8")
    tmp.replace(path)
    return data


def _state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"depth_offset": 0, "last_24h_hour": None, "banned_until": 0}


def snapshot(client=None, now=None, rotate_slices: int = 12) -> dict:
    client = client or X.ExchangeClient(key="", secret="")   # public endpoints only, no key needed
    now = now or datetime.now(timezone.utc)
    st = _state()
    if time.time() < st.get("banned_until", 0):
        return {"skipped": "418 ban window"}
    rec = {"ts": now.isoformat(timespec="seconds"), "hosts": {}}
    t_start = time.time()
    xinfo = cached_exchange_info(client)
    syms = candidates(xinfo)
    budget, ip_cap = weight_budget(xinfo)
    want = set(syms)
    books = {}
    for name, base in HOSTS.items():
        try:
            rows = client.book_tickers(base=base)
        except X.ExchangeError as e:
            if e.status == 418:
                st["banned_until"] = time.time() + 3600
            rec["hosts"][name] = {"error": f"{e.status} {e.code}"}
            continue
        b = {}
        for r in rows:
            s = r.get("symbol")
            if s not in want:
                continue
            bid, ask = float(r.get("bidPrice") or 0), float(r.get("askPrice") or 0)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                b[s] = [round(mid, 10), round((ask - bid) / mid * 1e4, 3)]
        books[name] = b
        rec["hosts"][name] = {"book": b, "n": len(b), "used_weight_1m": client.used_weight.get(X._host(base))}
    if "live" in books and "demo" in books:
        rec["divergence_bps"] = {s: round((books["demo"][s][0] / books["live"][s][0] - 1) * 1e4, 3)
                                 for s in books["demo"] if s in books["live"] and books["live"][s][0] > 0}
    # rotating depth slice
    k = max(1, math.ceil(len(syms) / rotate_slices))
    off = st.get("depth_offset", 0) % max(1, len(syms))
    slice_ = (syms[off:] + syms[:off])[:k]
    st["depth_offset"] = off + k
    depth = {}
    for name, base in HOSTS.items():
        d = {}
        host = X._host(base)
        start = client.used_weight.get(host, 0)
        for s in slice_:
            if time.time() - t_start > DEPTH_DEADLINE_SEC:
                rec.setdefault("deadline_stops", []).append({"host": name, "done": len(d)})
                break
            used = client.used_weight.get(host, 0)
            if used - start >= budget or used >= ip_cap:      # own spend, or the IP total the bot shares
                rec.setdefault("budget_stops", []).append({"host": name, "used_1m": used, "own": used - start})
                break
            mid = (books.get(name) or {}).get(s, [0])[0]
            if not mid:
                continue
            try:
                ob = client.depth(s, DEPTH_LIMIT, base=base)
            except X.ExchangeError as e:
                if e.status in (418, 429):
                    break
                continue
            d[s] = {"buy": {n: walk_cost(ob.get("asks", []), n, mid, "buy") for n in SIZES},
                    "sell": {n: walk_cost(ob.get("bids", []), n, mid, "sell") for n in SIZES}}
        depth[name] = d
    rec["depth"] = depth
    # 24h once an hour
    hour = now.strftime("%Y-%m-%dT%H")
    if st.get("last_24h_hour") != hour:
        v = {}
        for name, base in HOSTS.items():
            try:
                rows = client.tickers_24h(base=base)
            except X.ExchangeError:
                continue
            v[name] = {r["symbol"]: [round(float(r.get("quoteVolume") or 0), 2), int(r.get("count") or 0)]
                       for r in rows if r.get("symbol") in want}
        rec["vol24h"] = v
        st["last_24h_hour"] = hour
    rec["elapsed_sec"] = round(time.time() - t_start, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_DIR / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    STATE.write_text(json.dumps(st), encoding="utf-8")
    return rec


def summary(days: int = 7) -> dict:
    """Per pair over the last `days` files: median spread (demo/live), p95 |divergence|,
    p95 cost of a 1000 USDT buy (demo), median live 24h quote volume."""
    files = sorted(OUT_DIR.glob("20*.jsonl"))[-days:]
    acc = {}

    def add(sym, key, v):
        if v is not None:
            acc.setdefault(sym, {}).setdefault(key, []).append(v)
    n_snap = 0
    for fp in files:
        for line in io.open(fp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_snap += 1
            for h in ("live", "demo"):
                for s, (mid, spr) in ((r.get("hosts", {}).get(h) or {}).get("book") or {}).items():
                    add(s, f"spread_{h}", spr)
            for s, dv in (r.get("divergence_bps") or {}).items():
                add(s, "abs_div", abs(dv))
            for s, d in ((r.get("depth") or {}).get("demo") or {}).items():
                add(s, "impact_demo_1000", (d.get("buy") or {}).get("1000", (d.get("buy") or {}).get(1000)))
            for s, (qv, _) in ((r.get("vol24h") or {}).get("live") or {}).items():
                add(s, "qvol_live", qv)

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))] if v else None
    out = {s: {"spread_demo_med": q(a.get("spread_demo", []), .5), "spread_live_med": q(a.get("spread_live", []), .5),
               "abs_div_p95": q(a.get("abs_div", []), .95), "impact_demo_1000_p95": q(a.get("impact_demo_1000", []), .95),
               "qvol_live_med": q(a.get("qvol_live", []), .5), "n": len(a.get("spread_demo", []))}
           for s, a in acc.items()}
    return {"files": [f.name for f in files], "snapshots": n_snap, "pairs": out}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "loop", "summary"])
    ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args(argv)
    import audit
    audit.set_process("recorder")
    if a.cmd in ("snapshot", "loop") and not TC.MICROSTRUCTURE_RECORDER_ENABLED:
        print("MICROSTRUCTURE_RECORDER_ENABLED is False -- nothing recorded")
        return 0
    if a.cmd == "snapshot":
        # watchdog: a stalled connection (DNS resolution is not covered by the socket timeout) once
        # kept a snapshot alive > 10 min and, holding the shared log open, blocked later runs.
        # Past 200 s the stack is printed (where it hung) and the process exits.
        import faulthandler
        faulthandler.dump_traceback_later(200, exit=True)
    if a.cmd == "summary":
        s = summary(a.days)
        print(json.dumps({"files": s["files"], "snapshots": s["snapshots"], "pairs": len(s["pairs"])}))
        return 0
    client = X.ExchangeClient(key="", secret="")
    while True:
        t0 = time.time()
        r = snapshot(client)
        h = r.get("hosts", {})
        print(r.get("ts"), {k: v.get("n", v.get("error")) for k, v in h.items()},
              "div pairs", len(r.get("divergence_bps") or {}), "depth", {k: len(v) for k, v in (r.get("depth") or {}).items()},
              flush=True)
        if a.cmd == "snapshot":
            return 0
        time.sleep(max(5, int(TC.MICRO_RECORD_INTERVAL_SEC) - (time.time() - t0)))


if __name__ == "__main__":
    raise SystemExit(main())
