"""Entry price gap: the bot's event price (close of the last CLOSED bar) vs the market at the signal.

entry-price-gap-1013-spec.md. Every bot entry since 03-01 (execution_replay.load_events, the E1/H3
sample) gets the spot 1-minute candle containing its timestamp (api.binance.com, the bot's own
market, unsigned): gap = p_mid / price - 1, p_mid = (open + close) / 2 of that minute. Then E1 runs
the champion and X4 with the entry price replaced by p_mid, train and test halves, paired against
the unchanged run. Criteria are fixed in the spec (§4) before this runs.

  (from trader/) ../pyembed/python.exe -u experiment_entry_gap.py
Cache: trader/runtime/experiments/klines_1m/<sym>.jsonl (a rerun makes no request).
"""
from __future__ import annotations

import io
import json
import random
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

MIN_MS = 60_000
WINDOW_MIN = 1000
SPLIT = "2026-07-01"
MIN_SYSTEMATIC_BPS = 8.0
RATE_PER_SEC = 5.0
CACHE = TC.RUNTIME / "experiments" / "klines_1m"
OUT = TC.RUNTIME / "experiments" / "entry_price_gap_1013.json"


def _ms(d: datetime) -> int:
    return int(d.timestamp() * 1000)


def _minute_ms(d: datetime) -> int:
    return _ms(d) // MIN_MS * MIN_MS


def price_at(kl: dict, t: datetime):
    """(p_open, p_close, p_mid) of the 1m candle containing t, or None."""
    x = kl.get(_minute_ms(t))
    if x is None:
        return None
    o, c = float(x[0]), float(x[1])
    return (o, c, (o + c) / 2)


def bar_close(t: datetime, tf: str) -> datetime:
    """Close time of the last closed bar at t (= open of the forming bar)."""
    if tf == "1h":
        return t.replace(minute=0, second=0, microsecond=0)
    return t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)


def delay_sec(t: datetime, tf: str) -> float:
    return (t - bar_close(t, tf)).total_seconds()


def windows(entries) -> list:
    """[(sym, start_ms)]: 1000-minute windows per symbol, greedy, covering every entry minute."""
    by = {}
    for sym, t in entries:
        by.setdefault(sym, []).append(_minute_ms(t))
    out = []
    for sym in sorted(by):
        start = None
        for m in sorted(by[sym]):
            if start is None or m >= start + WINDOW_MIN * MIN_MS:
                start = m
                out.append((sym, start))
    return out


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(rows, seed: int = 29, boot: int = 2000) -> dict:
    """n, mean, median, p10/p90, share > 0, 95% CI of the mean by resampling whole UTC days."""
    xs = [float(r["gap_bps"]) for r in rows]
    if not xs:
        return {"n": 0}
    days = {}
    for r in rows:
        days.setdefault(r["day"], []).append(float(r["gap_bps"]))
    keys = sorted(days)
    rnd = random.Random(seed)
    means = []
    for _ in range(boot):
        s = n = 0
        for _k in keys:
            v = days[keys[rnd.randrange(len(keys))]]
            s += sum(v)
            n += len(v)
        means.append(s / n)
    means.sort()
    return {"n": len(xs), "days": len(keys), "mean": sum(xs) / len(xs), "median": statistics.median(xs),
            "p10": _q(xs, 0.10), "p90": _q(xs, 0.90), "pos_share": sum(1 for x in xs if x > 0) / len(xs),
            "lo95": means[int(0.025 * boot)], "hi95": means[int(0.975 * boot) - 1]}


def systematic(s: dict) -> bool:
    """Spec §4: the CI excludes 0 and |mean| >= 8 bps."""
    return (s["lo95"] > 0 or s["hi95"] < 0) and abs(s["mean"]) >= MIN_SYSTEMATIC_BPS


def trade_key(t: dict) -> str:
    ts = t["ts"].isoformat() if isinstance(t["ts"], datetime) else str(t["ts"])
    return f"{t['sym']}|{t.get('tf')}|{ts}"


def apply_gap(trades, gaps: dict):
    """Copies of trades with price = p_mid where known; (copies, number without a market price)."""
    out, missing = [], 0
    for t in trades:
        c = dict(t)
        g = gaps.get(trade_key(t))
        if g is None:
            missing += 1
        else:
            c["price"] = float(g)
        out.append(c)
    return out, missing


# ---------------------------------------------------------------- klines cache

def _load_cache(path: Path) -> dict:
    """{start_ms: {open_ms: (open, close)}} of the windows already fetched for one symbol."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:                                   # a run killed mid-append leaves a partial line
                r = json.loads(line)
                out[int(r["start"])] = {int(m): (float(o), float(c)) for m, o, c in r["rows"]}
            except (ValueError, KeyError, TypeError):
                continue
    return out


def fetch_window(client, sym: str, start_ms: int, cache_dir: Path) -> dict:
    path = Path(cache_dir) / f"{sym}.jsonl"
    have = _load_cache(path)
    if start_ms in have:
        return have[start_ms]
    rows = client.klines_from(sym, "1m", start_ms, WINDOW_MIN, base="https://api.binance.com")
    kl = {int(r[0]): (float(r[1]), float(r[4])) for r in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, 2)
            lead = "" if fh.read(1) == b"\n" else "\n"      # never glue onto a partial line
    with io.open(path, "a", encoding="utf-8") as fh:
        fh.write(lead + json.dumps({"start": start_ms, "rows": [[m, o, c] for m, (o, c) in sorted(kl.items())]}) + "\n")
    return kl


# ---------------------------------------------------------------- run

def measure(trades, client, cache_dir: Path = CACHE, log=print, workers: int = 4) -> list:
    """Fetch the 1m windows (one thread per symbol, a shared limit of RATE_PER_SEC requests) and
    compute the gap of every entry. The result does not depend on `workers`."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    ents = [(t["sym"], t["ts"]) for t in trades if t.get("price")]
    ws = windows(ents)
    log(f"entries {len(ents)}, windows {len(ws)}")
    by = {}
    for sym, st in ws:
        by.setdefault(sym, []).append(st)
    lock, gate, done = threading.Lock(), {"next": 0.0}, {"n": 0}

    def throttle():
        with lock:
            now = time.time()
            at = max(now, gate["next"])
            gate["next"] = at + 1.0 / RATE_PER_SEC
        if at > now:
            time.sleep(at - now)

    def one_symbol(sym):
        path = Path(cache_dir) / f"{sym}.jsonl"
        have = _load_cache(path)
        kl_s = {}
        for st in by[sym]:
            if st in have:
                kl_s.update(have[st])
            else:
                throttle()
                try:
                    kl_s.update(fetch_window(client, sym, st, cache_dir))
                except Exception as e:
                    log(f"  {sym} {st}: {e!r}"[:200])
            with lock:
                done["n"] += 1
                if done["n"] % 250 == 0:
                    log(f"  windows {done['n']}/{len(ws)}")
        return sym, kl_s

    kl = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for sym, k in pool.map(one_symbol, sorted(by)):
            kl[sym] = k
    rows = []
    for t in trades:
        if not t.get("price"):
            continue
        p = price_at(kl.get(t["sym"], {}), t["ts"])
        r = {"key": trade_key(t), "sym": t["sym"], "tf": t.get("tf"), "mode": t.get("mode"),
             "day": t["ts"].strftime("%Y-%m-%d"), "month": t["ts"].strftime("%Y-%m"), "price": float(t["price"]),
             "delay_sec": delay_sec(t["ts"], t.get("tf") or "15m")}
        if p is not None:
            r.update(p_open=p[0], p_close=p[1], p_mid=p[2], gap_bps=(p[2] / r["price"] - 1) * 1e4,
                     gap_open_bps=(p[0] / r["price"] - 1) * 1e4, gap_close_bps=(p[1] / r["price"] - 1) * 1e4)
        rows.append(r)
    return rows


def _slices(rows):
    ok = [r for r in rows if r.get("gap_bps") is not None]
    out = {"all": summary(ok),
           "open_minute": summary([dict(r, gap_bps=r["gap_open_bps"]) for r in ok]),
           "close_minute": summary([dict(r, gap_bps=r["gap_close_bps"]) for r in ok])}
    for name, f in (("tf", lambda r: r["tf"]), ("mode", lambda r: r["mode"]), ("month", lambda r: r["month"]),
                    ("delay", lambda r: "<60s" if r["delay_sec"] < 60 else ("60-180s" if r["delay_sec"] < 180 else ">180s"))):
        grp = {}
        for r in ok:
            grp.setdefault(str(f(r)), []).append(r)
        out[name] = {k: summary(v) for k, v in sorted(grp.items()) if len(v) >= 20}
    out["half"] = {"train": summary([r for r in ok if r["day"] < SPLIT]),
                   "test": summary([r for r in ok if r["day"] >= SPLIT])}
    out["missing"] = len(rows) - len(ok)
    return out


def money(trades, gaps, log=print) -> dict:
    import daily_revenue as DR
    import execution_replay as R
    import fill_model as FM
    import shadow_portfolio as S
    import trading_policy as TP
    grid = R._grid()
    R.attach_grid(trades, grid)
    gtr, missing = apply_gap(trades, gaps)
    fl, fm = R._floor(), FM.model("taker", 1)
    out = {"missing_market_price": missing}
    for name, pol in (("champion", TP.CHAMPION), ("x4_floor8", S.X4)):
        for half, lo, hi in (("train", None, SPLIT), ("test", SPLIT, None)):
            a = R.run(pol, fm, grid, trades, floor_of=fl, day_from=lo, day_to=hi, plan_cache={})
            b = R.run(pol, fm, grid, gtr, floor_of=fl, day_from=lo, day_to=hi, plan_cache={})
            out[f"{name}_{half}"] = {"event_price": a["agg"], "market_price": b["agg"],
                                     "paired_market_minus_event": DR.paired(b["series"], a["series"]),
                                     "trades": [len(a["trades"]), len(b["trades"])]}
            log(f"  {name} {half}: R {a['agg'].get('mean_pct')} -> {b['agg'].get('mean_pct')} %/day, paired "
                f"{out[f'{name}_{half}']['paired_market_minus_event'].get('mean_pct')} "
                f"[{out[f'{name}_{half}']['paired_market_minus_event'].get('lo95')}, "
                f"{out[f'{name}_{half}']['paired_market_minus_event'].get('hi95')}]")
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import audit as A
    import exchange_client as X
    import execution_replay as R
    A.set_process("entry_gap")
    t0 = time.time()
    trades, _ = R.load_events()
    print(f"trades {len(trades)} from {trades[0]['ts']:%Y-%m-%d} to {trades[-1]['ts']:%Y-%m-%d}")
    rows = measure(trades, X.ExchangeClient(key="", secret=""))
    sl = _slices(rows)
    s = sl["all"]
    verdict = {"systematic": systematic(s) if s.get("n") else None}
    print(f"gap all: n {s['n']}, mean {s['mean']:+.1f} bps [{s['lo95']:+.1f}, {s['hi95']:+.1f}], median "
          f"{s['median']:+.1f}, p10 {s['p10']:+.1f}, p90 {s['p90']:+.1f}, >0 {s['pos_share']:.0%}; missing {sl['missing']}")
    gaps = {r["key"]: r["p_mid"] for r in rows if r.get("p_mid") is not None}
    mon = money(trades, gaps)
    x4 = mon["x4_floor8_test"]["paired_market_minus_event"]
    verdict["money_material_x4_test"] = (x4.get("hi95") is not None and x4["hi95"] < 0)
    res = {"spec": "entry-price-gap-1013-spec.md", "run_utc": datetime.now(timezone.utc).isoformat(),
           "n_trades": len(trades), "gap": sl, "money": mon, "verdict": verdict,
           "seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(verdict), "->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
