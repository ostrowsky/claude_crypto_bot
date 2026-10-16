"""Phase B1 -- 15m history of the whole spot universe (wide-universe-b-1014-spec.md §2).

Pairs: TRADING + delisted (BREAK) USDT pairs of the demo exchangeInfo, stablecoins and fiat excluded
(trading_universe.STABLE_BASES). Delisted pairs are kept so the replay is not a survivors-only sample
(TH-05); a pair takes part only on the days it printed candles. Source: public klines of the live
spot market (api.binance.com, the bot's own market), 15m only; 1h is aggregated from 15m.
Storage: trader/runtime/history/<SYM>_15m.npz, append-only merge; gaps are logged, never filled.

  (from trader/) ../pyembed/python.exe -u history_store.py backfill [--since 2023-01-01] [--workers 4]
  (from trader/) ../pyembed/python.exe history_store.py status
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

HIST_DIR = TC.RUNTIME / "history"
BAR_MS = 15 * 60 * 1000
HOUR_MS = 60 * 60 * 1000
LIVE = "https://api.binance.com"
FIELDS = ("t", "o", "h", "l", "c", "v")
SINCE_DEFAULT = "2023-01-01"
RATE_PER_SEC = 5.0


def _stables():
    import trading_universe as TU
    return TU.STABLE_BASES


# ------------------------------------------------------------------ universe
def backtest_universe(xinfo) -> list:
    """TRADING and delisted (BREAK) USDT pairs, no stablecoins/fiat: the replay universe."""
    st = _stables()
    return sorted(s["symbol"] for s in xinfo.get("symbols", [])
                  if s.get("quoteAsset") == "USDT" and s.get("status") in ("TRADING", "BREAK") and s.get("baseAsset") not in st)


def live_universe(xinfo) -> list:
    st = _stables()
    return sorted(s["symbol"] for s in xinfo.get("symbols", [])
                  if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING" and s.get("isSpotTradingAllowed")
                  and s.get("baseAsset") not in st)


# ------------------------------------------------------------------ arrays
def _empty():
    return {"t": np.zeros(0, dtype=np.int64), **{k: np.zeros(0) for k in FIELDS[1:]}}


def merge(old, new):
    """-> (merged, added). Rows already present (same open time) are kept as they were."""
    if old is None or len(old["t"]) == 0:
        old = _empty()
    if new is None or len(new["t"]) == 0:
        return old, 0
    t = np.concatenate([old["t"], new["t"]]).astype(np.int64)
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    keep = np.ones(len(t_sorted), dtype=bool)
    keep[1:] = t_sorted[1:] != t_sorted[:-1]          # first occurrence wins: old rows come first in the concat
    idx = order[keep]
    out = {"t": t[idx]}
    for k in FIELDS[1:]:
        out[k] = np.concatenate([old[k], new[k]]).astype(float)[idx]
    return out, int(len(out["t"]) - len(old["t"]))


def gaps(t, bar_ms: int = BAR_MS) -> list:
    t = np.asarray(t, dtype=np.int64)
    out = []
    for a, b in zip(t[:-1], t[1:]):
        miss = int((b - a) // bar_ms) - 1
        if miss > 0:
            out.append({"from": int(a), "to": int(b), "missing": miss})
    return out


def agg_1h(a, complete_only: bool = False) -> dict:
    """1h OHLCV from 15m: open of the first, max high, min low, close of the last, volume sum; complete = 4 bars."""
    t = np.asarray(a["t"], dtype=np.int64)
    if len(t) == 0:
        return {**_empty(), "complete": np.zeros(0, dtype=bool)}
    hour = t // HOUR_MS * HOUR_MS
    starts = np.flatnonzero(np.r_[True, hour[1:] != hour[:-1]])
    ends = np.r_[starts[1:], len(t)]
    out = {"t": hour[starts], "o": a["o"][starts], "h": np.maximum.reduceat(a["h"], starts),
           "l": np.minimum.reduceat(a["l"], starts), "c": a["c"][ends - 1], "v": np.add.reduceat(a["v"], starts),
           "complete": (ends - starts) == 4}
    if complete_only:
        m = out["complete"]
        out = {k: v[m] for k, v in out.items()}
    return out


def save(path: Path, a: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{k: a[k] for k in FIELDS})
    tmp.replace(path)


def load(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in FIELDS}


# ------------------------------------------------------------------ fetch
RETRIES = 3
SAVE_EVERY_PAGES = 10


def _rows_to_arr(rows):
    if not rows:
        return None
    return {"t": np.array([int(r[0]) for r in rows], dtype=np.int64),
            **{k: np.array([float(r[i]) for r in rows]) for k, i in zip(FIELDS[1:], (1, 2, 3, 4, 5))}}


def _get_page(client, sym, cursor, limit, sleep):
    """One page with up to RETRIES retries on network errors, 5xx and rate limits (pause 2, 4, 8 s; 429/418 -> 60 s)."""
    import exchange_client as X
    for k in range(RETRIES + 1):
        try:
            return client.klines_from(sym, "15m", cursor, limit, base=LIVE)
        except X.ExchangeError as e:
            retry = e.status is None or (isinstance(e.status, int) and (e.status >= 500 or e.status in (418, 429)))
            if not retry or k == RETRIES:
                raise
            sleep(60 if e.status in (418, 429) else 2 ** (k + 1))
        except X.RequestDeadline:
            if k == RETRIES:
                raise
            sleep(2 ** (k + 1))


def fetch_symbol(client, sym, start_ms, end_ms, path, limit: int = 1000, throttle=None, now_ms=None, sleep=time.sleep) -> dict:
    """Page 15m klines from max(start, last stored bar + 1) up to end; keep only closed bars; merge and save.
    Saves every SAVE_EVERY_PAGES pages and before re-raising a persistent error, so a failure keeps what was fetched."""
    have = load(path)
    n0 = 0 if have is None else len(have["t"])
    cursor = int(start_ms) if not n0 else max(int(start_ms), int(have["t"][-1]) + BAR_MS)
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    req, pending, pages = 0, [], 0

    def flush():
        nonlocal have, pending
        if pending:
            merged, added = merge(have, _rows_to_arr(pending))
            if added:
                save(path, merged)
            have, pending = merged, []

    try:
        while True:
            if throttle:
                throttle()
            rows = _get_page(client, sym, cursor, limit, sleep)
            req += 1
            pages += 1
            full = len(rows) >= limit
            rows = [r for r in rows if int(r[0]) < end_ms and int(r[6]) < now_ms]
            pending.extend(rows)
            if pages % SAVE_EVERY_PAGES == 0:
                flush()
            if not full or not rows:
                break
            cursor = int(rows[-1][0]) + BAR_MS
            if cursor >= end_ms:
                break
    finally:
        flush()
    merged = have if have is not None else _empty()
    return {"sym": sym, "requests": req, "added": int(len(merged["t"]) - n0), "bars": int(len(merged["t"])),
            "gaps": len(gaps(merged["t"])), "first": int(merged["t"][0]) if len(merged["t"]) else None,
            "last": int(merged["t"][-1]) if len(merged["t"]) else None}


def path_of(sym: str) -> Path:
    return HIST_DIR / f"{sym}_15m.npz"


def load_1h(sym: str):
    a = load(path_of(sym))
    return None if a is None else agg_1h(a, complete_only=True)


# ------------------------------------------------------------------ CLI
def _order(universe, watchlist):
    """Watchlist first (the bot calibration needs it first), then the rest alphabetically."""
    wl = [s for s in universe if s in watchlist]
    return wl + [s for s in universe if s not in watchlist]


def backfill(since: str = SINCE_DEFAULT, workers: int = 4, symbols=None) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    import audit as A
    import exchange_client as X
    import microstructure_recorder as M
    A.set_process("history_backfill")
    pub = X.ExchangeClient(key="", secret="")
    xinfo = M.cached_exchange_info(pub)
    uni = symbols or backtest_universe(xinfo)
    try:
        wl = set(json.loads((TC.BOT_FILES / "watchlist.json").read_text(encoding="utf-8")))
    except Exception:
        wl = set()
    order = _order(uni, wl)
    start = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end = int(time.time() * 1000)
    lock, gate, done = threading.Lock(), {"next": 0.0}, {"n": 0, "req": 0, "added": 0}
    log = TC.RUNTIME / "history" / "backfill_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    def throttle():
        with lock:
            now = time.time()
            at = max(now, gate["next"])
            gate["next"] = at + 1.0 / RATE_PER_SEC
        if at > now:
            time.sleep(at - now)

    def one(sym):
        try:
            st = fetch_symbol(pub, sym, start, end, path_of(sym), throttle=throttle)
        except Exception as e:
            st = {"sym": sym, "error": repr(e)[:300]}
        st["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with lock:
            done["n"] += 1
            done["req"] += st.get("requests", 0)
            done["added"] += st.get("added", 0)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(st) + "\n")
            if done["n"] % 10 == 0 or done["n"] == len(order):
                print("pairs %d/%d | requests %d | bars added %d" % (done["n"], len(order), done["req"], done["added"]), flush=True)
        return st

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        res = list(pool.map(one, order))
    summary = {"pairs": len(order), "watchlist_first": len([s for s in order if s in wl]), "requests": done["req"],
               "bars_added": done["added"], "errors": sum(1 for r in res if "error" in r),
               "empty": sum(1 for r in res if r.get("bars") == 0), "since": since}
    (TC.RUNTIME / "history" / "BACKFILL_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def status() -> dict:
    files = sorted(HIST_DIR.glob("*_15m.npz"))
    bars = 0
    for f in files:
        try:
            bars += int(np.load(f, allow_pickle=False)["t"].shape[0])
        except Exception:
            pass
    return {"pairs_with_history": len(files), "bars": bars, "dir": str(HIST_DIR)}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "backfill":
        since = argv[argv.index("--since") + 1] if "--since" in argv else SINCE_DEFAULT
        workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 4
        print(json.dumps(backfill(since, workers), indent=1))
        return 0
    print(json.dumps(status(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
