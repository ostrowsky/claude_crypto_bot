"""FC-0 -- live USD-M funding history and perpetual 15m klines for the funding-carry money test
(carry-funding-money-1021-spec.md §3). Public, unsigned data only; append-only storage.

  funding   trader/runtime/funding/<SPOT>_funding.npz      t (ms), rate, mark (NaN when the API gives "")
  perp      trader/runtime/perp_history/<SPOT>_15m.npz      B1 layout (t, o, h, l, c, v), prices per PERP contract
  mapping   trader/runtime/funding/mapping.json             {spot: {"perp", "mult", "status"}}

Perp <-> spot: the same name if that spot pair exists (BTCUSDT, 1INCHUSDT, 1000SATSUSDT); otherwise a numeric prefix
1000 / 10000 / 1000000 / 1M is stripped (1000PEPEUSDT -> PEPEUSDT x1000). Prices are divided by the multiplier later
(carry.PerpPanel). Pairs whose perp/spot price ratio is off by more than 5% on a common day are dropped and logged.

  (from trader/) ../pyembed/python.exe -u funding_store.py backfill [--universe-only] [--since 2023-01-01]
  (from trader/) ../pyembed/python.exe funding_store.py status
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

FUNDING_DIR = TC.RUNTIME / "funding"
PERP_DIR = TC.RUNTIME / "perp_history"
MAPPING = FUNDING_DIR / "mapping.json"
SINCE_DEFAULT = "2023-01-01"
PREFIXES = (("1000000", 1_000_000), ("10000", 10_000), ("1000", 1000), ("1M", 1_000_000))
FUNDING_PER_SEC = 1.5          # /fapi/v1/fundingRate shares 500 requests / 5 min / IP
KLINES_PER_SEC = 6.0           # weight 5 per 1000-bar page against 2400 / min
BAR_MS = 15 * 60 * 1000


# ------------------------------------------------------------------ mapping
def perp_to_spot(perp: str, spot_syms):
    spot_syms = set(spot_syms)
    if perp in spot_syms:
        return perp, 1
    for pre, mult in PREFIXES:
        if perp.startswith(pre) and perp[len(pre):] in spot_syms:
            return perp[len(pre):], mult
    return None


# ------------------------------------------------------------------ funding arrays
def _empty():
    return {"t": np.zeros(0, dtype=np.int64), "rate": np.zeros(0), "mark": np.zeros(0)}


def parse_funding(rows) -> dict:
    if not rows:
        return _empty()
    t = np.array([int(r["fundingTime"]) for r in rows], dtype=np.int64)
    rate = np.array([float(r["fundingRate"]) for r in rows], dtype=float)
    mark = np.array([float(r["markPrice"]) if str(r.get("markPrice") or "").strip() else np.nan for r in rows], dtype=float)
    o = np.argsort(t, kind="stable")
    return {"t": t[o], "rate": rate[o], "mark": mark[o]}


def merge(old, new):
    """-> (merged, added). A time already stored keeps its stored row (append-only)."""
    old = old if old is not None and len(old["t"]) else _empty()
    if new is None or not len(new["t"]):
        return old, 0
    t = np.concatenate([old["t"], new["t"]]).astype(np.int64)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    keep = np.ones(len(ts), dtype=bool)
    keep[1:] = ts[1:] != ts[:-1]
    idx = order[keep]
    out = {"t": t[idx], "rate": np.concatenate([old["rate"], new["rate"]])[idx],
           "mark": np.concatenate([old["mark"], new["mark"]])[idx]}
    return out, int(len(out["t"]) - len(old["t"]))


def save_funding(path: Path, a: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, t=a["t"], rate=a["rate"], mark=a["mark"])
    tmp.replace(path)


def load_funding(path: Path):
    if not Path(path).exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {"t": z["t"], "rate": z["rate"], "mark": z["mark"]}


def funding_path(spot: str) -> Path:
    return FUNDING_DIR / f"{spot}_funding.npz"


def perp_path(spot: str) -> Path:
    return PERP_DIR / f"{spot}_15m.npz"


# ------------------------------------------------------------------ fetch
def _throttle_factory(per_sec: float):
    lock, gate = threading.Lock(), {"next": 0.0}

    def throttle():
        with lock:
            now = time.time()
            at = max(now, gate["next"])
            gate["next"] = at + 1.0 / per_sec
        if at > now:
            time.sleep(at - now)
    return throttle


def _with_retry(call, retries: int = 4):
    import exchange_client as X
    for k in range(retries + 1):
        try:
            return call()
        except X.ExchangeError as e:
            retry = e.status is None or (isinstance(e.status, int) and (e.status >= 500 or e.status in (418, 429)))
            if not retry or k == retries:
                raise
            time.sleep(60 if e.status in (418, 429) else 2 ** (k + 1))
        except X.RequestDeadline:
            if k == retries:
                raise
            time.sleep(2 ** (k + 1))


def fetch_funding(client, perp: str, spot: str, start_ms: int, throttle) -> dict:
    have = load_funding(funding_path(spot))
    cursor = int(start_ms) if have is None or not len(have["t"]) else max(int(start_ms), int(have["t"][-1]) + 1)
    pages = added = 0
    while True:
        throttle()
        rows = _with_retry(lambda: client.funding_rates(perp, cursor, 1000))
        pages += 1
        if not rows:
            break
        have, a = merge(have, parse_funding(rows))
        added += a
        if len(rows) < 1000:
            break
        cursor = int(rows[-1]["fundingTime"]) + 1
    if have is not None and len(have["t"]):
        save_funding(funding_path(spot), have)
    return {"pages": pages, "added": added, "events": 0 if have is None else int(len(have["t"]))}


def fetch_perp(client, perp: str, spot: str, start_ms: int, end_ms: int, throttle) -> dict:
    import history_store as HS
    path = perp_path(spot)
    have = HS.load(path)
    cursor = int(start_ms) if have is None or not len(have["t"]) else max(int(start_ms), int(have["t"][-1]) + BAR_MS)
    pages, pending = 0, []
    now_ms = int(time.time() * 1000)
    while cursor < end_ms:
        throttle()
        rows = _with_retry(lambda: client.futures_klines_from(perp, "15m", cursor, 1000))
        pages += 1
        if not rows:
            break
        pending.extend(r for r in rows if int(r[0]) < end_ms and int(r[6]) < now_ms)
        if len(rows) < 1000:
            break
        cursor = int(rows[-1][0]) + BAR_MS
        if len(pending) >= 20_000:
            have, _ = HS.merge(have, HS._rows_to_arr(pending))
            HS.save(path, have)
            pending = []
    if pending:
        have, _ = HS.merge(have, HS._rows_to_arr(pending))
        HS.save(path, have)
    return {"pages": pages, "bars": 0 if have is None else int(len(have["t"]))}


def universe_ever(since: str = "2023-03-01") -> set:
    """Spot pairs that enter the XSMOM liquidity universe (top-100) on any decision day from `since`."""
    import experiment_xsmom as XE
    import xsmom as XS
    panel = XE.load_panel()
    cfg = XS.Config(L=1, H=1, K=10)
    out = set()
    for r in range(max(0, panel.row(since)), panel.n + 1):
        out.update(XS.universe(panel, XS.day_str(panel.d0 + r), cfg))
    return out


def backfill(since: str = SINCE_DEFAULT, universe_only: bool = False, workers: int = 3) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    import audit as A
    import exchange_client as X
    import history_store as HS
    A.set_process("funding_backfill")
    pub = X.ExchangeClient(key="", secret="")
    xinfo = _with_retry(lambda: pub.futures_exchange_info())
    spot_syms = {p.name[:-len("_15m.npz")] for p in HS.HIST_DIR.glob("*_15m.npz")}
    keep = universe_ever() if universe_only else None
    mapping = {}
    for s in xinfo.get("symbols", []):
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
            continue
        m = perp_to_spot(s["symbol"], spot_syms)
        if m is None or (keep is not None and m[0] not in keep):
            continue
        spot, mult = m
        prev = mapping.get(spot)
        if prev is None or (prev["status"] != "TRADING" and s.get("status") == "TRADING"):
            mapping[spot] = {"perp": s["symbol"], "mult": mult, "status": s.get("status")}
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING.write_text(json.dumps(mapping, indent=1, sort_keys=True), encoding="utf-8")
    start = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end = int(time.time() * 1000)
    th_f, th_k = _throttle_factory(FUNDING_PER_SEC), _throttle_factory(KLINES_PER_SEC)
    log = FUNDING_DIR / "backfill_log.jsonl"
    lock, done = threading.Lock(), {"n": 0}

    def one(item):
        spot, m = item
        rec = {"spot": spot, "perp": m["perp"], "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            rec["funding"] = fetch_funding(pub, m["perp"], spot, start, th_f)
            rec["perp"] = fetch_perp(pub, m["perp"], spot, start, end, th_k)
        except Exception as e:  # one pair must not stop the run
            rec["error"] = repr(e)[:300]
        with lock:
            done["n"] += 1
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if done["n"] % 10 == 0 or done["n"] == len(mapping):
                print("pairs %d/%d" % (done["n"], len(mapping)), flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        res = list(pool.map(one, sorted(mapping.items())))
    summary = {"pairs": len(mapping), "errors": sum(1 for r in res if "error" in r), "since": since,
               "universe_only": universe_only, "utc": datetime.now(timezone.utc).isoformat()}
    (FUNDING_DIR / "BACKFILL_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def status() -> dict:
    return {"funding_files": len(list(FUNDING_DIR.glob("*_funding.npz"))), "perp_files": len(list(PERP_DIR.glob("*_15m.npz"))),
            "mapping": len(json.loads(MAPPING.read_text(encoding="utf-8"))) if MAPPING.exists() else 0}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "backfill":
        since = argv[argv.index("--since") + 1] if "--since" in argv else SINCE_DEFAULT
        print(json.dumps(backfill(since, universe_only="--universe-only" in argv), indent=1))
        return 0
    print(json.dumps(status(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
