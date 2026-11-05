"""OB data -- USD-M futures order-book depth from the public archive (orderbook-depth-ob-1105-spec.md §2-§3).

  parse_day   bookDepth CSV (timestamp, percentage, depth, notional; a snapshot every ~30 s) -> USDT notional of the
              bid (negative percentage) and ask (positive) side at the levels 0.2 / 1 / 2 / 3 / 4 / 5 %
  hourly      per hour close H of the day: the last snapshot in [H - 5 min, H) and the mean 1 % imbalance of the hour;
              hours without such a snapshot are skipped
  fetch_day   one GET per day (data.binance.vision only, via archive_bh.http_get); 404 -> None
  fetch/load  hourly summaries in runtime/bookdepth/<PERP>.npz, resumable by <PERP>.days.json; raw zips are not kept

  (from trader/) ../pyembed/python.exe -u bookdepth.py fetch [--shard i/n]   (n processes, each its own pairs)
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

LEVELS = (0.2, 1, 2, 3, 4, 5)
HOUR = 3600 * 1000
LAST_WINDOW = 5 * 60 * 1000
BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
STORE = TC.RUNTIME / "bookdepth"
PAIRS = TC.RUNTIME / "experiments" / "OB_PAIRS.json"
FIRST_DAY, LAST_DAY = "2023-01-01", "2026-11-03"
WORKERS = 16
CHUNK_DAYS = 100
KEYS = ("t", "bid", "ask", "imb1_mean")


class NotFound(Exception):
    pass


def imb(bid, ask):
    bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (bid - ask) / (bid + ask)


def _ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def parse_day(text: str) -> dict:
    """Vectorised: each distinct timestamp is parsed once (2 880 a day instead of 34 560 strptime calls)."""
    cols = [r.split(",") for r in text.splitlines()[1:] if r.count(",") >= 3]
    if not cols:
        z = np.zeros((0, len(LEVELS)))
        return {"t": np.zeros(0, dtype=np.int64), "bid": z, "ask": z.copy()}
    stamps = [c[0].strip() for c in cols]
    pct = np.array([c[1] for c in cols], dtype=float)
    notional = np.array([c[3] for c in cols], dtype=float)
    uniq = sorted(set(stamps))
    pos = {u: i for i, u in enumerate(uniq)}
    ts = np.array([int(datetime.fromisoformat(u).replace(tzinfo=timezone.utc).timestamp() * 1000) for u in uniq],
                  dtype=np.int64)
    row = np.array([pos[x] for x in stamps], dtype=np.int64)
    lv = np.full(len(pct), -1, dtype=np.int64)
    for k, level in enumerate(LEVELS):
        lv[np.isclose(np.abs(pct), level)] = k
    ok = lv >= 0
    bid = np.full((len(uniq), len(LEVELS)), np.nan)
    ask = np.full((len(uniq), len(LEVELS)), np.nan)
    b, a = ok & (pct < 0), ok & (pct > 0)
    bid[row[b], lv[b]] = notional[b]
    ask[row[a], lv[a]] = notional[a]
    o = np.argsort(ts, kind="stable")
    return {"t": ts[o], "bid": bid[o], "ask": ask[o]}


def hourly(p: dict, day: str) -> dict:
    d0 = _ms(day + " 00:00:00")
    t = p["t"]
    out = {"t": [], "bid": [], "ask": [], "imb1_mean": []}
    i1 = LEVELS.index(1)
    for k in range(1, 25):
        h = d0 + k * HOUR
        lo = int(np.searchsorted(t, h - LAST_WINDOW))
        hi = int(np.searchsorted(t, h))
        if hi <= lo:
            continue
        a = int(np.searchsorted(t, h - HOUR))
        out["t"].append(h)
        out["bid"].append(p["bid"][hi - 1])
        out["ask"].append(p["ask"][hi - 1])
        out["imb1_mean"].append(float(np.nanmean(imb(p["bid"][a:hi, i1], p["ask"][a:hi, i1]))))
    n = len(out["t"])
    return {"t": np.array(out["t"], dtype=np.int64), "bid": np.array(out["bid"]).reshape(n, len(LEVELS)),
            "ask": np.array(out["ask"]).reshape(n, len(LEVELS)), "imb1_mean": np.array(out["imb1_mean"], dtype=float)}


def url_for(perp: str, day: str) -> str:
    return f"{BASE}/{perp}/{perp}-bookDepth-{day}.zip"


def fetch_day(perp: str, day: str, get):
    try:
        raw = get(url_for(perp, day))
    except NotFound:
        return None
    if raw is None:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:                  # read() checks the CRC-32 of the member
        return hourly(parse_day(z.read(z.namelist()[0]).decode("utf-8", "replace")), day)


# ------------------------------------------------------------------ store
def _paths(perp: str, store: Path):
    return store / f"{perp}.npz", store / f"{perp}.days.json"


def done_days(perp: str, store: Path) -> dict:
    _, pj = _paths(perp, store)
    return json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}


def load(perp: str, store: Path = STORE):
    pn, _ = _paths(perp, store)
    if not pn.exists():
        return None
    with np.load(pn) as z:
        return {k: z[k] for k in KEYS}


def save(perp: str, new: dict, store: Path) -> None:
    store.mkdir(parents=True, exist_ok=True)
    pn, pj = _paths(perp, store)
    parts = [v for v in new.values() if v is not None and len(v["t"])]
    old = load(perp, store)
    if old is not None and len(old["t"]):
        parts.append(old)
    if parts:
        a = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
        o = np.argsort(a["t"], kind="stable")
        a = {k: v[o] for k, v in a.items()}
        keep = np.r_[True, a["t"][1:] != a["t"][:-1]]
        np.savez(pn, **{k: v[keep] for k, v in a.items()})
    st = done_days(perp, store)
    st.update({d: ("ok" if v is not None else "missing") for d, v in new.items()})
    pj.write_text(json.dumps(st, sort_keys=True), encoding="utf-8")


def fetch(jobs: dict, get, store: Path = STORE, workers: int = WORKERS, log=print) -> dict:
    """jobs: {perp: [days]}; days already done (ok or missing) are skipped; errors are retried on the next run."""
    tot = {"perps": 0, "files": 0, "missing": 0, "errors": 0}
    t0 = time.time()

    def one(arg):
        perp, d = arg
        try:
            return d, fetch_day(perp, d, get), None
        except Exception as e:                                     # noqa: BLE001
            return d, None, e
    ex = ThreadPoolExecutor(workers)
    for n, (perp, days) in enumerate(sorted(jobs.items()), 1):
        done = done_days(perp, store)
        todo = [d for d in days if d not in done]
        if not todo:
            continue
        ok_n = miss_n = errs = 0
        for c0 in range(0, len(todo), CHUNK_DAYS):                 # saved every CHUNK_DAYS: an interrupt loses one chunk
            got = {}
            for d, v, e in ex.map(one, [(perp, d) for d in todo[c0:c0 + CHUNK_DAYS]]):
                if e is not None:
                    errs += 1
                    continue
                got[d] = v
            save(perp, got, store)
            ok_n += sum(1 for v in got.values() if v is not None)
            miss_n += sum(1 for v in got.values() if v is None)
            log(f"  {perp}: {min(c0 + CHUNK_DAYS, len(todo))}/{len(todo)} days | {(time.time() - t0) / 60:.1f} min")
        tot["perps"] += 1
        tot["files"] += ok_n
        tot["missing"] += miss_n
        tot["errors"] += errs
        log(f"{n}/{len(jobs)} {perp}: {len(todo)} days, ok {ok_n}, missing {miss_n}, errors {errs} | "
            f"{(time.time() - t0) / 60:.1f} min")
    ex.shutdown()
    return tot


def shard(perps, i: int, n: int) -> list:
    return [p for k, p in enumerate(perps) if k % n == i]


def jobs_from_pairs(first: str = FIRST_DAY, last: str = LAST_DAY, i: int = 0, n: int = 1) -> dict:
    perps = shard(json.loads(PAIRS.read_text(encoding="utf-8"))["perp"], i, n)
    d, days = datetime.strptime(first, "%Y-%m-%d"), []
    end = datetime.strptime(last, "%Y-%m-%d")
    while d <= end:
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return {p: days for p in perps}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fetch"]:
        import archive_bh as A
        i, n = (int(x) for x in argv[2].split("/")) if argv[1:2] == ["--shard"] else (0, 1)
        tot = fetch(jobs_from_pairs(i=i, n=n), lambda url: A.http_get(url), workers=8, log=lambda s: print(s, flush=True))
        print(json.dumps(tot))
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
