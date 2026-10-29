"""Binance USD-M futures metrics archive (data.binance.vision), 5-minute rows (dir-positioning-dpos-1029-spec.md §2-§3).

  parse_csv / merge       CSV of one day -> arrays by create_time (sorted, deduped); missing days recorded, not filled
  url_for / allowed_day   public archive URL; days <= 2025-12-31 or the PIT window 2026-08-21..2026-10-28 only
  plan / fetch_day        days still to fetch; one day = GET zip (+ GET .CHECKSUM, sha256 checked), 404 -> None
  load / fetch            per-perp npz in runtime/metrics_um/ + <perp>.days.json {day: ok | missing}; resumable

  (from trader/) ../pyembed/python.exe -u metrics_store.py fetch     -- jobs from the DIR universe + the PIT window
  (from trader/) ../pyembed/python.exe metrics_store.py status
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

HOST = "https://data.binance.vision"
BASE = HOST + "/data/futures/um/daily/metrics"
STORE = TC.RUNTIME / "metrics_um"
TRAIN_LAST = "2025-12-31"
PIT_FROM, PIT_TO = "2026-08-21", "2026-10-28"
WARMUP_DAYS = 7
WORKERS = 16
COLS = {"sum_open_interest": "oi", "sum_open_interest_value": "oi_val", "count_toptrader_long_short_ratio": "top_acc",
        "sum_toptrader_long_short_ratio": "top_pos", "count_long_short_ratio": "ls_all",
        "sum_taker_long_short_vol_ratio": "taker"}
KEYS = ("t",) + tuple(COLS.values())


def _empty():
    return {"t": np.zeros(0, dtype=np.int64), **{k: np.zeros(0) for k in COLS.values()}}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def parse_csv(text: str) -> dict:
    rd = csv.DictReader(io.StringIO(text))
    by = {}
    for r in rd:
        try:
            t = int(datetime.strptime(r["create_time"].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    .timestamp() * 1000)
        except (KeyError, ValueError, AttributeError):
            continue
        by[t] = [_f(r.get(c)) for c in COLS]                     # a later duplicate replaces an earlier one
    if not by:
        return _empty()
    ts = np.array(sorted(by), dtype=np.int64)
    vals = np.array([by[int(t)] for t in ts], dtype=float)
    return {"t": ts, **{k: vals[:, i] for i, k in enumerate(COLS.values())}}


def merge(parsed_by_day: dict):
    parts = [parsed_by_day[d] for d in sorted(parsed_by_day) if parsed_by_day[d] is not None]
    missing = sorted(d for d, v in parsed_by_day.items() if v is None)
    if not parts:
        return _empty(), missing
    t = np.concatenate([p["t"] for p in parts])
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.r_[t[1:] != t[:-1], True]                          # dedupe across days, keep the last
    out = {"t": t[keep]}
    for k in COLS.values():
        out[k] = np.concatenate([p[k] for p in parts])[order][keep]
    return out, missing


def url_for(perp: str, day: str) -> str:
    return f"{BASE}/{perp}/{perp}-metrics-{day}.zip"


def allowed_day(day: str) -> bool:
    return day <= TRAIN_LAST or PIT_FROM <= day <= PIT_TO


def plan(days, have_days) -> list:
    have = set(have_days)
    return sorted({d for d in days if allowed_day(d) and d not in have})


def fetch_day(perp: str, day: str, get, verify_sha256: bool = False):
    """One GET per day; integrity by the zip CRC-32 (the .CHECKSUM round trip doubled the request count, 10-29)."""
    if not allowed_day(day):
        raise ValueError(f"metrics_store: day {day} is outside the allowed windows")
    url = url_for(perp, day)
    raw = get(url, method="GET")
    if raw is None:
        return None
    if verify_sha256:
        chk = get(url + ".CHECKSUM", method="GET")
        if chk and hashlib.sha256(raw).hexdigest() != chk.decode("ascii", "replace").split()[0].strip().lower():
            raise IOError(f"metrics_store: checksum mismatch {url}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:                     # read() checks the CRC-32 of the member
        name = z.namelist()[0]
        return parse_csv(z.read(name).decode("utf-8", "replace"))


_TLS = threading.local()


def http_get(url: str, method: str = "GET", retries: int = 4):
    """GET over a keep-alive HTTPS connection per thread (a new TLS handshake per file capped the rate at ~3/s)."""
    import http.client
    if method != "GET" or not url.startswith(HOST + "/"):
        raise ValueError("metrics_store: only GET to data.binance.vision")
    path = url[len(HOST):]
    for k in range(retries):
        try:
            c = getattr(_TLS, "conn", None)
            if c is None:
                c = _TLS.conn = http.client.HTTPSConnection("data.binance.vision", timeout=30)
            c.request("GET", path, headers={"User-Agent": "trader-dpos/1", "Connection": "keep-alive"})
            r = c.getresponse()
            body = r.read()
            if r.status == 404:
                return None
            if r.status == 200:
                return body
            raise IOError(f"HTTP {r.status}")
        except Exception:
            try:
                _TLS.conn.close()
            except Exception:
                pass
            _TLS.conn = None
            if k == retries - 1:
                raise
        time.sleep(1.5 * (k + 1))


# ------------------------------------------------------------------ store
def _paths(perp):
    return STORE / f"{perp}.npz", STORE / f"{perp}.days.json"


def done_days(perp) -> dict:
    _, pj = _paths(perp)
    return json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}


def load(perp: str):
    pn, _ = _paths(perp)
    if not pn.exists():
        return None
    with np.load(pn) as z:
        return {k: z[k] for k in KEYS}


def save(perp: str, new_by_day: dict) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    pn, pj = _paths(perp)
    old = load(perp)
    parts = {"_old": old} if old is not None and len(old["t"]) else {}
    parts.update({d: v for d, v in new_by_day.items()})
    a, _ = merge(parts)
    np.savez(pn, **a)
    st = done_days(perp)
    st.update({d: ("ok" if v is not None else "missing") for d, v in new_by_day.items()})
    pj.write_text(json.dumps(st, sort_keys=True), encoding="utf-8")


def fetch(jobs: dict, get=http_get, workers: int = WORKERS, log=print) -> dict:
    """jobs: {perp: [days]}; perps one by one, days of a perp in parallel; errors are retried on the next run."""
    tot = {"perps": 0, "files": 0, "missing": 0, "errors": 0}
    t0 = time.time()

    def one(arg):
        perp, d = arg
        try:
            return d, fetch_day(perp, d, get), None
        except Exception as e:                                     # noqa: BLE001
            return d, None, e
    ex = ThreadPoolExecutor(workers)                                # one pool: threads keep their connections
    for n, (perp, days) in enumerate(sorted(jobs.items()), 1):
        todo = plan(days, done_days(perp))
        if not todo:
            continue
        got, errs = {}, 0
        for d, v, e in ex.map(one, [(perp, d) for d in todo]):
            if e is not None:
                errs += 1
                continue
            got[d] = v
        save(perp, got)
        tot["perps"] += 1
        tot["files"] += sum(1 for v in got.values() if v is not None)
        tot["missing"] += sum(1 for v in got.values() if v is None)
        tot["errors"] += errs
        log(f"{n}/{len(jobs)} {perp}: {len(todo)} days, ok {sum(1 for v in got.values() if v is not None)}, "
            f"missing {sum(1 for v in got.values() if v is None)}, errors {errs} | total {tot} | "
            f"{(time.time() - t0) / 60:.1f} min")
    ex.shutdown()
    return tot


# ------------------------------------------------------------------ jobs
def mapping() -> dict:
    import funding_store as FS
    return json.loads(FS.MAPPING.read_text(encoding="utf-8")) if FS.MAPPING.exists() else {}


def _day(n: int) -> str:
    return date.fromordinal(n).isoformat()


def jobs_dir_universe() -> tuple:
    """{perp: days}: every day a spot pair is in the DIR universe (2023-03-01..2025-12-31) + 7 warm-up days."""
    import experiment_dir as EDIR
    import experiment_xsmom as EX
    import xsmom as XS
    P = EX.load_panel().cut(EDIR.TEST_FROM)
    cfg = XS.Config(L=1, H=1, K=10, min_age_days=60)
    mp = mapping()
    jobs, no_perp = {}, set()
    for d in range(P.row(EDIR.TRAIN_FROM), P.n):
        day = XS.day_str(P.d0 + d)
        o = date.fromisoformat(day).toordinal()
        for s in XS.universe(P, day, cfg):
            m = mp.get(s)
            if not m:
                no_perp.add(s)
                continue
            jobs.setdefault(m["perp"], set()).update(_day(o - k) for k in range(WARMUP_DAYS + 1))
    return {p: sorted(v) for p, v in jobs.items()}, sorted(no_perp)


def jobs_pit() -> dict:
    """{perp: days} of the bot's positioning snapshots inside the PIT window (data check only)."""
    mp = mapping()
    out = {}
    with open(TC.POSITIONING, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            day = str(r.get("ts", ""))[:10]
            if not (PIT_FROM <= day <= PIT_TO):
                continue
            s = r.get("sym")
            perp = (mp.get(s) or {}).get("perp", s)
            out.setdefault(perp, set()).update({day, (date.fromisoformat(day) - timedelta(days=1)).isoformat()})
    return {p: sorted(d for d in v if allowed_day(d)) for p, v in out.items()}


def status() -> dict:
    files = list(STORE.glob("*.days.json")) if STORE.exists() else []
    ok = miss = 0
    for f in files:
        st = json.loads(f.read_text(encoding="utf-8"))
        ok += sum(1 for v in st.values() if v == "ok")
        miss += sum(1 for v in st.values() if v == "missing")
    return {"perps": len(files), "days_ok": ok, "days_missing": miss}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "fetch":
        jd, no_perp = jobs_dir_universe()
        jp = jobs_pit()
        jobs = {p: sorted(set(jd.get(p, [])) | set(jp.get(p, []))) for p in set(jd) | set(jp)}
        print(f"DIR universe perps {len(jd)} ({sum(len(v) for v in jd.values())} days), spot pairs without a perp "
              f"{len(no_perp)}; PIT perps {len(jp)} ({sum(len(v) for v in jp.values())} days); "
              f"to fetch {sum(len(plan(v, done_days(p))) for p, v in jobs.items())} files", flush=True)
        print(json.dumps(fetch(jobs, log=lambda s: print(s, flush=True))))
        print(json.dumps(status()))
        return 0
    if cmd == "status":
        print(json.dumps(status()))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
