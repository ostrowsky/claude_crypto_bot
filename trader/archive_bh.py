"""BH19 data -- Binance public monthly archives 2017-11..2022-12 into a SEPARATE store (backward-holdout-bh19-1030-spec.md §3).

  parse_klines / parse_funding   monthly CSV (header or not; ms or microseconds) -> arrays
  parse_listing / months_in_range  S3 listing of data.binance.vision (XML, paged by marker) -> keys, months in range
  http_get                         GET only; data.binance.vision (files) and its S3 listing host
  fetch                            spot 15m of every *USDT pair in the archive (delisted included), USD-M 15m and
                                   funding of BTCUSDT / ETHUSDT; per-symbol npz in runtime/history_bh/, resumable
  daily_panel / build              xsmom.Panel of the store cut at 2022-12-31 + the FC market (perp panel + funding)

  (from trader/) ../pyembed/python.exe -u archive_bh.py fetch | build | status
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

FILES = "data.binance.vision"
LISTING = "s3-ap-northeast-1.amazonaws.com"
LIST_URL = f"https://{LISTING}/{FILES}"
STORE = TC.RUNTIME / "history_bh"
PERP = STORE / "perp"
FUND = STORE / "funding"
FIRST, LAST = "2017-11", "2022-12"
PERP_FIRST = "2020-01"
LAST_DAY = "2022-12-31"
FC_PAIRS = ("BTCUSDT", "ETHUSDT")
WORKERS = 16
PANEL = TC.RUNTIME / "experiments" / "bh_panel.npz"
MARKET = TC.RUNTIME / "experiments" / "bh_market.npz"
# bases of 2017-2022 that are pegged (stablecoins / fiat) but delisted before trading_universe.STABLE_BASES was written
OLD_PEGGED = frozenset({"PAX", "USDS", "USDSB", "SUSD", "AUD", "BKRW", "IDRT", "BIDR", "NGN", "RUB", "BVND", "VAI", "UST",
                        "USTC", "EUR", "GBP", "USDP", "TUSD", "BUSD", "USDC", "DAI"})


def keep_symbol(sym: str) -> bool:
    """Same replay universe as history_store: USDT pairs without stablecoins / fiat."""
    import trading_universe as TU
    if not sym.endswith("USDT"):
        return False
    base = sym[:-4]
    return base not in TU.STABLE_BASES and base not in OLD_PEGGED


# ------------------------------------------------------------------ parsing
def _rows(text):
    for r in csv.reader(io.StringIO(text)):
        if not r or not r[0].strip().lstrip("-").isdigit():
            continue                                               # header or blank
        yield r


def parse_klines(text: str) -> dict:
    by = {}
    for r in _rows(text):
        t = int(r[0])
        if t > 10 ** 14:                                           # microseconds (archives since 2025)
            t //= 1000
        by[t] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
    ts = np.array(sorted(by), dtype=np.int64)
    v = np.array([by[int(t)] for t in ts], dtype=float).reshape(-1, 5)
    return {"t": ts, "o": v[:, 0], "h": v[:, 1], "l": v[:, 2], "c": v[:, 3], "v": v[:, 4]}


def parse_funding(text: str) -> dict:
    by = {}
    for r in _rows(text):
        t = int(r[0])
        if t > 10 ** 14:
            t //= 1000
        by[t] = float(r[-1])
    ts = np.array(sorted(by), dtype=np.int64)
    return {"t": ts, "rate": np.array([by[int(t)] for t in ts], dtype=float), "mark": np.full(len(ts), np.nan)}


def parse_listing(xml: str):
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", xml)
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    truncated = "<IsTruncated>true</IsTruncated>" in xml
    nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
    items = [p for p in prefixes if xml.find(f"<CommonPrefixes><Prefix>{p}") >= 0] or keys
    marker = (nxt.group(1) if nxt else (items[-1] if items else None)) if truncated else None
    return items, marker


def months_in_range(keys, first: str = FIRST, last: str = LAST) -> list:
    out = set()
    for k in keys:
        m = re.search(r"-(\d{4}-\d{2})\.zip$", k)
        if m and first <= m.group(1) <= last:
            out.add(m.group(1))
    return sorted(out)


# ------------------------------------------------------------------ network (GET only, two hosts)
_TLS = threading.local()


def http_get(url: str, method: str = "GET", retries: int = 4):
    import http.client
    import urllib.parse
    u = urllib.parse.urlsplit(url)
    if method != "GET" or u.scheme != "https" or u.hostname not in (FILES, LISTING):
        raise ValueError("archive_bh: only GET to data.binance.vision and its S3 listing")
    if u.hostname == LISTING and not u.path.startswith(f"/{FILES}"):
        raise ValueError("archive_bh: listing host only for the data.binance.vision bucket")
    path = u.path + (("?" + u.query) if u.query else "")
    for k in range(retries):
        try:
            conns = getattr(_TLS, "c", None)
            if conns is None:
                conns = _TLS.c = {}
            c = conns.get(u.hostname)
            if c is None:
                c = conns[u.hostname] = http.client.HTTPSConnection(u.hostname, timeout=30)
            c.request("GET", path, headers={"User-Agent": "trader-bh19/1", "Connection": "keep-alive"})
            r = c.getresponse()
            body = r.read()
            if r.status == 404:
                return None
            if r.status == 200:
                return body
            raise IOError(f"HTTP {r.status}")
        except Exception:
            try:
                _TLS.c.pop(u.hostname).close()
            except Exception:
                pass
            if k == retries - 1:
                raise
        time.sleep(1.5 * (k + 1))


def list_all(prefix: str, delimiter: bool = True) -> list:
    out, marker = [], None
    while True:
        q = f"?prefix={prefix}" + ("&delimiter=/" if delimiter else "") + (f"&marker={marker}" if marker else "")
        items, marker = parse_listing(http_get(LIST_URL + q).decode("utf-8", "replace"))
        out += items
        if not marker:
            return out


def _zip_text(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:                     # read() checks the CRC-32
        return z.read(z.namelist()[0]).decode("utf-8", "replace")


# ------------------------------------------------------------------ store
def _save(path: Path, parts: list, keys) -> None:
    parts = [p for p in parts if p is not None and len(p["t"])]
    if not parts:
        return
    t = np.concatenate([p["t"] for p in parts])
    o = np.argsort(t, kind="stable")
    t = t[o]
    keep = np.r_[t[1:] != t[:-1], True]
    out = {"t": t[keep]}
    for k in keys:
        out[k] = np.concatenate([p[k] for p in parts])[o][keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def _done(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fetch_series(kind_prefix: str, sym: str, months: list, out_npz: Path, parser, keys, ex) -> dict:
    """Months of one symbol (already listed), resumable by <npz>.months.json."""
    log = out_npz.with_suffix(".months.json")
    done = _done(log)
    todo = [m for m in months if m not in done]
    if not todo:
        return {"new": 0}

    def one(m):
        name = kind_prefix.rstrip("/").split("/")[-1]
        url = f"https://{FILES}/{kind_prefix}{sym}-{name}-{m}.zip" if name in ("15m",) else \
            f"https://{FILES}/{kind_prefix}{sym}-fundingRate-{m}.zip"
        try:
            raw = http_get(url)
            return m, (parser(_zip_text(raw)) if raw else None), None
        except Exception as e:                                     # noqa: BLE001
            return m, None, e
    old = None
    if out_npz.exists():
        with np.load(out_npz) as z:
            old = {k: z[k] for k in ("t",) + tuple(keys)}
    got, errs = [], 0
    for m, a, e in ex.map(one, todo):
        if e is not None:
            errs += 1
            continue
        done[m] = "ok" if a is not None else "missing"
        got.append(a)
    _save(out_npz, [old] + got, keys)
    log.write_text(json.dumps(done, sort_keys=True), encoding="utf-8")
    return {"new": len(got), "errors": errs}


def fetch(log=print) -> dict:
    t0 = time.time()
    ex = ThreadPoolExecutor(WORKERS)
    syms = sorted(p.rstrip("/").split("/")[-1] for p in list_all("data/spot/monthly/klines/"))
    usdt = [s for s in syms if s.endswith("USDT")]
    log(f"archive symbols {len(syms)}, *USDT {len(usdt)}")

    def months_of(s):
        try:
            return s, months_in_range(list_all(f"data/spot/monthly/klines/{s}/15m/", delimiter=False))
        except Exception:                                          # noqa: BLE001
            return s, None
    plan = {}
    for n, (s, ms) in enumerate(ex.map(months_of, usdt), 1):
        if ms:
            plan[s] = ms
        if n % 200 == 0:
            log(f"  listed {n}/{len(usdt)}, with data in range {len(plan)}, {(time.time() - t0) / 60:.1f} min")
    log(f"pairs with spot 15m in {FIRST}..{LAST}: {len(plan)}, months {sum(len(v) for v in plan.values())}")
    tot = {"pairs": 0, "months": 0, "errors": 0}
    kl = ("o", "h", "l", "c", "v")
    for n, (s, ms) in enumerate(sorted(plan.items()), 1):
        r = fetch_series("data/spot/monthly/klines/" + s + "/15m/", s, ms, STORE / f"{s}_15m.npz", parse_klines, kl, ex)
        tot["pairs"] += 1
        tot["months"] += r.get("new", 0)
        tot["errors"] += r.get("errors", 0)
        if n % 50 == 0:
            log(f"  spot {n}/{len(plan)} {tot} {(time.time() - t0) / 60:.1f} min")
    for s in FC_PAIRS:
        pm = months_in_range(list_all(f"data/futures/um/monthly/klines/{s}/15m/", delimiter=False), PERP_FIRST, LAST)
        fetch_series("data/futures/um/monthly/klines/" + s + "/15m/", s, pm, PERP / f"{s}_15m.npz", parse_klines, kl, ex)
        fm = months_in_range(list_all(f"data/futures/um/monthly/fundingRate/{s}/", delimiter=False), PERP_FIRST, LAST)
        fetch_series("data/futures/um/monthly/fundingRate/" + s + "/", s, fm, FUND / f"{s}_funding.npz", parse_funding,
                     ("rate", "mark"), ex)
    ex.shutdown()
    log(json.dumps({**tot, "minutes": round((time.time() - t0) / 60, 1)}))
    return tot


# ------------------------------------------------------------------ panels
def daily_panel(daily_by_sym: dict, last_day: str = LAST_DAY):
    import xsmom as XS
    lim = XS.day_num(last_day)
    cut = {}
    for s, d in daily_by_sym.items():
        if not keep_symbol(s):
            continue
        keep = np.asarray(d["day"]) <= lim
        cut[s] = {k: np.asarray(v)[keep] for k, v in d.items()}
    p = XS.Panel.from_daily({s: d for s, d in cut.items() if len(d["day"])})
    return p.cut(XS.day_str(lim + 1))


def _load(path: Path, keys):
    with np.load(path) as z:
        return {k: z[k] for k in keys}


def build() -> dict:
    import carry as C
    import xsmom as XS
    kl = ("t", "o", "h", "l", "c", "v")
    daily = {}
    for f in sorted(STORE.glob("*_15m.npz")):
        daily[f.name[:-len("_15m.npz")]] = XS.daily_from_15m(_load(f, kl))
    p = daily_panel(daily)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PANEL, days=p.days, syms=np.array(p.syms), close=p.close, first_close=p.first_close, qv=p.qv,
                        complete=p.complete, has_first=p.has_first, has_bar=p.has_bar)
    pd, fund = {}, {}
    lim = XS.day_num(LAST_DAY)
    for s in FC_PAIRS:
        a = _load(PERP / f"{s}_15m.npz", kl)
        d = C.daily_perp(a)
        keep = np.asarray(d["day"]) <= lim
        pd[s] = {k: np.asarray(v)[keep] for k, v in d.items()}
        f = _load(FUND / f"{s}_funding.npz", ("t", "rate"))
        fund[s] = f
    perp = C.PerpPanel.from_daily(pd, mult={s: 1 for s in FC_PAIRS})
    np.savez_compressed(MARKET, days=perp.days, syms=np.array(perp.syms), close=perp.close, first_close=perp.first_close,
                        qv=perp.qv, complete=perp.complete, has_first=perp.has_first, has_bar=perp.has_bar, high=perp.high,
                        **{f"f_t__{s}": fund[s]["t"] for s in FC_PAIRS}, **{f"f_r__{s}": fund[s]["rate"] for s in FC_PAIRS})
    r = {"panel_days": int(p.n), "panel_from": XS.day_str(p.d0), "panel_to": XS.day_str(p.d0 + p.n - 1),
         "pairs": len(p.syms), "perp_days": int(perp.n)}
    print(json.dumps(r))
    return r


def load_panel():
    import xsmom as XS
    z = np.load(PANEL, allow_pickle=False)
    return XS.Panel(z["days"], [str(s) for s in z["syms"]], z["close"], z["first_close"], z["qv"], z["complete"],
                    z["has_first"], z["has_bar"])


def load_market():
    import carry as C
    z = np.load(MARKET, allow_pickle=False)
    syms = [str(s) for s in z["syms"]]
    perp = C.PerpPanel(z["days"], syms, z["close"], z["first_close"], z["qv"], z["complete"], z["has_first"],
                       z["has_bar"], z["high"])
    return C.Market(spot=load_panel(), perp=perp, funding={s: {"t": z[f"f_t__{s}"], "rate": z[f"f_r__{s}"]} for s in syms})


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "fetch":
        fetch(log=lambda s: print(s, flush=True))
        return 0
    if cmd == "build":
        build()
        return 0
    if cmd == "status":
        print(json.dumps({"spot_files": len(list(STORE.glob("*_15m.npz"))) if STORE.exists() else 0}))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
