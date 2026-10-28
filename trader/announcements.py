"""LA -- exchange announcements as information before the price (listing-announcements-la-1028-spec.md).

LA-1 collection: Binance CMS (catalog 48 new listings, 161 delisting), OKX new listings, Bybit new_crypto; public
read-only GET, own store trader/runtime/announcements/<source>.jsonl, append-only, idempotent by (source, id).
LA-2 labelling: deterministic type + tickers from the title, mapped to Binance spot pairs.
Timing helpers for LA-3: tradable, entry bar, pre-publication return (leak check), random controls.

  (from trader/) ../pyembed/python.exe -u announcements.py backfill [--since 2023-01-01]
  (from trader/) ../pyembed/python.exe announcements.py step          first page of every source (task, every 5 min)
  (from trader/) ../pyembed/python.exe announcements.py status
"""
from __future__ import annotations

import io
import json
import math
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

ROOT = TC.RUNTIME / "announcements"
HOSTS = {"binance": "https://www.binance.com", "okx": "https://www.okx.com", "bybit": "https://api.bybit.com"}
BINANCE_CATALOGS = (48, 161)
BAR_MS = 15 * 60 * 1000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
MIGRATION_CLUSTER = 20
CONTROL_GAP_MS = 48 * HOUR_MS
PAUSE_SEC = 1.1
UA = "Mozilla/5.0 (crypto-bot trader; read-only announcements)"


def _iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


# ------------------------------------------------------------------ parsing (spec §2)
def parse_binance(a: dict, catalog: int) -> dict:
    code = str(a.get("code") or a.get("id"))
    return {"source": "binance", "id": code, "catalog": int(catalog), "title": a.get("title", ""),
            "url": f"https://www.binance.com/en/support/announcement/{code}", "t_pub": int(a["releaseDate"]), "raw": a}


def parse_okx(a: dict) -> dict:
    url = a.get("url") or ""
    return {"source": "okx", "id": url or f"{a.get('pTime')}|{a.get('title')}", "catalog": None, "title": a.get("title", ""),
            "url": url, "t_pub": int(a["pTime"]), "raw": a}


def parse_bybit(a: dict) -> dict:
    t = a.get("publishTime") or a.get("dateTimestamp")
    url = a.get("url") or ""
    return {"source": "bybit", "id": url or f"{t}|{a.get('title')}", "catalog": None, "title": a.get("title", ""),
            "url": url, "t_pub": int(t), "raw": a}


class AnnouncementStore:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)

    def path(self, source: str) -> Path:
        return self.root / f"{source}.jsonl"

    def load(self, source: str) -> list:
        p = self.path(source)
        if not p.exists():
            return []
        out = []
        for x in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(x))
            except Exception:
                continue
        return out

    def append(self, records, now: datetime) -> int:
        seen_ms = int(now.timestamp() * 1000)
        added = 0
        by_src = {}
        for r in records:
            by_src.setdefault(r["source"], []).append(r)
        for src, rs in by_src.items():
            have = {r["id"] for r in self.load(src)}
            new = []
            for r in rs:
                if r["id"] in have:
                    continue
                have.add(r["id"])
                new.append({**r, "t_seen": seen_ms})
            if new:
                self.root.mkdir(parents=True, exist_ok=True)
                with io.open(self.path(src), "a", encoding="utf-8") as fh:
                    for r in new:
                        fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                added += len(new)
        return added


MIGRATION_WINDOW_MS = 60 * 60_000


def drop_migrated(records, min_cluster: int = MIGRATION_CLUSTER, window_ms: int = MIGRATION_WINDOW_MS) -> list:
    """Bulk re-posts carry false publication times (OKX 2023-03-03 07:09..07:54). Any record inside a window of
    `window_ms` holding >= min_cluster records of one source is dropped (audit 10-28: one minute was too narrow)."""
    bad = set()
    by_src = {}
    for k, r in enumerate(records):
        by_src.setdefault(r["source"], []).append((r["t_pub"], k))
    for items in by_src.values():
        items.sort()
        j = 0
        for i in range(len(items)):
            if j < i:
                j = i
            while j + 1 < len(items) and items[j + 1][0] - items[i][0] < window_ms:
                j += 1
            if j - i + 1 >= min_cluster:
                bad.update(k for _, k in items[i:j + 1])
    return [r for k, r in enumerate(records) if k not in bad]


def backfill(fetch_page, parse, since_ms: int, max_pages: int = 500) -> list:
    out = []
    for p in range(1, max_pages + 1):
        raw = fetch_page(p) or []
        if not raw:
            break
        recs = [parse(a) for a in raw]
        out.extend(r for r in recs if r["t_pub"] >= since_ms)
        if min(r["t_pub"] for r in recs) < since_ms:
            break
    return out


# ------------------------------------------------------------------ labelling (spec §3)
_EXCLUDE_ASSET = re.compile(r"TradFi|Equity|Tokenized Stock|Pre-IPO|Index Perpetual|\bStocks?\b|bStocks Tokenized|Collateral",
                            re.I)


def classify(r: dict) -> str:
    t, src = r["title"], r["source"]
    if src == "binance":
        if r.get("catalog") == 161:
            return "BN_DELIST" if re.match(r"Binance Will Delist ", t) else "BN_OTHER"
        if (re.search(r"Binance Futures (Will Launch|Will Relaunch|to Relaunch)\b.*Perpetual", t)
                or re.search(r"Will Be Available on Binance .*Futures", t)):
            if _EXCLUDE_ASSET.search(t) or not extract_tickers(t, "BN_PERP_NEW"):
                return "BN_OTHER"
            return "BN_PERP_NEW"
        if re.search(r"Binance (Will Add|Adds) .* on (Earn|Buy Crypto|Convert)", t) or "Collateral" in t:
            return "BN_OTHER"
        if re.search(r"(Futures|Options) Will List", t):
            return "BN_OTHER"
        if (re.match(r"Binance (Lists|Will Continue to List)\b", t) or re.search(r"\bWill List\b", t)
                or re.match(r"Binance Exchange Adds .*Trading Pairs? on Binance Spot", t)
                or re.match(r"Introducing .* on Binance (HODLer Airdrops|Launchpool)", t)):
            return "BN_SPOT_NEW"
        if re.search(r"Margin", t) and re.match(r"Binance (Adds|Margin Will Add|Margin Adds)\b", t):
            return "BN_MARGIN"
        return "BN_OTHER"
    if src == "okx":
        if _EXCLUDE_ASSET.search(t):
            return "OKX_OTHER"
        if re.search(r"for spot trading", t) and re.match(r"OKX (to list|will launch)\b", t):
            return "OKX_SPOT_NEW"
        if re.match(r"OKX to list\b", t) and (re.search(r"perpetual( futures)? for .* crypto", t) or re.search(r"\bX-Perp\b", t)):
            return "OKX_PERP_NEW"
        return "OKX_OTHER"
    if src == "bybit":
        if _EXCLUDE_ASSET.search(t) or re.search(r"^Converting|Postponement|Token Splash|Convert &|Savings", t):
            return "BYBIT_OTHER"
        if (re.match(r"Bybit to List .* on\s+Spot\s*$", t.strip())
                or re.match(r"(New listing|Coming Soon)\s*:\s*[A-Z0-9]{1,15}/USDT\b", t, re.I)
                or re.match(r"Listing of [A-Z0-9]{1,15}/USDT\b", t)):
            return "BYBIT_SPOT_NEW"
        if (re.search(r"New listing\s*:", t, re.I) and re.search(r"Perpetual", t)
                or re.match(r"Listing of \w+USDT on Bybit Perpetual", t)
                or re.match(r"^(🔥\s*)?(New|Coming Soon:)\s*[A-Z0-9]+USDT\b.*Contract", t)):
            return "BYBIT_PERP_NEW"
        return "BYBIT_OTHER"
    return "OTHER"


_TICK = r"[A-Z0-9]{2,15}"


def _paren(t: str) -> list:
    """Tickers in parentheses; one letter allowed there ("SynFutures (F)", audit 10-28), digits-only never."""
    return [x for x in re.findall(r"\(([^()]+)\)", t) if re.fullmatch(r"[A-Z0-9]{1,15}", x) and not re.fullmatch(r"\d+", x)]


def _uniq(xs) -> list:
    out = []
    for x in xs:
        if x and x not in out:
            out.append(x)
    return out


def extract_tickers(title: str, etype: str) -> list:
    t = title
    if etype in ("BN_PERP_NEW",):
        a = re.findall(r"\b([A-Z0-9]{2,20})USDT\b", t)
        if a:
            return _uniq(a)
        m = re.search(r"USDⓈ-M ([A-Z0-9]{2,15}) Perpetual", t) or re.search(r"\b([A-Z0-9]{2,15}) USDT-Margined Perpetual", t)
        if m:
            return [m.group(1)]
        return _uniq(_paren(t))                       # "Yei Finance (CLO) Will Be Available on ... Binance Futures"
    if etype in ("BN_SPOT_NEW", "OKX_SPOT_NEW", "BYBIT_SPOT_NEW"):
        p = _paren(t)
        if p:
            return _uniq(p)
        m = (re.search(r"launch ([A-Z0-9]{2,15})/USD", t) or re.search(r"\bto list ([A-Z0-9]{2,15})\b", t)
             or re.search(r"(?:New listing|Coming Soon)\s*:\s*([A-Z0-9]{1,15})/USDT\b", t, re.I)
             or re.match(r"Listing of ([A-Z0-9]{1,15})/USDT\b", t))
        return [m.group(1)] if m else []
    if etype == "BN_MARGIN":
        m = re.match(r"Binance (?:Margin )?Adds (.+?)\s+(?:&|on\b)", t)
        if not m:
            return []
        parts = re.split(r",\s*|\s+and\s+", m.group(1))
        return _uniq(x.strip() for x in parts if re.fullmatch(_TICK, x.strip()))
    if etype == "BN_DELIST":
        m = re.match(r"Binance Will Delist (.+?) on \d{4}-\d{2}-\d{2}", t)
        if not m:
            return []
        return _uniq(x.strip() for x in re.split(r",\s*|\s+and\s+", m.group(1)) if re.fullmatch(_TICK, x.strip()))
    if etype == "OKX_PERP_NEW":
        m = re.search(r"perpetual(?: futures)? for (.+?) crypto", t)
        if m:
            body = re.sub(r"\([^)]*\)", "", m.group(1))
            # one letter allowed here: "perpetual for S crypto" (Sonic, audit 10-28)
            return _uniq(x.strip() for x in re.split(r",\s*|\s+and\s+", body) if re.fullmatch(r"[A-Z0-9]{1,15}", x.strip()))
        m = re.search(r"\b([A-Z0-9]{2,15})USD X-Perp", t)
        return [m.group(1)] if m else []
    if etype == "BYBIT_PERP_NEW":
        return _uniq(re.findall(r"\b([A-Z0-9]{2,20})USDT\b", t))
    return []


def to_spot(tickers, etype: str, spot_syms) -> list:
    import funding_store as FS
    spot = set(spot_syms)
    out = []
    for tk in tickers:
        sym = f"{tk}USDT"
        if sym in spot:
            out.append(sym)
            continue
        if etype.endswith("PERP_NEW"):
            m = FS.perp_to_spot(sym, spot)
            if m:
                out.append(m[0])
    return _uniq(out)


# ------------------------------------------------------------------ timing helpers (spec §4)
def tradable(sym: str, t_pub: int, first_bar_ms: dict) -> bool:
    fb = first_bar_ms.get(sym)
    return fb is not None and fb + BAR_MS <= t_pub


def entry_bar_open(t_pub: int, delay_ms: int = 60_000) -> int:
    x = t_pub + delay_ms
    return -(-x // BAR_MS) * BAR_MS


def pre_return(arr, t_pub: int, hours: float):
    t = np.asarray(arr["t"], dtype=np.int64)
    c = np.asarray(arr["c"], dtype=float)
    i = int(np.searchsorted(t, t_pub - BAR_MS, "right")) - 1        # last bar closed by t_pub
    j = int(np.searchsorted(t, t_pub - BAR_MS - int(hours * HOUR_MS), "right")) - 1
    if i < 0 or j < 0 or not (math.isfinite(c[i]) and math.isfinite(c[j])) or c[j] <= 0:
        return None
    return float(c[i] / c[j] - 1)


def _month_bounds(t: int):
    d = datetime.fromtimestamp(t / 1000, timezone.utc)
    lo = datetime(d.year, d.month, 1, tzinfo=timezone.utc)
    hi = datetime(d.year + (d.month == 12), d.month % 12 + 1, 1, tzinfo=timezone.utc)
    return int(lo.timestamp() * 1000), int(hi.timestamp() * 1000)


def control_time(sym: str, t_pub: int, events_by_sym: dict, rnd: random.Random, tries: int = 500):
    lo, hi = _month_bounds(t_pub)
    evs = events_by_sym.get(sym, [])
    for _ in range(tries):
        x = lo + rnd.randrange((hi - lo) // BAR_MS) * BAR_MS
        if all(abs(x - e) >= CONTROL_GAP_MS for e in evs):
            return x
    return None


def control_coin(t_pub: int, coins, events_by_sym: dict, rnd: random.Random):
    pool = sorted(s for s in coins if all(abs(t_pub - e) >= CONTROL_GAP_MS for e in events_by_sym.get(s, [])))
    return rnd.choice(pool) if pool else None


# ------------------------------------------------------------------ fetching (LA-1)
def _get(url: str):
    import audit as A
    import exchange_client as X
    host = urllib.parse.urlparse(url).netloc
    if host not in X.PUBLIC_HOSTS:
        raise X.HostNotAllowed(f"request to {host!r} refused")
    t0 = time.time()
    ok, status, err = False, None, None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"}),
                                    timeout=30) as r:
            status = r.status
            data = json.load(r)
            ok = True
            return data
    except Exception as ex:
        err = repr(ex)[:200]
        raise
    finally:
        try:
            w = A.default_writer("exchange")
            if w is not None:
                w.write({"host": host, "method": "GET", "path": urllib.parse.urlparse(url).path, "signed": False,
                         "ok": ok, "status": status, "error_msg": err, "latency_ms": round((time.time() - t0) * 1000, 1)})
        except Exception:
            pass


def fetchers() -> dict:
    """source -> (fetch_page(page) -> raw list, parse)."""
    def bn(cat):
        def f(p):
            time.sleep(PAUSE_SEC)
            d = _get(f"{HOSTS['binance']}/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId={cat}"
                     f"&pageNo={p}&pageSize=20")
            cats = (d.get("data") or {}).get("catalogs") or []
            return cats[0].get("articles", []) if cats else []
        return f, (lambda a, c=cat: parse_binance(a, c))

    def okx(p):
        time.sleep(PAUSE_SEC)
        d = _get(f"{HOSTS['okx']}/api/v5/support/announcements?annType=announcements-new-listings&page={p}")
        data = d.get("data") or []
        return data[0].get("details", []) if data else []

    def bybit(p):
        time.sleep(PAUSE_SEC)
        d = _get(f"{HOSTS['bybit']}/v5/announcements/index?locale=en-US&type=new_crypto&limit=20&page={p}")
        return (d.get("result") or {}).get("list", [])
    out = {f"binance_{c}": bn(c) for c in BINANCE_CATALOGS}
    out["okx"] = (okx, parse_okx)
    out["bybit"] = (bybit, parse_bybit)
    return out


def run_backfill(since: str = "2023-01-01", root: Path = ROOT) -> dict:
    since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    st = AnnouncementStore(root)
    res = {}
    for name, (f, parse) in fetchers().items():
        t0 = time.time()
        try:
            recs = backfill(f, parse, since_ms)
            kept = drop_migrated(recs)
            res[name] = {"fetched": len(recs), "dropped_migrated": len(recs) - len(kept),
                         "added": st.append(kept, datetime.now(timezone.utc)),
                         "oldest": _iso_ms(min(r["t_pub"] for r in kept)) if kept else None,
                         "sec": round(time.time() - t0, 1)}
        except Exception as ex:
            res[name] = {"error": repr(ex)[:300]}
        print(name, json.dumps(res[name], ensure_ascii=False), flush=True)
    return res


def run_step(root: Path = ROOT) -> dict:
    st = AnnouncementStore(root)
    res = {}
    for name, (f, parse) in fetchers().items():
        try:
            recs = drop_migrated([parse(a) for a in f(1)])
            res[name] = st.append(recs, datetime.now(timezone.utc))
        except Exception as ex:
            res[name] = "error: " + repr(ex)[:200]
    return res


def status(root: Path = ROOT) -> dict:
    st = AnnouncementStore(root)
    out = {}
    for src in ("binance", "okx", "bybit"):
        rows = st.load(src)
        types = {}
        for r in rows:
            k = classify(r)
            types[k] = types.get(k, 0) + 1
        lag = sorted(r["t_seen"] - r["t_pub"] for r in rows if r.get("t_seen") and r["t_seen"] - r["t_pub"] < 2 * DAY_MS)
        out[src] = {"rows": len(rows), "types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
                    "oldest": _iso_ms(min(r["t_pub"] for r in rows)) if rows else None,
                    "lag_p50_sec": lag[len(lag) // 2] / 1000 if lag else None}
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import audit as A
    A.set_process("announcements")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "backfill":
        since = argv[argv.index("--since") + 1] if "--since" in argv else "2023-01-01"
        run_backfill(since)
        return 0
    if cmd == "step":
        if not getattr(TC, "ANNOUNCEMENTS_ENABLED", False):
            print("ANNOUNCEMENTS_ENABLED is False -- nothing done")
            return 0
        print(json.dumps(run_step(), ensure_ascii=False))
        return 0
    if cmd == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=1))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
