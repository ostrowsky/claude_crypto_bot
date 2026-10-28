"""W1 -- shadow scanner of the whole Binance spot market (wide-scan-shadow-w1-1028-spec.md).

The trader must watch every coin tradable on Binance spot (operator decision #2, 2026-10-14). The executor still takes
its entries from the signal bot's 93 coins; W1 closes the gap without money at risk: every closed 15m bar it evaluates
all pairs of the universe, keeps their candles, writes candidates and runs a paper book on them. Public GET requests
only (no key, no order); writes only trader/runtime/scan/ and trader/runtime/learning/wide_candidates.jsonl.

  sources   wide_gen  the B2 generator (candidate_gen.generate = the bot's own entry rules) on 15m and 1h
            leader    first time in a UTC day among the top-3 of the universe by return since the open, >= +5%
            surge     15m quote volume >= 5x the median of the previous 96 bars and bar return >= +2%, once a day
            leader / surge are written for MEASUREMENT only (event "flag"); only wide_gen feeds the book
  book      x4_wide: the E7 engine (shadow_portfolio.ShadowEngine) with the X4 policy on wide_gen entries
  learning  one row per event with features known at its bar; outcomes appended once, 24 h later

  (from trader/) ../pyembed/python.exe -u wide_scan.py step       scan + book + outcomes (run_shadow.cmd)
  (from trader/) ../pyembed/python.exe wide_scan.py report [DAY]   coverage, how early leaders were seen, book money
  (from trader/) ../pyembed/python.exe wide_scan.py readout        READOUT_W1.json (spec §4)
Rollback: trader_config.WIDE_SCAN_ENABLED = False (the step then does nothing).
"""
from __future__ import annotations

import io
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import history_store as HS        # noqa: E402
import trader_config as TC        # noqa: E402

ROOT = TC.RUNTIME / "scan"
LEARNING = TC.RUNTIME / "learning" / "wide_candidates.jsonl"
KLINES_HOST = "https://api.binance.com"
BAR_MS = 15 * 60 * 1000
DAY_MS = 86_400_000
FIELDS = HS.FIELDS
SEED_BARS = 1500
FETCH_BACK_BARS = 1000
GEN_BARS_15M = 250                 # >= 4x the bot's slowest EMA (50): the seed's weight at the last bar is ~e^-8
GEN_BARS_FOR_1H = 4 * 250
FETCH_WORKERS = 8
GEN_WORKERS = 6                    # process pool for the B2 generator (8 cores; speed, journal 10-28)
LEADER_TOP = 3
LEADER_MIN_RET = 0.05
SURGE_MULT = 5.0
SURGE_MIN_RET = 0.02
SURGE_LOOKBACK = 96
CROSS = 0.025
TOP_N_DAY = 20
MIN_PAIR_SHARE = 0.5
RATE_PER_SEC = 15.0                # klines weight 2 -> 1 800 / min, 30% of REQUEST_WEIGHT 6 000
KEYS_MAX = 20_000
READOUT_NOT_BEFORE = "2026-12-01"
READOUT_MIN_DAYS = 30
COUNTED_COVERAGE = 0.95


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _ts_ms(ts) -> int:
    if isinstance(ts, datetime):
        return _ms(ts)
    return _ms(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))


# ------------------------------------------------------------------ bar store (spec §2.2)
class BarStore:
    """trader/runtime/scan/bars/<SYM>_15m.npz in the history_store layout; append-only, closed bars only."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache = {}

    def path(self, sym: str) -> Path:
        return self.root / f"{sym}_15m.npz"

    def load(self, sym: str):
        if sym in self._cache:
            return self._cache[sym]
        p = self.path(sym)
        a = None
        if p.exists():
            with np.load(p, allow_pickle=False) as z:           # closed at once: Windows keeps open files locked
                a = {k: z[k] for k in FIELDS}
        self._cache[sym] = a
        return a

    def last_t(self, sym: str):
        a = self.load(sym)
        return int(a["t"][-1]) if a is not None and len(a["t"]) else None

    def syms(self) -> list:
        return sorted(p.name[:-len("_15m.npz")] for p in self.root.glob("*_15m.npz")) if self.root.exists() else []

    def _write(self, sym, a):
        """Uncompressed npz (speed, journal 10-28): loading 508 compressed files cost ~37 s a step."""
        p = self.path(sym)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, **{k: a[k] for k in FIELDS})
        tmp.replace(p)
        self._cache[sym] = a

    def append(self, sym: str, rows, now: datetime) -> int:
        now_ms = _ms(now)
        keep = [r for r in (rows or []) if int(r[0]) + BAR_MS <= now_ms]
        if not keep:
            return 0
        new = {"t": np.array([int(r[0]) for r in keep], dtype=np.int64)}
        for k, col in (("o", 1), ("h", 2), ("l", 3), ("c", 4), ("v", 5)):
            new[k] = np.array([float(r[col]) for r in keep])
        old = self.load(sym)
        last = int(old["t"][-1]) if old is not None and len(old["t"]) else None
        if last is not None:
            m = new["t"] > last                      # append-only: nothing before the stored tail is rewritten
            new = {k: v[m] for k, v in new.items()}
            if not len(new["t"]):
                return 0
        merged, added = HS.merge(old, new)
        if added:
            self._write(sym, merged)
        return added

    def seed(self, sym: str, arr, n: int) -> int:
        if self.last_t(sym) is not None or arr is None or not len(arr["t"]):
            return 0
        a = {k: np.asarray(arr[k])[-n:] for k in FIELDS}
        a["t"] = a["t"].astype(np.int64)
        self._write(sym, a)
        return len(a["t"])


# ------------------------------------------------------------------ detectors (spec §2.3)
def ret_since_open(arr, i: int):
    t = np.asarray(arr["t"], dtype=np.int64)
    d0 = int(t[i]) // DAY_MS * DAY_MS
    j = int(np.searchsorted(t, d0, "left"))
    if j >= len(t) or int(t[j]) != d0 or j > i:
        return None
    o = float(arr["o"][j])
    return float(arr["c"][i]) / o - 1 if o > 0 else None


def detect_leader(rets: dict, flagged, top: int = LEADER_TOP, min_ret: float = LEADER_MIN_RET) -> list:
    ranked = sorted(((v, s) for s, v in rets.items() if v is not None), key=lambda x: (-x[0], x[1]))[:top]
    return [s for v, s in ranked if v >= min_ret and s not in flagged]


def detect_surge(arr, i: int) -> bool:
    if i < SURGE_LOOKBACK or i < 1:
        return False
    c, v = np.asarray(arr["c"], float), np.asarray(arr["v"], float)
    qv = c[i - SURGE_LOOKBACK:i] * v[i - SURGE_LOOKBACK:i]
    med = float(np.median(qv))
    if not (med > 0 and c[i - 1] > 0):
        return False
    return c[i] * v[i] >= SURGE_MULT * med and c[i] / c[i - 1] - 1 >= SURGE_MIN_RET


# ------------------------------------------------------------------ learning rows (spec §2.5)
def event_key(e: dict) -> str:
    return "%s|%s|%s|%s" % (e.get("source"), e.get("sym"), e.get("tf"), e.get("ts"))


def candidate_row(event: dict, arr, i: int, rank, n_universe, in_watchlist) -> dict:
    c = np.asarray(arr["c"], float)
    h, l, v = np.asarray(arr["h"], float), np.asarray(arr["l"], float), np.asarray(arr["v"], float)
    qv = c * v
    prev = qv[max(0, i - SURGE_LOOKBACK):i]
    med = float(np.median(prev)) if len(prev) else float("nan")
    rng = (h[max(0, i - 13):i + 1] - l[max(0, i - 13):i + 1])

    def r(k):
        return float(c[i] / c[i - k] - 1) if i - k >= 0 and c[i - k] > 0 else None
    return {"kind": "candidate", "key": event_key(event), "source": event.get("source"), "sym": event.get("sym"),
            "tf": event.get("tf"), "ts": event.get("ts"), "price": float(c[i]), "bar_open_ms": int(arr["t"][i]),
            "ret_since_open": ret_since_open(arr, i), "rank": rank, "n_universe": n_universe,
            "ret_15m": r(1), "ret_1h": r(4), "ret_4h": r(16),
            "qv_15m_ratio": float(qv[i] / med) if med and math.isfinite(med) and med > 0 else None,
            "atr_pct": float(np.mean(rng) / c[i]) if len(rng) and c[i] > 0 else None,
            "in_watchlist": bool(in_watchlist)}


def learning_path(root: Path) -> Path:
    return LEARNING if Path(root) == ROOT else Path(root) / "learning" / "wide_candidates.jsonl"


def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    for x in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(x))
        except Exception:
            continue
    return out


def append_candidates(root: Path, rows) -> int:
    p = learning_path(root)
    have = {r.get("key") for r in _read_jsonl(p) if r.get("kind") == "candidate"}
    new = [r for r in rows if r.get("key") not in have]
    if not new:
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with io.open(p, "a", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps({**r, "recorded_utc": rec}, ensure_ascii=False, default=str) + "\n")
    return len(new)


def label_outcomes(root: Path, now: datetime) -> int:
    p = learning_path(root)
    rows = _read_jsonl(p)
    done = {r.get("key") for r in rows if r.get("kind") == "outcome"}
    store = BarStore(Path(root) / "bars")
    now_ms = _ms(now)
    leaders_cache, out = {}, []
    for r in rows:
        if r.get("kind") != "candidate" or r.get("key") in done:
            continue
        t_i = r.get("bar_open_ms")
        if t_i is None or t_i + BAR_MS + DAY_MS > now_ms:
            continue
        a = store.load(r["sym"])
        if a is None:
            continue
        t = np.asarray(a["t"], dtype=np.int64)
        i = int(np.searchsorted(t, t_i))
        j = int(np.searchsorted(t, t_i + 96 * BAR_MS))
        if i >= len(t) or int(t[i]) != t_i or j >= len(t) or int(t[j]) != t_i + 96 * BAR_MS:
            if now_ms - t_i > 3 * DAY_MS:
                out.append({"kind": "outcome", "key": r["key"], "missing_bars": True})
            continue
        c0 = float(a["c"][i])
        k4 = int(np.searchsorted(t, t_i + 16 * BAR_MS))
        day = _day(t_i)
        if day not in leaders_cache:
            leaders_cache[day] = leaders_of_day(store, store.syms(), day)
        top = leaders_cache[day]
        out.append({"kind": "outcome", "key": r["key"],
                    "fwd_4h": float(a["c"][k4]) / c0 - 1 if k4 < len(t) and int(t[k4]) == t_i + 16 * BAR_MS else None,
                    "fwd_24h": float(a["c"][j]) / c0 - 1,
                    "max_24h": float(np.max(a["h"][i + 1:j + 1])) / c0 - 1,
                    "min_24h": float(np.min(a["l"][i + 1:j + 1])) / c0 - 1,
                    "top20_day": (r["sym"] in top) if top is not None else None})
    if out:
        p.parent.mkdir(parents=True, exist_ok=True)
        with io.open(p, "a", encoding="utf-8") as fh:
            for o in out:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    return len(out)


# ------------------------------------------------------------------ leaders of the day (spec §3.2)
def _day_slice(a, day: str):
    d0 = _ms(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    t = np.asarray(a["t"], dtype=np.int64)
    lo, hi = int(np.searchsorted(t, d0)), int(np.searchsorted(t, d0 + DAY_MS))
    return lo, hi, d0


def day_returns(store: BarStore, syms, day: str) -> dict:
    out = {}
    for s in syms:
        a = store.load(s)
        if a is None:
            continue
        lo, hi, d0 = _day_slice(a, day)
        if hi - lo != 96 or int(a["t"][lo]) != d0:
            continue
        o = float(a["o"][lo])
        if o > 0:
            out[s] = float(a["c"][hi - 1]) / o - 1
    return out


def leaders_of_day(store: BarStore, syms, day: str, top: int = TOP_N_DAY):
    syms = list(syms)
    rets = day_returns(store, syms, day)
    if not rets or len(rets) < MIN_PAIR_SHARE * len(syms):
        return None                                  # not enough data: "no data", never "missed" (TH-05)
    return [s for s, _ in sorted(rets.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]


def _ret_at(store: BarStore, sym: str, ts_ms: int):
    a = store.load(sym)
    if a is None:
        return None
    t = np.asarray(a["t"], dtype=np.int64)
    i = int(np.searchsorted(t, ts_ms - BAR_MS, "right")) - 1      # last bar with close <= ts
    if i < 0:
        return None
    return ret_since_open(a, i)


def early_seen(leaders, events, store: BarStore, day: str) -> dict:
    ev = [e for e in events if str(e.get("ts", ""))[:10] == day]
    first = {}
    for e in sorted(ev, key=lambda e: str(e["ts"])):
        first.setdefault(e["sym"], e)
    flagged = set(first)
    n_pairs = len(day_returns(store, store.syms(), day)) or None
    dr = day_returns(store, leaders, day)
    before, done = 0, []
    for s in leaders:
        if s not in first:
            continue
        r = _ret_at(store, s, _ts_ms(first[s]["ts"]))
        if r is not None and r < CROSS:
            before += 1
        if r is not None and dr.get(s) and dr[s] > 0:
            done.append(max(0.0, r) / dr[s])
    hit = len(flagged & set(leaders))
    return {"day": day, "n_leaders": len(leaders), "n_flagged_leaders": hit, "flagged_before_cross": before,
            "share_before_cross": before / len(leaders) if leaders else None,
            "n_flagged_pairs": len(flagged), "flagged_hit_share": hit / len(flagged) if flagged else None,
            "n_pairs_day": n_pairs, "base_rate": (len(leaders) / n_pairs) if n_pairs else None,
            "move_done_at_flag_median": statistics.median(done) if done else None}


# ------------------------------------------------------------------ book (spec §2.4)
def _policies():
    import shadow_portfolio as SP
    return {"x4_wide": replace(SP.X4, name="x4_wide")}


POLICIES = _policies()


def book_events(events) -> list:
    return [e for e in events if e.get("event") == "entry" and e.get("source") == "wide_gen"]


_TF_ORDER = {"15m": 0, "1h": 1}


def book_filter(events, held) -> tuple:
    """One position per coin, as the executor (reason asset_held; defect 2026-10-28: a 15m and a 1h entry of one coin on
    the same bar opened two positions). Earliest event per coin wins, 15m before 1h on a tie; time order is kept.
    -> (kept, [(event, reason)])"""
    held = set(held)
    kept, skipped = [], []
    for e in sorted(events, key=lambda e: (str(e.get("ts")), _TF_ORDER.get(e.get("tf"), 9), str(e.get("sym")))):
        if e.get("sym") in held:
            skipped.append((e, "asset_held"))
            continue
        held.add(e.get("sym"))
        kept.append(e)
    return kept, skipped


# ------------------------------------------------------------------ step (spec §2.3)
def default_gen():
    import candidate_gen as CG
    return CG.generate


def _load_state(root: Path) -> dict:
    p = Path(root) / "state.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_state(root: Path, st: dict) -> None:
    p = Path(root) / "state.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _tail(a, n):
    return {k: np.asarray(a[k])[-n:] for k in FIELDS}


_GEN = None


def _gen_init():
    global _GEN
    import candidate_gen as CG
    rules, cfg = CG.bot_rules()
    _GEN = (CG.generate, rules, cfg)


def _gen_task(task):
    sym, tf, arr, now_iso, start_ms = task
    gen, rules, cfg = _GEN
    return gen(sym, tf, arr, cfg=cfg, rules=rules, now=datetime.fromisoformat(now_iso), start_ms=start_ms)


def _run_gen(tasks, now, gen_fn, workers):
    if gen_fn is not None or not workers or workers <= 1 or len(tasks) < 2:
        g = gen_fn or default_gen()
        return [g(sym, tf, arr, now=now, start_ms=sm) or [] for sym, tf, arr, sm in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_gen_init) as ex:
        return [r or [] for r in ex.map(_gen_task, [(sym, tf, arr, now.isoformat(), sm) for sym, tf, arr, sm in tasks],
                                        chunksize=8)]


def run_step(now: datetime, fetch, universe, root: Path, gen_fn=None, enabled: bool = True, watchlist=(),
             seed_dir: Path | None = None, workers: int | None = None) -> dict:
    out = {"enabled": bool(enabled), "same_bar": False, "evaluated": 0, "errors": {}, "events": 0, "universe": len(universe)}
    if not enabled:
        return out
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    st = _load_state(root)
    now_ms = _ms(now)
    bar_close = now_ms // BAR_MS * BAR_MS                    # close time of the latest closed 15m bar
    if st.get("last_bar_close") == bar_close:
        out["same_bar"] = True
        return out
    store = BarStore(root / "bars")
    tasks, per_sym = [], {}
    done = st.setdefault("done", {})
    wl = set(watchlist)
    events, latest = [], {}
    starts = {}
    for sym in universe:
        try:
            if seed_dir is not None and store.last_t(sym) is None:
                store.seed(sym, HS.load(Path(seed_dir) / f"{sym}_15m.npz"), SEED_BARS)
            last = store.last_t(sym)
            starts[sym] = last + BAR_MS if last is not None else now_ms - FETCH_BACK_BARS * BAR_MS
        except Exception as ex:
            out["errors"][sym] = repr(ex)[:200]
    fetched = {}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futs = {sym: pool.submit(fetch, sym, st0) for sym, st0 in starts.items()}
        for sym, f in futs.items():
            try:
                fetched[sym] = f.result()
            except Exception as ex:
                out["errors"][sym] = repr(ex)[:200]
    for sym in universe:
        if sym not in fetched:
            continue
        try:
            store.append(sym, fetched[sym], now)
            a = store.load(sym)
            if a is None or not len(a["t"]):
                continue
            last_close = int(a["t"][-1]) + BAR_MS
            prev = done.get(sym)
            if prev is not None and last_close <= prev:
                continue
            out["evaluated"] += 1
            start_ms = prev + 1 if prev is not None else last_close
            n = len(a["t"])
            t15 = _tail(a, GEN_BARS_15M)
            a1 = HS.agg_1h(_tail(a, GEN_BARS_FOR_1H), complete_only=True)
            a1 = {k: a1[k] for k in FIELDS}
            new_hour = bool(len(a1["t"])) and (prev is None or int(a1["t"][-1]) + 4 * BAR_MS > prev)
            for tf, arr in (("15m", t15), ("1h", a1)):
                if not len(arr["t"]) or (tf == "1h" and not new_hour):
                    continue
                tasks.append((sym, tf, arr, start_ms))
            per_sym[sym] = []
            t = np.asarray(a["t"], dtype=np.int64)
            i0 = int(np.searchsorted(t, (prev if prev is not None else last_close) - BAR_MS + 1)) \
                if prev is not None else n - 1
            sday = st.setdefault("surge_day", {})
            for i in range(max(i0, 0), n):
                d = _day(int(t[i]))
                if sday.get(sym) != d and detect_surge(a, i):
                    sday[sym] = d
                    per_sym.setdefault(sym, []).append({"event": "flag", "source": "surge", "sym": sym, "tf": "15m",
                                   "ts": _iso(int(t[i]) + BAR_MS), "price": float(a["c"][i])})
            done[sym] = last_close
            latest[sym] = (a, n - 1)
        except Exception as ex:
            out["errors"][sym] = repr(ex)[:200]
    gen_out = _run_gen(tasks, now, gen_fn, workers if workers is not None else (None if gen_fn else GEN_WORKERS))
    by_sym = {}
    for (sym, tf, arr, sm), evs in zip(tasks, gen_out):
        by_sym.setdefault(sym, []).extend({**e, "source": "wide_gen"} for e in evs)
    for sym in universe:                           # per pair: its generator events, then its surge flags
        events.extend(by_sym.get(sym, []))
        events.extend(per_sym.get(sym, []))
    # leaders on the latest closed bar
    rets = {s: ret_since_open(a, i) for s, (a, i) in latest.items() if int(a["t"][i]) + BAR_MS == bar_close}
    day = _day(bar_close - BAR_MS)
    lead = st.get("leader", {})
    flagged = set(lead.get(day, []))
    new = detect_leader(rets, flagged)
    for s in new:
        a, i = latest[s]
        events.append({"event": "flag", "source": "leader", "sym": s, "tf": "15m", "ts": _iso(bar_close),
                       "price": float(a["c"][i]), "ret_since_open": rets[s]})
    st["leader"] = {day: sorted(flagged | set(new))}
    # write events once (key) and their learning rows
    keys = st.setdefault("keys", [])
    seen = set(keys)
    ranked = {s: k + 1 for k, (s, _) in enumerate(sorted(((s, v) for s, v in rets.items() if v is not None),
                                                         key=lambda kv: (-kv[1], kv[0])))}
    fresh, rows = [], []
    for e in events:
        k = event_key(e)
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
        fresh.append(e)
        a = store.load(e["sym"])
        tt = np.asarray(a["t"], dtype=np.int64)
        i = int(np.searchsorted(tt, _ts_ms(e["ts"]) - BAR_MS))
        if i < len(tt) and int(tt[i]) + BAR_MS == _ts_ms(e["ts"]):
            on_latest = int(tt[i]) + BAR_MS == bar_close
            rows.append(candidate_row(e, a, i, ranked.get(e["sym"]) if on_latest else None, len(universe),
                                      e["sym"] in wl))
    st["keys"] = keys[-KEYS_MAX:]
    if fresh:
        with io.open(root / "events.jsonl", "a", encoding="utf-8") as fh:
            for e in fresh:
                fh.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        append_candidates(root, rows)
    out["events"] = len(fresh)
    out["latency_sec"] = (now_ms - bar_close) / 1000
    st["last_bar_close"] = bar_close
    st["last_step_utc"] = now.isoformat()
    _save_state(root, st)
    with io.open(root / "steps.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now.isoformat(), "bar_close": _iso(bar_close), "universe": len(universe),
                             "evaluated": out["evaluated"], "fresh_pairs": len(rets), "errors": len(out["errors"]),
                             "events": out["events"], "latency_sec": out["latency_sec"]}) + "\n")
    return out


# ------------------------------------------------------------------ live wiring
def live_fetch(client):
    import threading
    gap = 1.0 / RATE_PER_SEC
    nxt = [0.0]
    lock = threading.Lock()

    def fetch(sym, start_ms):
        with lock:                                   # one shared rate limit across the fetch threads
            t = max(time.monotonic(), nxt[0])
            nxt[0] = t + gap
        w = t - time.monotonic()
        if w > 0:
            time.sleep(w)
        return client.klines_from(sym, "15m", start_ms, limit=1000, base=KLINES_HOST)
    return fetch


def live_universe(root: Path, client, now: datetime) -> list:
    """Re-read once a UTC day from the demo exchangeInfo (spec §2.1); changes logged; fallback: trading/universe.json."""
    p = Path(root) / "universe.json"
    day = now.strftime("%Y-%m-%d")
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    if old and old.get("day") == day:
        return old["pairs"]
    try:
        import trading_universe as TU
        pairs = sorted(x["symbol"] for x in TU.build(client.exchange_info())["pairs"])
    except Exception as ex:
        if old:
            return old["pairs"]
        u = json.loads((TC.RUNTIME / "trading" / "universe.json").read_text(encoding="utf-8"))
        pairs = sorted(x["symbol"] for x in u["pairs"])
        print("universe: exchangeInfo failed (%r), fallback to trading/universe.json" % ex)
    if old:
        add, gone = sorted(set(pairs) - set(old["pairs"])), sorted(set(old["pairs"]) - set(pairs))
        if add or gone:
            with io.open(Path(root) / "universe_changes.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"day": day, "added": add, "gone": gone, "reason": "demo exchangeInfo"}) + "\n")
    Path(root).mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"day": day, "pairs": pairs}), encoding="utf-8")
    return pairs


def bot_symbols(max_bytes: int = 30_000_000) -> set:
    """Symbols the signal bot evaluates, from the tail of files/bot_events.jsonl (read-only)."""
    p = TC.BOT_EVENTS
    if not p.exists():
        return set()
    out = set()
    with open(p, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        fh.readline()
        for raw in fh:
            k = raw.find(b'"sym": "')
            if k >= 0:
                e = raw.find(b'"', k + 8)
                out.add(raw[k + 8:e].decode("ascii", "replace"))
    return out


def reset_book(root: Path = ROOT, stamp: str | None = None) -> dict:
    """Archive the x4_wide state and ledger (never deleted) and start a new book that reads events only after now."""
    import shadow_portfolio as SP
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    sd = Path(root) / "shadow"
    sd.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, ext in (("state", ".json"), ("ledger", ".jsonl")):
        p = sd / f"{name}{ext}"
        if p.exists():
            p.replace(sd / f"{name}.pre_reset_{stamp}{ext}")
            n += 1
    evp = Path(root) / "events.jsonl"
    size = evp.stat().st_size if evp.exists() else 0
    now = datetime.now(timezone.utc).isoformat()
    SP.save_state(sd / "state.json", {"tail": {"offset": size, "seen": []}, "books": None, "started_utc": now,
                                      "reset_utc": now})
    return {"archived": n, "offset": size, "stamp": stamp}


def book_step(root: Path, now: datetime) -> dict:
    import exchange_client as X
    import fill_model as FM
    import shadow_portfolio as SP
    TC.bot_readonly()
    import config as BOT
    import exit_validator as EV
    sd = Path(root) / "shadow"
    sd.mkdir(parents=True, exist_ok=True)
    state_p, ledger_p = sd / "state.json", sd / "ledger.jsonl"
    st = SP.load_state(state_p)
    st = st or {"tail": {}, "books": None, "started_utc": now.isoformat()}
    evp = Path(root) / "events.jsonl"
    if not evp.exists():
        evp.touch()
    tail = SP.EventTail(evp, st["tail"])
    bars = SP.LiveBars(X.ExchangeClient(key="", secret=""), atr_period=int(getattr(BOT, "ATR_PERIOD", 14)))
    eng = SP.ShadowEngine(POLICIES, FM.model("taker", 1), floor_of=lambda m: EV.floor_of(m, BOT), bars=bars)
    if st.get("books"):
        eng.load(st["books"])
    for b in eng.books.values():
        if b.cur_day is None:
            b.cur_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            b.day_start_eq.setdefault(b.cur_day.strftime("%Y-%m-%d"), b.cash)
    held = {p["sym"] for b in eng.books.values() for p in b.positions}
    evs, skipped = book_filter(book_events(tail.read_new()), held)
    eng.process(evs, now)
    st["books"] = eng.to_state()
    st["last_step_utc"] = now.isoformat()
    rows = [{"type": "skip", "book": "x4_wide", "sym": e.get("sym"), "tf": e.get("tf"), "ts": e.get("ts"), "reason": r}
            for e, r in skipped] + list(eng.ledger)
    if rows:
        with io.open(ledger_p, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    SP.save_state(state_p, st)
    return {"events": len(evs), "skipped_held": len(skipped),
            "open": {n: len(b.positions) for n, b in eng.books.items()}}


class LockedWriter:
    """The fetch threads share one audit writer; each line is written under a lock (no interleaved appends)."""

    def __init__(self, inner):
        import threading
        self.inner, self.lock = inner, threading.Lock()

    def write(self, rec, now=None):
        if self.inner is None:
            return False
        with self.lock:
            return self.inner.write(rec, now)


def step(now: datetime | None = None) -> dict:
    import audit as A
    import exchange_client as X
    now = now or datetime.now(timezone.utc)
    if not getattr(TC, "WIDE_SCAN_ENABLED", False):
        return {"enabled": False}
    TC.bot_readonly()
    client = X.ExchangeClient(key="", secret="", audit=LockedWriter(A.default_writer("exchange")))
    universe = live_universe(ROOT, client, now)
    st = _load_state(ROOT)
    wl_day = now.strftime("%Y-%m-%d")
    if st.get("watchlist_day") != wl_day:
        st["watchlist"], st["watchlist_day"] = sorted(bot_symbols()), wl_day
        ROOT.mkdir(parents=True, exist_ok=True)
        _save_state(ROOT, st)
    t0 = time.time()
    res = run_step(now, live_fetch(client), universe, ROOT, watchlist=st.get("watchlist", []), seed_dir=HS.HIST_DIR)
    res["scan_sec"] = round(time.time() - t0, 1)
    if not res.get("same_bar"):
        res["book"] = book_step(ROOT, datetime.now(timezone.utc))
        res["outcomes"] = label_outcomes(ROOT, datetime.now(timezone.utc))
    return res


# ------------------------------------------------------------------ report / readout (spec §3, §4)
def _book_series(state_path: Path, book: str):
    import shadow_readout as SR
    if not state_path.exists():
        return []
    st = json.loads(state_path.read_text(encoding="utf-8"))
    b = (st.get("books") or {}).get(book) or {}
    return SR.series_from_marks(b.get("day_start_eq") or {}, st.get("started_utc"))


def coverage_by_day(root: Path = ROOT) -> dict:
    by = {}
    for r in _read_jsonl(Path(root) / "steps.jsonl"):
        d = r["bar_close"][:10]
        x = by.setdefault(d, {"steps": 0, "pairs": 0, "universe": 0, "lat": []})
        x["steps"] += 1
        x["pairs"] += r.get("fresh_pairs", 0)
        x["universe"] += r.get("universe", 0)
        x["lat"].append(r.get("latency_sec", 0))
    return {d: {"bars_share": x["steps"] / 96, "pair_share": x["pairs"] / x["universe"] if x["universe"] else 0.0,
                "latency_p50": statistics.median(x["lat"]), "latency_p95": sorted(x["lat"])[int(0.95 * (len(x["lat"]) - 1))]}
            for d, x in by.items()}


def report(day: str | None = None, root: Path = ROOT) -> dict:
    import daily_revenue as DR
    day = day or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    store = BarStore(Path(root) / "bars")
    cov = coverage_by_day(root).get(day)
    leaders = leaders_of_day(store, store.syms(), day)
    evs = _read_jsonl(Path(root) / "events.jsonl")
    es = early_seen(leaders, evs, store, day) if leaders else None
    wl = set(_load_state(root).get("watchlist", []))
    es_out = early_seen([s for s in leaders if s not in wl], evs, store, day) if leaders else None
    wide = dict(_book_series(Path(root) / "shadow" / "state.json", "x4_wide"))
    e7 = dict(_book_series(TC.RUNTIME / "shadow" / "state.json", "x4_floor8"))
    paired = DR.paired(list(wide.items()), list(e7.items())) if wide and e7 else None
    return {"day": day, "coverage": cov, "leaders": leaders, "early_seen": es, "early_seen_outside_bot": es_out,
            "x4_wide_day": wide.get(day), "x4_floor8_day": e7.get(day), "paired_all": paired,
            "disk_mb": round(sum(p.stat().st_size for p in (Path(root) / "bars").glob("*.npz")) / 1e6, 1)
            if (Path(root) / "bars").exists() else 0.0}


def readout(root: Path = ROOT, today: str | None = None) -> dict:
    import daily_revenue as DR
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cov = coverage_by_day(root)
    wide = dict(_book_series(Path(root) / "shadow" / "state.json", "x4_wide"))
    e7 = dict(_book_series(TC.RUNTIME / "shadow" / "state.json", "x4_floor8"))
    days = sorted(d for d in set(wide) & set(e7) if cov.get(d, {}).get("bars_share", 0) >= COUNTED_COVERAGE)
    out = {"spec": "wide-scan-shadow-w1-1028-spec.md", "today": today, "counted_days": len(days),
           "not_before": READOUT_NOT_BEFORE, "min_days": READOUT_MIN_DAYS}
    bad = [d for d, x in cov.items() if x["pair_share"] < 0.90 and d < today]
    if bad:
        out["anomaly"] = {"low_coverage_days": bad}
    if today < READOUT_NOT_BEFORE or len(days) < READOUT_MIN_DAYS:
        out["verdict"] = "COLLECTING"
    else:
        pr = DR.paired([(d, wide[d]) for d in days], [(d, e7[d]) for d in days])
        out["paired"] = pr
        out["verdict"] = ("FOR_OPERATOR_BUILD_B4" if pr["lo95"] > 0 else
                          "B3_CONFIRMED_LIVE" if pr["hi95"] < 0 else "INCONCLUSIVE")
    if "anomaly" in out:
        out["verdict"] = "ANOMALY" if out["verdict"] == "COLLECTING" else out["verdict"] + "+ANOMALY"
    (Path(root)).mkdir(parents=True, exist_ok=True)
    (Path(root) / "READOUT_W1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import audit as A
    A.set_process("wide_scan")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "step":
        import faulthandler
        faulthandler.dump_traceback_later(540, exit=True)        # task CryptoBot_TraderWideScan limit: 10 min
        print(json.dumps(step(), ensure_ascii=False, default=str)[:2000])
        return 0
    if cmd == "report":
        print(json.dumps(report(argv[1] if len(argv) > 1 else None), ensure_ascii=False, indent=1, default=str))
        return 0
    if cmd == "readout":
        print(json.dumps(readout(), ensure_ascii=False, indent=1, default=str))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
