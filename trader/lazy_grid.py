"""Lazy bar grid for the B3 replay on the whole universe (wide-universe-b-1014-spec.md §4a).

A dense float matrix of ~700 pairs x ~130k 15m bars does not fit next to two running bots, so each pair is
kept on disk as its own dense array from its first to its last bar (missing bar = NaN, as in the bot's grid)
and read through np.load(mmap_mode="r"). The interface is what execution_replay.run and
trading_policy.exit_plan read: syms, col, t0, N, idx(dt), C/O/H/L/V/ATR[ix, j]; outside a pair's range -> NaN.
ATR is the bot's own indicators._atr over the pair's bars without gaps, scattered to its places
(as execution_replay._grid completes L and V).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

STEP = timedelta(minutes=15)
BAR_MS = 15 * 60 * 1000
CACHE_DIR = TC.RUNTIME / "history" / "mm"
FIELDS = ("C", "O", "H", "L", "V", "ATR")
DTYPE = {"C": np.float64, "O": np.float32, "H": np.float32, "L": np.float32, "V": np.float32, "ATR": np.float32}
NAN = float("nan")


def bot_atr():
    TC.bot_readonly()
    import config as BOT
    from indicators import _atr
    period = int(getattr(BOT, "ATR_PERIOD", 14))
    return lambda h, l, c: _atr(h, l, c, period)


def build_cache(sym, arr, out_dir=CACHE_DIR, atr_fn=None) -> dict:
    """Dense per-pair arrays from the first to the last stored 15m bar; gaps = NaN; ATR over bars without gaps."""
    t = np.asarray(arr["t"], dtype=np.int64)
    if not len(t):
        return {"sym": sym, "bars": 0}
    atr_fn = atr_fn or bot_atr()
    n = int((t[-1] - t[0]) // BAR_MS) + 1
    pos = ((t - t[0]) // BAR_MS).astype(np.int64)
    src = {"C": arr["c"], "O": arr["o"], "H": arr["h"], "L": arr["l"], "V": arr["v"],
           "ATR": np.asarray(atr_fn(np.asarray(arr["h"], float), np.asarray(arr["l"], float), np.asarray(arr["c"], float)), float)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        d = np.full(n, np.nan, dtype=DTYPE[f])
        d[pos] = np.asarray(src[f], dtype=DTYPE[f])
        np.save(out_dir / f"{sym}.{f}.npy", d)
    meta = {"sym": sym, "first_open_ms": int(t[0]), "n": n, "stored_bars": int(len(t)), "gaps": int(n - len(t))}
    (out_dir / f"{sym}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


class _Field:
    def __init__(self, grid, name):
        self.g, self.name = grid, name

    def __getitem__(self, key):
        ix, j = key
        a = self.g._arrays[j][self.name]
        k = int(ix) - self.g._first[j]
        if 0 <= k < len(a):
            v = float(a[k])
            return v
        return NAN


class LazyGrid:
    def __init__(self, syms, t0, N, arrays, first_ix):
        self.syms = list(syms)
        self.col = {s: j for j, s in enumerate(self.syms)}
        self.t0 = t0
        self.N = int(N)
        self._arrays = [arrays[s] for s in self.syms]
        self._first = [int(first_ix[s]) for s in self.syms]
        for f in FIELDS:
            setattr(self, f, _Field(self, f))

    def idx(self, dt):
        return int((dt - self.t0) / STEP)

    def close(self):
        """Drop the memmaps (Windows keeps the files locked while they are open)."""
        import gc
        self._arrays = [{} for _ in self.syms]
        gc.collect()

    @classmethod
    def from_arrays(cls, arrays, first_ix, t0, N):
        return cls(list(arrays), t0, N, arrays, first_ix)

    @classmethod
    def load(cls, cache_dir=CACHE_DIR, syms=None, t0=None, N=None):
        cache_dir = Path(cache_dir)
        if syms is None:
            syms = sorted(p.name[:-len(".meta.json")] for p in cache_dir.glob("*.meta.json"))
        metas = {s: json.loads((cache_dir / f"{s}.meta.json").read_text(encoding="utf-8")) for s in syms}
        if t0 is None:
            t0 = datetime.fromtimestamp(min(m["first_open_ms"] for m in metas.values()) / 1000, timezone.utc)
        t0_ms = int(t0.timestamp() * 1000)
        first = {s: int((m["first_open_ms"] - t0_ms) // BAR_MS) for s, m in metas.items()}
        arrays = {s: {f: np.load(cache_dir / f"{s}.{f}.npy", mmap_mode="r") for f in FIELDS} for s in syms}
        if N is None:
            N = max(first[s] + metas[s]["n"] for s in syms) if syms else 0
        return cls(syms, t0, N, arrays, first)


def build_all(out_dir=CACHE_DIR) -> dict:
    import history_store as HS
    atr = bot_atr()
    done = 0
    for p in sorted(HS.HIST_DIR.glob("*_15m.npz")):
        sym = p.name[:-len("_15m.npz")]
        a = HS.load(p)
        if a is not None and len(a["t"]):
            build_cache(sym, a, out_dir, atr)
            done += 1
    return {"pairs": done, "dir": str(out_dir)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(build_all(), indent=1))
    _ = math
