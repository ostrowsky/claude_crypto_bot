"""SN -- short a new Binance USD-M perp after its launch (short-new-perps-n6-1029-spec.md §3-§4).

  launch    the perp's first 15m bar after the cut-off (trader/runtime/perp_history)
  trade     short at the close of bar d after the first one, hold H days (H x 96 bars); L = 1 isolated: q = M / F_entry;
            value of 1 margin at exit = 1 + q (F_entry - F_exit) - fees + funding received; liquidation when a bar high
            reaches F_entry x (1 + 1/L - MMR) -> margin lost; funding events in (entry close, exit close], priced at the
            last closed bar; marks at 00:00 UTC in the experiment_la.book format (vectorised for the controls)
  control   the same entry time and hold on a random OLD perp (listed >= 180 days before, no new launch +-30 days)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiment_la import book    # noqa: E402,F401  (the LA book, reused unchanged)

BAR = 15 * 60 * 1000
DAY = 86_400_000
LEV = 1.0
MMR = 0.01
OLD_DAYS = 180
NEW_GAP_DAYS = 30


def launches(first_bar: dict, cutoff_ms: int) -> dict:
    return {s: int(t) for s, t in first_bar.items() if int(t) > cutoff_ms}


def entry_ms_for(arr, launch_ms: int, d: int):
    t = np.asarray(arr["t"], dtype=np.int64)
    i = int(np.searchsorted(t, launch_ms)) + int(d)
    return int(t[i]) if i < len(t) else None


def in_train(x: dict, test_from_ms: int) -> bool:
    return x["exit_close_ms"] < test_from_ms


def trade(arr, funding, entry_ms: int, H_days: int, fees):
    t = np.asarray(arr["t"], dtype=np.int64)
    i = int(np.searchsorted(t, entry_ms))
    if i >= len(t) or int(t[i]) != entry_ms:
        return None
    n = int(H_days) * 96
    j = i + n
    if j >= len(t) or int(t[j]) != int(t[i]) + n * BAR:
        return None
    c = np.asarray(arr["c"], dtype=float)
    h = np.asarray(arr["h"], dtype=float)
    if not (np.isfinite(c[i]) and c[i] > 0):
        return None
    fe = float(c[i]) * (1 - fees.slip)
    q = 1.0 / fe
    fee_in = q * fe * fees.perp_taker
    liq = fe * (1 + 1.0 / LEV - MMR)
    hit = np.flatnonzero(h[i + 1:j + 1] >= liq)
    k_end, reason = (i + 1 + int(hit[0]), "liquidation") if len(hit) else (j, "time")
    entry_close, exit_close = int(t[i]) + BAR, int(t[k_end]) + BAR
    # funding received by the short, events in (entry close, exit close]
    ft = fr = None
    if funding is not None and len(funding["t"]):
        ft = np.asarray(funding["t"], dtype=np.int64)
        fr = np.asarray(funding["rate"], dtype=float)
        a, b = int(np.searchsorted(ft, entry_close, "right")), int(np.searchsorted(ft, exit_close, "right"))
        ft, fr = ft[a:b], fr[a:b]
        kb = np.searchsorted(t, ft - BAR, "right") - 1                # last bar closed by the event
        amt = q * c[np.clip(kb, 0, len(c) - 1)] * fr
        cum = np.cumsum(amt)
    fund_total = float(cum[-1]) if ft is not None and len(ft) else 0.0
    if reason == "liquidation":
        gross = fund_total - fee_in
    else:
        fill = float(c[j]) * (1 + fees.slip)
        gross = 1.0 + q * (fe - fill) - fee_in - q * fill * fees.perp_taker + fund_total
    marks = {}
    d = (entry_close // DAY + 1) * DAY
    while d < exit_close:
        k = int(np.searchsorted(t, d - BAR, "right")) - 1
        f_sofar = float(cum[int(np.searchsorted(ft, d, "right")) - 1]) if ft is not None and len(ft) and ft[0] <= d else 0.0
        marks[d] = 1.0 + q * (fe - float(c[k])) - fee_in + f_sofar
        d += DAY
    return {"entry_ms": int(t[i]), "entry_close_ms": entry_close, "exit_close_ms": exit_close, "gross_mult": gross,
            "ret_pct": (gross - 1) * 100, "marks": marks, "reason": reason, "funding": fund_total}


def old_pool(first_bar: dict, launches_: dict, t_ms: int) -> list:
    return sorted(s for s, fb in first_bar.items()
                  if int(fb) <= t_ms - OLD_DAYS * DAY
                  and not (s in launches_ and abs(launches_[s] - t_ms) < NEW_GAP_DAYS * DAY))


def control(entries, first_bar: dict, launches_: dict, seed: int) -> list:
    rnd = random.Random(seed)
    out = []
    for e in entries:
        pool = [s for s in old_pool(first_bar, launches_, e["entry_ms"]) if s != e["sym"]]
        if pool:
            out.append({**e, "sym": rnd.choice(pool)})
    return out
