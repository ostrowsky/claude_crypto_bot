"""LQ -- rebound after a long-liquidation flush (liquidation-flush-lq-1030-spec.md §3-§4).

  oi_change_1h    log OI change over 1 h from the futures metrics archive (row T known at T + 5 min, 30-min staleness;
                  the same rule as experiment_dpos.pos_features_many)
  price_change_1h log(close[i] / close[i - 4]) on spot 15m closes
  events          PO: both drops; P: the price drop with OI change > -1%; the first event per coin per 24 h
  trade           spot buy at the close of the next bar, exit hold_bars later; delisted inside the hold -> haircut
  random_bars     bars of the same coin and the same months as the events, same count, reproducible by seed
  trade_mean_ci   spec §5.1: the mean TRADE with a block bootstrap over days (blocks of daily_revenue.BLOCK days)
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BAR = 15 * 60 * 1000
H1 = 3600 * 1000
DAY = 24 * H1
P_ONLY_OI_MIN = -0.01
DELIST_R = -0.5


def oi_change_1h(m, ts) -> np.ndarray:
    import experiment_dpos as E
    ts = np.asarray(ts, dtype=np.int64)
    return E.pos_features_many(m, ts, H1, np.full(len(ts), np.nan))["oi_c1"]


def price_change_1h(c) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    out = np.full(len(c), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[4:] = np.log(c[4:] / c[:-4])
    out[~np.isfinite(out)] = np.nan
    return out


def events(t15, dp, doi, thr_oi: float, thr_p: float, kind: str) -> list:
    t15 = np.asarray(t15, dtype=np.int64)
    dp, doi = np.asarray(dp, float), np.asarray(doi, float)
    with np.errstate(invalid="ignore"):
        hit = dp <= thr_p
        hit &= (doi <= thr_oi) if kind == "PO" else (doi > P_ONLY_OI_MIN)
    out, last = [], None
    for i in np.flatnonzero(hit):
        if last is None or int(t15[i]) - int(t15[last]) >= DAY:
            out.append(int(i))
            last = i
    return out


def trade(t15, c15, i: int, hold_bars: int, fm, delisted: bool):
    t15 = np.asarray(t15, dtype=np.int64)
    c15 = np.asarray(c15, dtype=float)
    e, x = i + 1, i + 1 + int(hold_bars)
    if e >= len(c15) or not np.isfinite(c15[e]) or c15[e] <= 0:
        return None
    entry_ms = int(t15[e]) + BAR
    if x >= len(c15) or int(t15[x]) != int(t15[e]) + hold_bars * BAR or not np.isfinite(c15[x]):
        if not delisted:
            return None
        g = 1.0 + DELIST_R
        return {"ret_pct": DELIST_R * 100, "gross_mult": g, "entry_close_ms": entry_ms,
                "exit_close_ms": entry_ms + hold_bars * BAR, "marks": {}, "delisted": True}
    buy = fm.buy(float(c15[e])) / (1 - fm.taker_fee)
    sell = fm.sell(float(c15[x])) * (1 - fm.taker_fee)
    g = sell / buy
    marks = {}
    d = (entry_ms // DAY + 1) * DAY
    exit_ms = int(t15[x]) + BAR
    while d < exit_ms:
        k = int(np.searchsorted(t15, d - BAR, side="right")) - 1
        marks[d] = float(c15[k]) / buy
        d += DAY
    return {"ret_pct": (g - 1) * 100, "gross_mult": g, "entry_close_ms": entry_ms, "exit_close_ms": exit_ms,
            "marks": marks, "delisted": False}


def _month(t_ms) -> str:
    return datetime.fromtimestamp(int(t_ms) / 1000, timezone.utc).strftime("%Y-%m")


def random_bars(t15, events_idx, n: int, seed: int) -> list:
    """For each event, one random bar of the same month (the multiset of months is kept)."""
    t15 = np.asarray(t15, dtype=np.int64)
    rnd = random.Random(seed)
    months = np.array([_month(t) for t in t15[:: 96]])                # month of each day start (cheap index)
    out = []
    for i in list(events_idx)[:n]:
        m = _month(t15[i])
        days = np.flatnonzero(months == m)
        if not len(days):
            continue
        d = int(days[rnd.randrange(len(days))]) * 96
        out.append(min(len(t15) - 1, d + rnd.randrange(96)))
    return out


def trade_mean_ci(trades, alpha: float = 0.05, seed: int = 29, boot: int = 2000) -> dict:
    """Mean of all trades; CI from resampling blocks of consecutive trade days (a day keeps all its trades, so clustered
    crash days stay together). Also: share of the summed return made by the 5 best days."""
    import daily_revenue as DR
    by = {}
    for x in trades:
        by.setdefault(int(x["entry_close_ms"]) // DAY, []).append(float(x["ret_pct"]))
    days = sorted(by)
    n_tr = sum(len(v) for v in by.values())
    if not n_tr:
        return {"n_trades": 0, "n_days": 0}
    s = np.array([sum(by[d]) for d in days])
    c = np.array([len(by[d]) for d in days], dtype=float)
    n, b = len(days), max(1, min(DR.BLOCK, len(days)))
    rnd = random.Random(seed)
    means = []
    for _ in range(boot):
        ss = cc = 0.0
        k = 0
        while k < n:
            i = rnd.randrange(n - b + 1)
            j = min(i + b, i + n - k)
            ss += float(s[i:j].sum())
            cc += float(c[i:j].sum())
            k += j - i
        means.append(ss / cc)
    means.sort()
    top5 = float(np.sort(s)[::-1][:5].sum())
    return {"n_trades": n_tr, "n_days": n, "mean_pct": round(float(s.sum() / c.sum()), 4),
            "lo95": round(means[int(alpha / 2 * boot)], 4), "hi95": round(means[min(boot - 1, int((1 - alpha / 2) * boot))], 4),
            "top5_days_share_of_sum": round(top5 / float(s.sum()), 3) if s.sum() else None}
