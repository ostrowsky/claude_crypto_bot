"""FS -- BTC+ETH funding carry switched on by trailing funding (fc-switch-n4-1028-spec.md §3-§4).

  schedule         per pair and decision day: on / off by the annualised W-day funding before 00:00 (carry.annualized_funding)
                   with hysteresis theta_in / theta_out = theta_in / 2; no data -> off
  schedule_chooser carry.simulate chooser that holds exactly the pairs whose schedule is on that day
  subset_market    the carry Market restricted to a few pairs (BTC + ETH): the passive book is unchanged (test)
Days are 'YYYY-MM-DD' strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import carry as C                 # noqa: E402
import xsmom as XS                # noqa: E402

PAIRS = C.PASSIVE_PAIRS


def schedule(funding: dict, syms, days, W: int, theta_in: float) -> dict:
    theta_out = theta_in / 2
    out = {}
    for s in syms:
        f = (funding or {}).get(s)
        on, state = np.zeros(len(days), dtype=int), 0
        for i, d in enumerate(days):
            v = None if f is None else C.annualized_funding(f["t"], f["rate"], C.day_ms(d), W)
            if v is None:
                state = 0
            elif state:
                state = 1 if v >= theta_out else 0
            else:
                state = 1 if v >= theta_in else 0
            on[i] = state
        out[s] = on
    return out


def n_switches(sched: dict) -> int:
    """On/off changes from a flat start (an 'on' first day counts as an entry)."""
    total = 0
    for v in sched.values():
        x = np.r_[0, np.asarray(v)]
        total += int(np.sum(x[1:] != x[:-1]))
    return total


def schedule_chooser(sched: dict, days):
    idx = {d: i for i, d in enumerate(days)}
    syms = sorted(sched)

    def chooser(ctx) -> list:
        i = idx.get(ctx.day)
        if i is None:
            return []
        return [s for s in syms if np.asarray(sched[s])[i] > 0.5]
    return chooser


def _sub_panel(p, syms, cls=None):
    cols = [p.col[s] for s in syms if s in p.col]
    names = [s for s in syms if s in p.col]
    take = lambda a: a[:, cols]                     # noqa: E731
    if cls is C.PerpPanel:
        return C.PerpPanel(p.days, names, take(p.close), take(p.first_close), take(p.qv), take(p.complete),
                           take(p.has_first), take(p.has_bar), take(p.high))
    return XS.Panel(p.days, names, take(p.close), take(p.first_close), take(p.qv), take(p.complete),
                    take(p.has_first), take(p.has_bar))


def subset_market(m, syms=PAIRS):
    syms = sorted(syms)
    return C.Market(spot=_sub_panel(m.spot, syms), perp=_sub_panel(m.perp, syms, C.PerpPanel),
                    funding={s: m.funding[s] for s in syms if s in m.funding})


def days_between(day_from: str, day_to: str) -> list:
    return [XS.day_str(n) for n in range(XS.day_num(day_from), XS.day_num(day_to))]
