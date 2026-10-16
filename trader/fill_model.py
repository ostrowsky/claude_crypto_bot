"""Fill model shared by the E1 replay and the phase-C paper executor.

execution-replay-e1-1010-spec.md §4. Prices are fractions (0.0005 = 5 bps).

  market buy   p * (1 + half_spread + impact(size))            taker fee
  market sell  p * (1 - half_spread - impact(size))            taker fee
  TP (limit)   fills at TP only if bar high >= TP * (1 + tick) maker fee (trade-through, no queue optimism)
  SL (stop)    triggers on bar low <= stop; fills at min(stop, bar open) * (1 - half_spread - impact)
  same bar     TP and SL both touched -> SL first (pessimistic)
  impact       k * sqrt(size_usdt / quote_volume_of_the_bar)   (square-root law)

Until A0 has 7 days of book data the replay reports every result at LEVELS 0 / 1 / 2;
"profitable" needs level 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FillModel:
    taker_fee: float = 0.001
    maker_fee: float = 0.001
    half_spread: float = 0.0
    impact_k: float = 0.0
    tp_tick: float = 0.0001

    def impact(self, size_usdt: float, bar_quote_vol: float | None) -> float:
        if self.impact_k <= 0 or size_usdt <= 0:
            return 0.0
        if not bar_quote_vol or not math.isfinite(bar_quote_vol) or bar_quote_vol <= 0:
            return self.impact_k            # no volume known: charge as if size == bar volume
        return self.impact_k * math.sqrt(size_usdt / bar_quote_vol)

    def buy(self, price: float, size_usdt: float = 0.0, bar_quote_vol: float | None = None) -> float:
        return price * (1 + self.half_spread + self.impact(size_usdt, bar_quote_vol))

    def sell(self, price: float, size_usdt: float = 0.0, bar_quote_vol: float | None = None) -> float:
        return price * (1 - self.half_spread - self.impact(size_usdt, bar_quote_vol))

    def tp_fill(self, tp: float, bar_high: float):
        return tp if bar_high >= tp * (1 + self.tp_tick) else None

    def sl_fill(self, stop: float, bar_open: float, bar_low: float, size_usdt: float = 0.0,
                bar_quote_vol: float | None = None):
        if bar_low > stop:
            return None
        return self.sell(min(stop, bar_open), size_usdt, bar_quote_vol)


FEES = {"taker": (0.001, 0.001), "bnb": (0.00075, 0.00075), "report": (0.0005, 0.0005)}
# (cost per side from mid, impact_k). PROVISIONAL 2026-10-10, from the first two A0 snapshots
# (demo book, 42 pairs): a 1000 USDT market order costs 6.6 bps (buy) / 10.0 bps (sell) from mid at
# the median, 20-23 bps at p90. Level 1 = 8 bps per side ~ median, level 2 = 20 bps ~ p90; no size
# term until 7 days of A0 fix it per pair. The first draft (5 bps + k=0.1 on 15m bar volume) charged
# ~1.4% per side -- about 100x the measured book -- and was replaced before any result was used.
SLIP = {0: (0.0, 0.0), 1: (0.0008, 0.0), 2: (0.0020, 0.0)}


def model(fees: str = "taker", slip: int = 1) -> FillModel:
    m, t = FEES[fees]
    hs, k = SLIP[int(slip)]
    return FillModel(taker_fee=t, maker_fee=m, half_spread=hs, impact_k=k)
