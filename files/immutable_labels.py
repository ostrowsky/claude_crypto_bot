"""Winner labels from the immutable store — the TH-03 replacement.

`label_top20` calls a coin a winner using the same rolling-24h snapshot that
produced its features, so a coin already up 14% is a winner *because* it is up
14%. This module answers the same question from exchange klines at the day's
close: rank the day's symbols by `eod_return_pct` and take the top N.

The distinction that matters: `eod_return_pct` is close ÷ open over a finished
UTC day. It cannot be known before that day ends, which is what makes it usable
as ground truth for features computed during the day.

Stdlib plus the label store. No bot dataset is read here, and a test asserts it.

Spec: docs/specs/features/north-star-immutable-labels-spec.md
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from label_store import LabelStore, WELL_COVERED_FRACTION  # noqa: E402

DEFAULT_TOP_N = 20


def _by_day(records: Iterable[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("complete"):
            out[rec["utc_day"]].append(rec)
    return out


def winners_by_day(*, top_n: int = DEFAULT_TOP_N, watchlist: set[str] | None = None,
                   store: LabelStore | None = None,
                   min_universe: int | None = None) -> tuple[set, dict]:
    """`(winners, eod_return)` where a winner is a day's top-N by EOD return.

    A day whose universe is too thin yields **no** winners rather than a short
    list: ranking six symbols and calling the top three "the day's top-20" would
    manufacture winners out of missing data (TH-05).
    """
    store = store or LabelStore()
    records = store.records()
    if watchlist is not None:
        records = [r for r in records if r["symbol"] in watchlist]

    per_day = _by_day(records)
    if min_universe is None:
        widest = max((len(v) for v in per_day.values()), default=0)
        min_universe = max(top_n, int(widest * WELL_COVERED_FRACTION))

    winners: set[tuple[str, str]] = set()
    eod: dict[tuple[str, str], float] = {}
    for day, rows in per_day.items():
        for rec in rows:
            eod[(day, rec["symbol"])] = rec["eod_return_pct"]
        if len(rows) < min_universe:
            continue
        ranked = sorted(rows, key=lambda r: -float(r["eod_return_pct"]))
        for rec in ranked[:top_n]:
            winners.add((day, rec["symbol"]))
    return winners, eod


def label_for(symbol: str, utc_day: str, *, top_n: int = DEFAULT_TOP_N,
              cache: dict | None = None,
              store: LabelStore | None = None) -> int | None:
    """1 / 0 for a single (symbol, day), or **None** when the store does not
    know it. None is not zero: an unlabelled row must be dropped, not counted
    as a non-winner, or absence of data becomes evidence of failure."""
    if cache is None or "winners" not in cache:
        winners, _ = winners_by_day(top_n=top_n, store=store)
        known = {(d, s) for d, s in winners}
        all_keys = {(r["utc_day"], r["symbol"])
                    for r in (store or LabelStore()).records() if r.get("complete")}
        if cache is not None:
            cache["winners"], cache["known"] = known, all_keys
    else:
        known, all_keys = cache["winners"], cache["known"]
    key = (utc_day, symbol)
    if key not in all_keys:
        return None
    return 1 if key in known else 0


def summary(*, top_n: int = DEFAULT_TOP_N,
            watchlist: set[str] | None = None) -> dict[str, Any]:
    winners, eod = winners_by_day(top_n=top_n, watchlist=watchlist)
    days = sorted({d for d, _ in winners})
    return {
        "label_provenance": "immutable_later_eod_klines",
        "top_n": top_n,
        "winners": len(winners),
        "days": len(days),
        "window": f"{days[0]}..{days[-1]}" if days else "",
        "winners_per_day": round(len(winners) / len(days), 2) if days else 0.0,
        "labelled_pairs": len(eod),
    }
