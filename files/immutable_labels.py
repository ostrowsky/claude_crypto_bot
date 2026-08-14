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


def tier_labels(days: list, symbols: list, *, tiers=(5, 10, 20, 50),
                floor: float = 5.0, store: LabelStore | None = None,
                min_universe: int | None = None) -> tuple[list, dict, dict]:
    """Immutable tier labels for training rows.

    Returns `(keep, labels, stats)` where `keep` is the indices of rows the
    store can label, and `labels["topN"]` is aligned to `keep` (not to the input
    rows). A row the store does not know is **dropped**: labelling it 0 would
    teach the model that every symbol outside the store failed to move.

    A positive is rank ≤ N by EOD return within the day's store universe **AND**
    a return of at least `floor`. The floor is load-bearing: a pure rank mints
    exactly N winners a day whatever the market does, which fixes the base rate
    by construction and carries no information about whether the day was worth
    trading. The entry bandit already paid for that — rank-only scored lift
    1.02×, and only the return floor took it to 4.07×.
    """
    store = store or LabelStore()
    records = [r for r in store.records() if r.get("complete")]
    per_day = _by_day(records)
    known = {(r["utc_day"], r["symbol"]) for r in records}

    if min_universe is None:
        widest = max((len(v) for v in per_day.values()), default=0)
        min_universe = int(widest * WELL_COVERED_FRACTION)

    # (day -> set of winning symbols) per tier, computed once per day.
    winners: dict[int, set] = {n: set() for n in tiers}
    for day, rows in per_day.items():
        if len(rows) < min_universe:
            continue                       # thin day: no positives, see TH-05
        ranked = sorted(rows, key=lambda r: -float(r["eod_return_pct"]))
        for n in tiers:
            for rec in ranked[:n]:
                if float(rec["eod_return_pct"]) >= floor:
                    winners[n].add((day, rec["symbol"]))

    keep: list[int] = []
    labels: dict[str, list] = {f"top{n}": [] for n in tiers}
    for i, (day, sym) in enumerate(zip(days, symbols)):
        if (day, sym) not in known:
            continue
        keep.append(i)
        for n in tiers:
            labels[f"top{n}"].append(1.0 if (day, sym) in winners[n] else 0.0)

    n_lab = len(keep)
    stats = {
        "label_provenance": "immutable_later_eod_klines",
        "label_timing": "immutable_later_eod_close",
        "floor_pct": floor,
        "n_rows_in": len(days),
        "n_labelled": n_lab,
        "dropped_unlabelled": len(days) - n_lab,
        # TH-01: a positives count is unreadable without the base rate it sits on.
        "base_rate": {k: (sum(v) / n_lab if n_lab else 0.0)
                      for k, v in labels.items()},
        "store_universe_note": "watchlist-scoped: tiers are top-N WITHIN the "
                               "watchlist, not top-N on the exchange",
    }
    return keep, labels, stats


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
