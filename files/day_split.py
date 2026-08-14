"""Split rows by complete UTC days, never by row index.

`train_top_gainer.py` splits sorted rows by index, which is the TH-04 finding:
rows from one day share market beta, and with T+N labels they share outcomes, so
putting some of a day on each side leaks the answer across the boundary. A model
validated that way reports a number it did not earn.

The embargo is the second half. When a label matures N days after its row, the
last training days still "know" the first holdout days; withholding the boundary
is what makes the separation real rather than nominal.

Stdlib only. Deterministic. Used by the label store and, once its before/after
evidence exists, by the model trainer.

Spec: docs/specs/features/immutable-label-store-spec.md
"""
from __future__ import annotations

from typing import Any, Sequence


def day_keys(rows: Sequence[dict], day_key: str) -> list[str]:
    """Sorted unique day values. Rows without one are not silently dropped —
    they raise, because a row that cannot be placed in time cannot be split."""
    days = set()
    for row in rows:
        value = row.get(day_key)
        if not value:
            raise ValueError(f"row without {day_key!r} cannot be split by day")
        days.add(str(value))
    return sorted(days)


def split_by_day(rows: Sequence[dict], day_key: str, *, train_frac: float = 0.7,
                 embargo_days: int = 0) -> tuple[list[dict], list[dict]]:
    """Chronological split on whole days.

    Returns `(train, holdout)`. No day appears on both sides. With
    `embargo_days > 0` the days immediately before the holdout are withheld
    from training entirely — they are neither trained on nor evaluated.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1")
    days = day_keys(rows, day_key)
    if len(days) < 2:
        raise ValueError("need at least two distinct days to split")

    cut = int(len(days) * train_frac)
    cut = max(1, min(cut, len(days) - 1))       # both sides non-empty
    train_days = set(days[:cut])
    holdout_days = set(days[cut:])

    if embargo_days > 0:
        # Withhold the last `embargo_days` training days. They are dropped, not
        # moved: moving them into holdout would re-create the leak from the
        # other direction.
        embargoed = set(days[max(0, cut - embargo_days):cut])
        train_days -= embargoed
        if not train_days:
            raise ValueError("embargo consumed the entire training window")

    train = [r for r in rows if str(r[day_key]) in train_days]
    holdout = [r for r in rows if str(r[day_key]) in holdout_days]
    return train, holdout


DAY_MS = 86_400_000


def split_indices_by_day(timestamps_ms, *, train_frac: float = 0.8,
                         embargo_days: int = 0, by: str = "rows"):
    """Index arrays for a day-grouped split of chronological ms timestamps.

    The array form exists so the model trainer can keep its numpy layout while
    using this one splitter — a second implementation of "split by day" would
    eventually disagree with the first, and only one of them would be tested.

    Returns `(train_idx, holdout_idx)` as lists. Embargoed rows appear in
    neither: moving them into holdout would re-create the leak from the other
    side.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1")
    day_of = [int(int(ts) // DAY_MS) for ts in timestamps_ms]
    days = sorted(set(day_of))
    if len(days) < 2:
        raise ValueError("need at least two distinct UTC days to split")

    if by == "rows":
        # Row density is wildly uneven here: early days carry a handful of rows,
        # recent ones carry ~105 symbols x 4 snapshots. Splitting 80/20 by DAY
        # COUNT produced a 49/51 split by rows, so a comparison against the old
        # row-index cut was measuring training-set size, not the split.
        counts: dict[int, int] = {}
        for d in day_of:
            counts[d] = counts.get(d, 0) + 1
        target = len(day_of) * train_frac
        running = 0
        cut = 1
        for i, day in enumerate(days):
            running += counts[day]
            if running >= target:
                cut = max(1, min(i + 1, len(days) - 1))
                break
        else:
            cut = len(days) - 1
    else:
        cut = int(len(days) * train_frac)
        cut = max(1, min(cut, len(days) - 1))
    train_days = set(days[:cut])
    holdout_days = set(days[cut:])
    if embargo_days > 0:
        train_days -= set(days[max(0, cut - embargo_days):cut])
        if not train_days:
            raise ValueError("embargo consumed the entire training window")

    train_idx = [i for i, d in enumerate(day_of) if d in train_days]
    holdout_idx = [i for i, d in enumerate(day_of) if d in holdout_days]
    return train_idx, holdout_idx


def rolling_origin(rows: Sequence[dict], day_key: str, *, folds: int = 4,
                   embargo_days: int = 0) -> list[tuple[list[dict], list[dict]]]:
    """Walk-forward folds, each respecting whole days and the embargo.

    Preferred over a single split when events are scarce: it uses every day for
    evaluation exactly once without permanently sealing away the newest data,
    which at this project's event rate would cost the most relevant evidence.
    """
    days = day_keys(rows, day_key)
    if len(days) < folds + 1:
        raise ValueError("not enough days for the requested folds")
    out: list[tuple[list[dict], list[dict]]] = []
    step = len(days) // (folds + 1)
    for fold in range(1, folds + 1):
        cut = step * fold
        frac = cut / len(days)
        try:
            out.append(split_by_day(rows, day_key, train_frac=frac,
                                    embargo_days=embargo_days))
        except ValueError:
            continue
    return out
