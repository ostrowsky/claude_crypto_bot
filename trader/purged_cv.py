"""Purged walk-forward split by time (trader-learning-loop-f-1020-spec.md §2, §6.3).

Rows carry t0 (decision time) and t1 (time the outcome is known). Days of t0 are cut into n_folds + 1
contiguous blocks; fold i tests block i + 1 and trains only on rows whose outcome was known more than
embargo_days before the test block starts -- so no training label overlaps the test period.
"""
from __future__ import annotations

from datetime import timedelta


def walk_forward(rows, n_folds: int = 4, embargo_days: float = 1.0, t0="t0", t1="t1") -> list:
    days = sorted({r[t0].date() for r in rows})
    if len(days) < n_folds + 1:
        raise ValueError("fewer days than folds + 1")
    size = len(days) / (n_folds + 1)
    blocks = [days[int(round(i * size)): int(round((i + 1) * size))] for i in range(n_folds + 1)]
    folds = []
    for b in blocks[1:]:
        bset = set(b)
        te = [i for i, r in enumerate(rows) if r[t0].date() in bset]
        if not te:
            continue
        start = min(rows[i][t0] for i in te)
        cut = start - timedelta(days=embargo_days)
        tr = [i for i, r in enumerate(rows) if r[t1] < cut]
        folds.append((tr, te))
    return folds
