"""Can the day's winners be named at 00 UTC, before the move?

Serves goal 1 (spot the winners early) and goal 2 (signal entry early). The
seven live entry modes all require ADX / slope / volume / breakout confirmation,
so by construction none can fire ahead of the move — measured, the bot's
move-relative lead is 0.02 and its early alerts carry lift 0.72x, worse than
chance. Any progress on goals 1-2 therefore needs a path that does not wait for
confirmation.

This asks whether such a path could exist at all, before anything is built: rank
the watchlist from the 00 UTC snapshot — nothing of the day has elapsed — and
see how many of that day's actual winners land in the top K.

Ground truth is the immutable label store (global top-20 INTERSECT watchlist by
close/open). Split by TIME. The comparison that matters is at a MATCHED alert
budget: the live bot spends ~31.5 alerts a day, so a top-5 list winning on
coverage would be earlier AND cheaper.

  pyembed\python.exe files\_backtest_early_ranking.py

Spec: docs/specs/features/early-ranking-shadow-spec.md
"""
from __future__ import annotations

import io
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

import immutable_labels as IL  # noqa: E402
from top_gainer_model import FEATURE_NAMES  # noqa: E402

SNAPSHOT_HOUR = 0          # nothing of the UTC day has elapsed yet
TRAIN_FRAC = 0.7
TOP_K = (3, 5, 10, 20)
DRAWS = 200


def load_snapshots(watchlist: set) -> dict:
    """(day -> [(symbol, features)]) from the earliest snapshot of each day."""
    by_day: dict[str, list] = defaultdict(list)
    with io.open(HERE / "top_gainer_dataset.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = e.get("symbol")
            if sym not in watchlist:
                continue
            ts = e.get("ts") or 0
            dt = datetime.fromtimestamp(ts / 1000 if ts > 1e11 else ts, timezone.utc)
            if dt.hour != SNAPSHOT_HOUR:
                continue
            feat = e.get("features") or {}
            x = [float(feat.get(f, 0.0) or 0.0) for f in FEATURE_NAMES]
            by_day[dt.strftime("%Y-%m-%d")].append((sym, x))
    return by_day


def coverage_at_k(ranked_days: dict, winners: set, k: int) -> tuple[int, int]:
    """(winners caught, winners available) with a k-name list per day."""
    caught = total = 0
    for day, ranked in ranked_days.items():
        day_winners = {s for d, s in winners if d == day}
        if not day_winners:
            continue
        total += len(day_winners)
        caught += len(day_winners & {s for s, _ in ranked[:k]})
    return caught, total


def main() -> int:
    watchlist = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
    winners, _ = IL.winners_by_day(top_n=20, watchlist=watchlist,
                                   rank_before_filter=True)
    label_days = {d for d, _ in winners}
    snaps = {d: v for d, v in load_snapshots(watchlist).items() if d in label_days}
    days = sorted(snaps)
    if len(days) < 20:
        print(f"only {len(days)} days with a {SNAPSHOT_HOUR:02d} UTC snapshot "
              f"and a label — too few to split")
        return 1

    cut = days[int(len(days) * TRAIN_FRAC)]
    train_days = [d for d in days if d < cut]
    test_days = [d for d in days if d >= cut]

    X, y = [], []
    for d in train_days:
        for sym, x in snaps[d]:
            X.append(x)
            y.append(1 if (d, sym) in winners else 0)
    X, y = np.array(X), np.array(y)

    print("=" * 78)
    print(f"Naming the day's winners from the {SNAPSHOT_HOUR:02d} UTC snapshot")
    print("=" * 78)
    print(f"days with a {SNAPSHOT_HOUR:02d} UTC snapshot and a label   {len(days)}")
    print(f"train {len(train_days)} days / {len(X)} rows   "
          f"test {len(test_days)} days   (split at {cut})")
    print(f"train base rate {100*y.mean():.2f}%")

    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                               verbose=0, random_seed=42,
                               auto_class_weights="Balanced")
    model.fit(X, y)

    ranked_days = {}
    for d in test_days:
        rows = snaps[d]
        if not rows:
            continue
        p = model.predict_proba(np.array([x for _, x in rows]))[:, 1]
        ranked_days[d] = sorted(zip([s for s, _ in rows], p),
                                key=lambda t: -t[1])

    total_w = sum(1 for d, _ in winners if d in ranked_days)
    universe = sum(len(v) for v in ranked_days.values())
    base = total_w / universe if universe else 0.0
    print(f"\ntest: {len(ranked_days)} days, {universe} coin-days, "
          f"{total_w} winners  (base rate {100*base:.2f}%)")
    print(f"\n  {'list size':<12}{'winners caught':>16}{'coverage':>11}"
          f"{'precision':>11}{'lift':>8}   random coverage")
    for k in TOP_K:
        caught, total = coverage_at_k(ranked_days, winners, k)
        picks = sum(min(k, len(v)) for v in ranked_days.values())
        # A random k-name list catches k/universe of the day's winners.
        rnd = []
        for seed in range(DRAWS):
            rng = random.Random(seed)
            shuffled = {d: rng.sample(v, len(v)) for d, v in ranked_days.items()}
            c, t = coverage_at_k(shuffled, winners, k)
            rnd.append(100.0 * c / t if t else 0.0)
        rnd.sort()
        lo, hi = rnd[int(0.025 * DRAWS)], rnd[int(0.975 * DRAWS) - 1]
        cov = 100.0 * caught / total if total else 0.0
        prec = 100.0 * caught / picks if picks else 0.0
        print(f"  top-{k:<8}{caught:>10}/{total:<5}{cov:>10.1f}%{prec:>10.1f}%"
              f"{prec/100/base if base else 0:>7.2f}x   "
              f"{sum(rnd)/len(rnd):.1f}% [{lo:.1f}, {hi:.1f}]")

    print()
    print("Reading it: coverage is the share of that day's winners named before")
    print("the day began. The live bot spends ~31.5 alerts a day for coverage")
    print("0.80 measured at ANY hour; a top-k list here spends k and names them")
    print("at 00 UTC, which is a move-relative lead of 1.0 by construction.")
    print()
    print("This is a ceiling, not a plan: it says whether the signal exists")
    print("early. It does not say the alert should fire at 00 UTC with no")
    print("confirmation — that is what the shadow path is for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
