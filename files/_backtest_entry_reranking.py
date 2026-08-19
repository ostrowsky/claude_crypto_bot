"""Can the bot's own entries be ordered better, at a fixed alert budget?

The entry path is where the evidence points: portfolio alpha is negative on
every window, the epoch take baseline is about -0.8% forward return, and half
the EX1 misses are trades sitting within two hours of a detected uptrend.

This asks the narrow, non-gate-changing question first (TH-06): holding the
number of alerts fixed, does any signal available AT ENTRY TIME order them
better than the order the bot used? A reordering costs nothing and changes no
gate, so it is the cheapest possible lever — and if nothing reorders, that is a
negative result worth committing rather than a tuning exercise.

Ground truth is the immutable label store: did the entry land on a coin that
finished the UTC day in the global top-20 intersected with the watchlist. Split
by TIME, never at random. Every ratio is reported against a 200-draw random
control, because at a few hundred rows a 2pp difference is noise.

  pyembed\python.exe files\_backtest_entry_reranking.py

Spec: docs/specs/features/entry-reranking-spec.md
"""
from __future__ import annotations

import io
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import immutable_labels as IL  # noqa: E402

# Everything here is logged by botlog at the moment of entry, so nothing is
# available to the ranking that was not available to the decision.
SIGNALS = ["candidate_score", "ml_proba", "ranker_ev", "ranker_final_score",
           "ranker_quality_proba", "ranker_top_gainer_prob", "adx", "rsi",
           "slope_pct", "vol_x", "daily_range", "macd_hist", "decoupling_score"]
TRAIN_FRAC = 0.7
DRAWS = 200
BUDGETS = (1.0, 0.75, 0.5, 0.33)


def load_rows(winners: set, wdays: set) -> list:
    rows = []
    with io.open(HERE / "bot_events.jsonl", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"entry"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "entry":
                continue
            day = str(e.get("ts", ""))[:10]
            # Only days the label store covers: a day without ground truth is
            # not a day the bot got wrong (TH-05).
            if day not in wdays:
                continue
            vals = {}
            for f in SIGNALS:
                v = e.get(f)
                try:
                    vals[f] = float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    vals[f] = 0.0
            rows.append({"day": day, "sym": e.get("sym"), "x": vals,
                         "y": 1 if (day, e.get("sym")) in winners else 0})
    rows.sort(key=lambda r: r["day"])
    return rows


def precision_at_budget(rows: list, score_of, frac: float) -> tuple[float, int]:
    """Keep the top `frac` of each day's alerts. The budget is per DAY, so a
    quiet day is not scored against a busy one."""
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append((score_of(r), r["y"]))
    kept = []
    for lst in by_day.values():
        k = max(1, int(round(len(lst) * frac)))
        kept += [y for _, y in sorted(lst, key=lambda t: -t[0])[:k]]
    return (100.0 * sum(kept) / len(kept) if kept else 0.0), sum(kept)


def random_band(rows: list, frac: float) -> tuple[float, float, float]:
    draws = []
    for seed in range(DRAWS):
        rng = random.Random(seed)
        draws.append(precision_at_budget(rows, lambda _r, g=rng: g.random(), frac)[0])
    draws.sort()
    return st.mean(draws), draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]


def main() -> int:
    watchlist = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
    winners, _ = IL.winners_by_day(top_n=20, watchlist=watchlist,
                                   rank_before_filter=True)
    wdays = {d for d, _ in winners}
    rows = load_rows(winners, wdays)
    if not rows:
        print("no entries on labelled days")
        return 1

    days = sorted({r["day"] for r in rows})
    cut = days[int(len(days) * TRAIN_FRAC)]
    test = [r for r in rows if r["day"] >= cut]
    base_rate = len(winners) / (len(wdays) * len(watchlist))

    print("=" * 78)
    print("Entry reranking at a fixed alert budget · immutable labels, time split")
    print("=" * 78)
    print(f"entries on labelled days  {len(rows)} over {len(days)} days "
          f"({len(rows)/len(days):.1f}/day)")
    print(f"holdout                   {len(test)} rows over "
          f"{len({r['day'] for r in test})} days, from {cut}")
    prec_all = 100.0 * sum(r["y"] for r in test) / len(test)
    print(f"the bot's own precision   {prec_all:.2f}%   base rate {100*base_rate:.2f}%   "
          f"lift {prec_all/100/base_rate:.2f}x")
    print()

    for frac in BUDGETS:
        mean, lo, hi = random_band(test, frac)
        print(f"--- budget {int(frac*100)}% of the day's alerts   "
              f"random control {mean:.2f}% [{lo:.2f}, {hi:.2f}] ---")
        scored = []
        for f in SIGNALS:
            p, w = precision_at_budget(test, lambda r, _f=f: r["x"][_f], frac)
            scored.append((p, w, f))
        for p, w, f in sorted(scored, reverse=True):
            verdict = ("ABOVE" if p > hi else "BELOW" if p < lo else "noise")
            print(f"   {f:<26}{p:6.2f}%  winners {w:<4} {verdict}")
        print()

    print("Reading it: a signal inside the random band orders nothing. BELOW the")
    print("band is not neutral — it orders the bot's own entries BACKWARDS.")
    print()
    print("Selection caveat (TH-06): these are entries that already PASSED every")
    print("gate, including the ranker's own hard veto. A signal that looks")
    print("anti-informative here is anti-informative AMONG THE SURVIVORS; this")
    print("says nothing about whether the veto itself should be relaxed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
