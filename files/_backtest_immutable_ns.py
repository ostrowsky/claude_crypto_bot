"""North Star on leaky vs immutable labels, over the window both cover.

The old winner set is "already moved by the snapshot"; the new one is "finished
the day highest". They overlap but are not the same population, so the two
values are not versions of one number — they are answers to two different
questions, and only the second one can serve as ground truth for features
computed during the day.

Publishes both, with the winner-set overlap, so nobody reads a change in the
figure as a change in the bot.

Read-only.  pyembed\\python.exe files\\_backtest_immutable_ns.py
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _compute_early_capture as CE  # noqa: E402
import immutable_labels as IL  # noqa: E402

DAYS = 120


def _rank_top_n(eod: dict, *, top_n: int) -> set:
    """Top-N per UTC day by EOD return — the same rule the immutable side uses,
    applied to the snapshot-derived returns."""
    by_day: dict[str, list] = {}
    for (day, symbol), value in eod.items():
        if isinstance(value, (int, float)):
            by_day.setdefault(day, []).append((float(value), symbol))
    out = set()
    for day, rows in by_day.items():
        if len(rows) < top_n:
            continue                    # a thin day mints no winners
        for _, symbol in sorted(rows, reverse=True)[:top_n]:
            out.add((day, symbol))
    return out


def main() -> int:
    cut = datetime.now(timezone.utc) - timedelta(days=DAYS)
    watchlist = CE.load_watchlist()

    # Hold the RULE constant and vary only the data source. The first version of
    # this script compared `label_top20` (global top-20 intersected with the
    # watchlist, ~3.8/day) against watchlist-top-20 (20/day by construction) and
    # would have reported the definitional gap as the leakage effect. Two
    # changes at once is not an experiment.
    _, leaky_eod = CE.load_winners(
        CE.ROOT / "files" / "top_gainer_dataset.jsonl", "label_top20", cut,
        watchlist=watchlist)
    leaky_winners = _rank_top_n(leaky_eod, top_n=20)
    imm_winners, imm_eod = IL.winners_by_day(top_n=20, watchlist=watchlist)

    # Compare only where both label sources have days, or the difference is
    # calendar coverage rather than labelling (TH-04).
    leaky_days = {d for d, _ in leaky_winners}
    imm_days = {d for d, _ in imm_winners}
    common = leaky_days & imm_days
    if not common:
        print("no overlapping days — cannot compare")
        return 1

    lw = {(d, s) for d, s in leaky_winners if d in common}
    iw = {(d, s) for d, s in imm_winners if d in common}
    overlap = lw & iw

    print("=" * 78)
    print("North Star ground truth · leaky snapshot vs immutable later-EOD")
    print("=" * 78)
    print(f"common days {len(common)}  ({min(common)}..{max(common)})")
    print()
    print(f"  {'winner set':<34}{'n':>8}{'per day':>10}")
    print(f"  {'snapshot-derived returns':<34}{len(lw):>8}{len(lw)/len(common):>10.1f}")
    print(f"  {'immutable (top-20 by EOD close)':<34}{len(iw):>8}{len(iw)/len(common):>10.1f}")
    print(f"  {'in both':<34}{len(overlap):>8}"
          f"{100*len(overlap)/max(1,len(iw)):>9.0f}%")
    print()

    uptime = CE.load_uptime(cut)
    full_days = uptime[0] if isinstance(uptime, tuple) else uptime
    entries = CE.load_entries(cut)
    first_entry, pnl_pairs = entries[0], entries[1]

    for name, winners, eod in (("snapshot returns", lw, leaky_eod),
                               ("immutable later-EOD", iw, imm_eod)):
        scoped = {k for k in winners if k[0] in set(full_days)} if full_days else winners
        res = CE.compute_north_star(scoped, eod, first_entry, pnl_pairs, name)
        print(f"  {name:<24}EC={res['early_capture']:.4f}  n={res['n']:<5}"
              f"cov={res['decomp_coverage']:.2f}  cap={res['decomp_capture_mean']:.2f}  "
              f"lead={res['decomp_time_lead_mean']:.2f}")

    print()
    print("The two figures answer different questions and must not be read as a")
    print("trend. The immutable one is the only one usable as ground truth for")
    print("features the bot computed during the day.")
    print()
    print("Universe caveat: the store covers 98 symbols (3 stale) against the")
    print("watchlist's 105, and begins 2026-01-26.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
