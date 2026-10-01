"""What the gate-off replays would cost in messages, and at what precision.

The goal criterion (goal_validator) judges early capture and per-trade pnl; it
does not see the message stream, and the operator's other requirement is "no
junk signals" (31 messages/day at 13.6% precision today). For each unsupported
lock: added candidates per day (one per coin-day, the downstream pass rate p
applied for the expected number of messages), and their precision -- share on
immutable top-20 winner-days -- beside the current entries'.
Also the ML gate on the current model's era only (2026-09-07, peak label).
"""
import collections
import sys
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import goal_validator as GV  # noqa: E402
import pipeline_replay_validator as RV  # noqa: E402

CASES = [("mode_range_quality", ("mode_range_quality",), "2026-05-01"),
         ("ml_proba_zone", ("ml_zone", "ml_proba_zone"), "2026-05-01"),
         ("ml_proba_zone since 09-07", ("ml_zone", "ml_proba_zone"), "2026-09-07")]


def main():
    for name, codes, since in CASES:
        spec = RV.ReplaySpec(frozenset(codes), None, lambda e, c: True, (), frozenset({"relax"}), "")
        rows = list(RV._iter_rows(spec, since))
        winners = {(d, s) for d, s, _, _ in GV.winner_days(since)}
        full_days = {d for d, _, _, _ in GV.winner_days(since)}
        past = collections.defaultdict(set)
        added, ent = set(), set()
        for e in rows:
            day = e["_dt"].strftime("%Y-%m-%d")
            if day not in full_days:
                continue
            if e["_blocked_here"]:
                added.add((day, e["sym"]))
            else:
                past[(e["sym"], e["_dt"].replace(minute=0, second=0, microsecond=0))].add(
                    "entry" if e.get("event") == "entry" else "b")
                if e.get("event") == "entry":
                    ent.add((day, e["sym"]))
        p = sum(1 for v in past.values() if "entry" in v) / max(1, len(past))
        new_only = added - ent
        ndays = max(1, len(full_days))
        prec_add = sum(1 for k in new_only if k in winners) / max(1, len(new_only))
        prec_cur = sum(1 for k in ent if k in winners) / max(1, len(ent))
        if since == "2026-09-07":
            r = GV.validate_gate_off(codes, since)
            print("\n%s: goal replay -> %s | %s" % (name, r["verdict"], r.get("reason")))
        print("\n%s (from %s, %d days with the bot up and labels):" % (name, since, ndays))
        print("  current entries: %d coin-days (%.1f/day), precision %.3f" % (len(ent), len(ent) / ndays, prec_cur))
        print("  gate removed: +%d new coin-days blocked by it (%.1f/day); expected messages x p=%.2f -> +%.1f/day; "
              "precision of the added %.3f (%s the current)" % (
                  len(new_only), len(new_only) / ndays, p, p * len(new_only) / ndays, prec_add,
                  "above" if prec_add > prec_cur else "below"))


if __name__ == "__main__":
    main()
