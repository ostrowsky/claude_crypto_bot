"""Re-verify the do_not_touch gate locks by the GOAL, on the maximum period.

The locks (.runtime/pipeline/do_not_touch.json) were last verified 2026-05-28 by
a proxy -- the blocked bucket's 5-bar return vs the entries' -- and are 126 days
past a 30-day budget (TH-10). This asks the project's question instead: if the
gate were removed entirely, would the bot enter more winner-days before the
first +2.5% crossing, and would the added trades be non-inferior?
(goal_validator.validate_gate_off; the same rule as every L3 verdict.)

  confirmed     removal rejected (no goal gain, or the trades fail -0.10 pp)
  unsupported   removal accepted -- the lock is not backed by the goal; goes to
                the operator, never changed here
  review/data   needs_review / needs_data, as returned

mtf and cooldown are locked too but never appear as `blocked` events, so the
bot's own decisions cannot replay them; they are reported as such.

    pyembed\\python.exe files\\_replay_locked_gates_goal.py            (read-only)
    pyembed\\python.exe files\\_replay_locked_gates_goal.py --write    (record the result)
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import goal_validator as GV  # noqa: E402
import pipeline_lib as PL  # noqa: E402

GATES = {
    "trend_1h_chop": dict(codes=("trend_chop", "trend_1h_chop"), tf="1h", control_mode="trend"),
    # the gate's population depends on the model: only the current model's era
    # (peak label, 2026-09-07) says anything about the gate running now
    "ml_proba_zone": dict(codes=("ml_zone", "ml_proba_zone"), tf=None, control_mode="", since="2026-09-07"),
    "mode_range_quality": dict(codes=("mode_range_quality",), tf=None, control_mode=""),
    "open_cluster_cap": dict(codes=("open_cluster_cap",), tf=None, control_mode=""),
}
NOT_REPLAYABLE = ("mtf", "cooldown")
STATUS = {"reject": "confirmed", "accept": "unsupported"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=GV.SINCE_DEFAULT)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    full_days = None
    results = {}
    for name, g in GATES.items():
        since = max(a.since, g.get("since", a.since))
        winners = GV.winner_days(since)
        print("\n%s: from %s, winner-days %d" % (name, since, len(winners)))
        r = GV.validate_gate_off(g["codes"], since, tf=g["tf"], control_mode=g["control_mode"], winners=winners)
        st = STATUS.get(r["verdict"], r["verdict"])
        results[name] = {"status": st, "verdict": r["verdict"], "reason": r.get("reason"), "since": since,
                         "goal": r.get("goal"), "per_trade": {k: v for k, v in (r.get("per_trade") or {}).items()
                                                               if k != "by_month"}}
        print("\n%-20s %-12s %s" % (name, st.upper(), r.get("reason")))
    for name in NOT_REPLAYABLE:
        results[name] = {"status": "not_replayable",
                         "reason": "never logged as a blocked event; the bot's decisions cannot replay it"}
        print("\n%-20s NOT_REPLAYABLE (no blocked events)" % name)
    if a.write:
        p = PL.DO_NOT_TOUCH
        dnt = json.loads(p.read_text(encoding="utf-8"))
        today = date.today().isoformat()
        dnt.setdefault("goal_verification", {})[today] = results
        keep, contested = [], dnt.get("contested", [])
        for x in dnt.get("gates", []):
            r = results.get(x["name"])
            if r and r["status"] == "unsupported":
                # the goal does not back the lock; it is NOT relaxed here -- the
                # operator decides (the junk cost is in the spec)
                contested.append(dict(x, contested_on=today, goal_evidence=r["reason"]))
                continue
            if r and r["status"] == "confirmed":
                x["last_verified"] = today
                x["verified_via"] = "goal_validator.validate_gate_off"
            keep.append(x)
        dnt["gates"], dnt["contested"] = keep, contested
        # every replayable lock was re-run on the goal; what is not replayable says so
        dnt["last_verified"] = today
        dnt["verified_via"] = "goal criterion, gate removed entirely (gate-locks-goal-1001-spec.md)"
        p.write_text(json.dumps(dnt, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nwritten to %s; last_verified %s" % (p, dnt.get("last_verified")))
    return results


if __name__ == "__main__":
    main()
