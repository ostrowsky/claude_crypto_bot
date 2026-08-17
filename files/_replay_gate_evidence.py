"""Re-verify the do_not_touch gate locks — the TH-10 expiry finding.

`do_not_touch.json` protects gates proven NOT to over-block, and blocks any
hypothesis that would relax them. Its evidence was verified 2026-05-28 against a
30-day budget, so it is ~81 days stale and the harness fails closed.

Re-running the canonical blocked-bucket counterfactual is not enough on its own.
The stored evidence was computed over the WHOLE critic_dataset, which spans
several behaviour changes (soft gate 06-01, trail rollback 06-05,
fallback-to-trend 06-12, bandit label rebuild 08-13). A gate verified across
policy eras has not been verified under the policy running now — the population
it rejects today is not the one it rejected in April.

So both are published: the maximum-period figure (comparable with the stored
one) and the current-epoch figure (the one that describes today). Where they
disagree, the epoch figure wins and the max-period one explains why.

Read-only unless --write is passed.

    pyembed\\python.exe files\\_replay_gate_evidence.py
    pyembed\\python.exe files\\_replay_gate_evidence.py --epoch-start 2026-08-13

Spec: docs/specs/features/gate-evidence-replay-spec.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pipeline_lib as PL  # noqa: E402

# The last behaviour change that alters which candidates the gates ever see.
# Anything earlier is a different policy and a different rejected population.
DEFAULT_EPOCH_START = "2026-08-13"
MIN_N = 20                       # below this the counterfactual is not evidence


def _day_of(event: dict) -> str | None:
    """critic_dataset rows carry `ts_signal` (ISO) and `bar_ts` (ms).

    The first version of this looked for `ts` / `ts_ms` / `decision.ts`, none of
    which exist here, so every row silently fell out of the epoch window and the
    epoch column read n=0 across the board — which looks exactly like "the bot
    blocked nothing recently" rather than "the filter never matched".
    """
    iso = event.get("ts_signal")
    if isinstance(iso, str) and len(iso) >= 10:
        return iso[:10]
    ts = event.get("bar_ts")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts > 1e11:
        ts /= 1000.0
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def collect(epoch_start: str) -> tuple[dict, dict]:
    """One pass over critic_dataset, bucketed by gate, for both windows."""
    crit = PL.FILES_DIR / "critic_dataset.jsonl"
    full = {"take": [], "gates": {}}
    epoch = {"take": [], "gates": {}}
    if not crit.exists():
        return full, epoch

    for e in PL.iter_jsonl(crit):
        dec = e.get("decision") or {}
        lab = e.get("labels") or {}
        r5 = lab.get("ret_5")
        if r5 is None:
            continue
        try:
            r5 = float(r5)
        except (TypeError, ValueError):
            continue
        day = _day_of(e)
        in_epoch = day is not None and day >= epoch_start
        act = dec.get("action")
        if act == "take":
            full["take"].append(r5)
            if in_epoch:
                epoch["take"].append(r5)
        elif act == "blocked":
            gate = dec.get("reason_code")
            if not gate:
                continue
            full["gates"].setdefault(gate, []).append(r5)
            if in_epoch:
                epoch["gates"].setdefault(gate, []).append(r5)
    return full, epoch


def verdict_for(bucket: list, take: list) -> dict:
    """A gate over-blocks iff what it rejected out-returned what we took."""
    n = len(bucket)
    if n < MIN_N:
        # Fail closed: too little evidence is not evidence of correctness.
        return {"available": False, "n": n,
                "reason": f"n={n} < {MIN_N}; cannot verify, lock stands"}
    take_avg = sum(take) / len(take) if take else 0.0
    b_avg = sum(bucket) / n
    sd = (sum((x - b_avg) ** 2 for x in bucket) / n) ** 0.5
    sharpe = (b_avg / sd * math.sqrt(n)) if sd > 0 else 0.0
    miss = b_avg - take_avg
    return {
        "available": True, "n": n,
        "take_baseline_avg_r5": round(take_avg, 4),
        "blocked_avg_r5": round(b_avg, 4),
        "miss_vs_take": round(miss, 4),
        "sharpe_sqrt_n": round(sharpe, 2),
        # Over-blocking needs BOTH a positive miss and a meaningful Sharpe:
        # a positive average on 25 noisy rows is not a finding.
        "over_blocking": bool(miss > 0 and sharpe > 1.5),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-start", default=DEFAULT_EPOCH_START)
    ap.add_argument("--write", action="store_true",
                    help="refresh do_not_touch.json when every lock re-verifies")
    args = ap.parse_args(argv)

    dnt = PL.load_do_not_touch()
    locked = [g["name"] for g in dnt.get("gates", [])]
    stored = {g["name"]: g for g in dnt.get("gates", [])}

    full, epoch = collect(args.epoch_start)

    print("=" * 84)
    print("do_not_touch gate locks · re-verification")
    print("=" * 84)
    print(f"stored evidence   last_verified={dnt.get('last_verified')} "
          f"budget={dnt.get('verify_every_days')}d")
    _avg = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    print(f"current epoch     from {args.epoch_start}")
    # `miss` is blocked_avg MINUS take_avg, so a positive miss against a deeply
    # negative baseline means "lost less than we did", not "was profitable".
    # Without the baseline on screen the column reads backwards.
    print(f"take baseline     epoch avg_r5={_avg(epoch['take']):+.3f} "
          f"(n={len(epoch['take'])})   max-period avg_r5={_avg(full['take']):+.3f} "
          f"(n={len(full['take'])})")
    print()
    print(f"  {'gate':<22}{'n max':>7}{'miss':>9}{'Sharpe':>8}"
          f"{'n epoch':>9}{'miss':>9}{'Sharpe':>8}  verdict")

    still_locked, unverifiable, over = [], [], []
    for gate in locked:
        vf = verdict_for(full["gates"].get(gate, []), full["take"])
        ve = verdict_for(epoch["gates"].get(gate, []), epoch["take"])

        if ve["available"]:
            src, tag = ve, "epoch"
        elif vf["available"]:
            src, tag = vf, "max-period only"
        else:
            src, tag = None, "no evidence"

        # Max-period evidence is NOT re-verification under the current policy:
        # it mixes eras in which the gate saw a different rejected population.
        # The first version counted it as "confirmed", contradicting the very
        # caveat this script prints.
        if src is None:
            unverifiable.append(gate); note = "LOCK STANDS (no evidence)"
        elif src["over_blocking"]:
            over.append(gate); note = f"OVER-BLOCKING ({tag})"
        elif tag == "epoch":
            still_locked.append(gate); note = "confirmed (current policy)"
        else:
            unverifiable.append(gate)
            note = "LOCK STANDS (max-period only, not current policy)"

        def cell(v, key, fmt="{:>9.3f}"):
            return fmt.format(v[key]) if v["available"] else f"{'-':>9}"
        print(f"  {gate:<22}{vf.get('n',0):>7}{cell(vf,'miss_vs_take')}"
              f"{cell(vf,'sharpe_sqrt_n','{:>8.2f}')}"
              f"{ve.get('n',0):>9}{cell(ve,'miss_vs_take')}"
              f"{cell(ve,'sharpe_sqrt_n','{:>8.2f}')}  {note}")

    print()
    print(f"confirmed still correct : {len(still_locked)}")
    print(f"now over-blocking       : {len(over)}"
          + (f"  -> {', '.join(over)}" if over else ""))
    print(f"unverifiable            : {len(unverifiable)}"
          + (f"  -> {', '.join(unverifiable)}" if unverifiable else ""))
    print()
    print("Reading it: a gate with no epoch evidence is NOT re-verified — the")
    print("lock stands because absence of data is not evidence of correctness")
    print("(TH-05, fail closed). Refreshing the timestamp for such a gate would")
    print("convert 'we did not measure' into 'we measured and it was fine'.")

    if args.write:
        if unverifiable or over:
            print("\nrefusing --write: not every lock re-verified under the "
                  "current policy; a blanket timestamp refresh would be a lie.")
            return 1
        dnt["last_verified"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dnt["verified_via"] = (f"_replay_gate_evidence.py, epoch from "
                               f"{args.epoch_start}")
        for g in dnt["gates"]:
            g["last_verified"] = dnt["last_verified"]
        PL.DO_NOT_TOUCH.write_text(json.dumps(dnt, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nrefreshed {PL.DO_NOT_TOUCH}")
    else:
        _ = stored
        print("\n(read-only; pass --write to refresh do_not_touch.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
