"""
Decoupling shadow feature check — 2026-06-28 scheduled audit.
Reads bot_events.jsonl (streamed), reports on shadow logging coverage
and compares outcomes between decoupling_flag=True vs False entries.
No writes to config or bot state.
"""
import json
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

EVENTS_FILE = Path(__file__).parent / "bot_events.jsonl"
CRITIC_FILE = Path(__file__).parent / "critic_dataset.jsonl"

CUTOFF_TS = datetime(2026, 6, 21, 0, 0, 0, tzinfo=timezone.utc).timestamp()

def parse_ts(rec):
    ts = rec.get("ts")
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        # ISO string
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def main():
    print("=" * 70)
    print("DECOUPLING SHADOW CHECK — 2026-06-28")
    print("Window: 2026-06-21 00:00 UTC to present")
    print("=" * 70)

    # ── Pass 1: bot_events.jsonl ──────────────────────────────────────────
    entries_recent = []         # all entry events in window
    entries_with_decouple = []  # entry events that have decoupling fields
    exit_map = defaultdict(list)  # sym -> list of exit records (pnl, entry_price, ts)

    total_events = 0
    total_entries_recent = 0
    parse_errors = 0

    print(f"\nStreaming {EVENTS_FILE.name} ...")
    with open(EVENTS_FILE, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                parse_errors += 1
                continue
            total_events += 1

            event = rec.get("event")
            ts = parse_ts(rec)

            if event == "entry":
                if ts is not None and ts >= CUTOFF_TS:
                    total_entries_recent += 1
                    entries_recent.append(rec)
                    has_dec = "decoupling_score" in rec
                    if has_dec:
                        entries_with_decouple.append(rec)

            elif event == "exit":
                sym = rec.get("sym")
                if sym:
                    exit_map[sym].append({
                        "ts": ts,
                        "entry_price": rec.get("entry_price"),
                        "pnl_pct": rec.get("pnl_pct"),
                        "reason": rec.get("reason"),
                        "bars_held": rec.get("bars_held"),
                    })

    print(f"  Total events scanned: {total_events:,}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Entry events since 2026-06-21: {total_entries_recent}")
    print(f"  With decoupling_score field:   {len(entries_with_decouple)}")
    coverage_pct = (len(entries_with_decouple) / max(1, total_entries_recent)) * 100
    print(f"  Coverage: {coverage_pct:.1f}%")

    if len(entries_with_decouple) == 0:
        print("\n⚠️  NO decoupling fields logged — shadow feature likely not active.")
        print("    Check: DECOUPLING_SHADOW_ENABLED=True and bot restarted after 2026-05-07.")
        _write_report(total_entries_recent, 0, 0, None, None, None)
        return

    # ── Breakdown by decoupling_flag ──────────────────────────────────────
    flagged = [r for r in entries_with_decouple if r.get("decoupling_flag") is True]
    unflagged = [r for r in entries_with_decouple if r.get("decoupling_flag") is False]
    flag_none = [r for r in entries_with_decouple if r.get("decoupling_flag") is None]

    print(f"\n  decoupling_flag=True:  {len(flagged)}")
    print(f"  decoupling_flag=False: {len(unflagged)}")
    print(f"  decoupling_flag=None:  {len(flag_none)}")

    # Score stats for flagged
    scores = [r["decoupling_score"] for r in entries_with_decouple if r.get("decoupling_score") is not None]
    if scores:
        print(f"\n  decoupling_score stats (n={len(scores)}):")
        print(f"    mean={sum(scores)/len(scores):.4f}  "
              f"min={min(scores):.4f}  max={max(scores):.4f}")

    corrs = [r["decoupling_corr"] for r in entries_with_decouple if r.get("decoupling_corr") is not None]
    if corrs:
        print(f"  decoupling_corr stats (n={len(corrs)}):")
        print(f"    mean={sum(corrs)/len(corrs):.4f}  "
              f"min={min(corrs):.4f}  max={max(corrs):.4f}")

    # Mode breakdown
    from collections import Counter
    mode_counts_flag = Counter(r.get("mode", "?") for r in flagged)
    mode_counts_unflag = Counter(r.get("mode", "?") for r in unflagged)
    print(f"\n  Mode distribution (flagged):   {dict(mode_counts_flag.most_common(6))}")
    print(f"  Mode distribution (unflagged): {dict(mode_counts_unflag.most_common(6))}")

    # ── Pass 2: match with exits for resolved P&L ─────────────────────────
    print("\n" + "─" * 70)
    print("OUTCOME MATCHING (entry → exit via sym+entry_price)")

    def find_exit(entry_rec):
        sym = entry_rec.get("sym")
        ep = entry_rec.get("price")
        ts_entry = parse_ts(entry_rec)
        if sym not in exit_map or ep is None:
            return None
        for ex in exit_map[sym]:
            ep_exit = ex.get("entry_price")
            ts_ex = ex.get("ts")
            if ep_exit is None:
                continue
            # price match within 0.01% and exit after entry
            if abs(ep_exit - ep) / max(abs(ep), 1e-9) < 0.0001:
                if ts_ex is None or ts_entry is None or ts_ex >= ts_entry:
                    return ex
        return None

    def outcome_stats(entries, label):
        resolved = []
        unresolved = 0
        for e in entries:
            ex = find_exit(e)
            if ex is not None and ex.get("pnl_pct") is not None:
                resolved.append(ex["pnl_pct"])
            else:
                unresolved += 1
        n_res = len(resolved)
        if n_res == 0:
            return (label, len(entries), 0, None, None, None)
        avg = sum(resolved) / n_res
        wins = sum(1 for p in resolved if p > 0)
        win_pct = wins / n_res * 100
        return (label, len(entries), n_res, avg, win_pct, resolved)

    st_flag   = outcome_stats(flagged,   "flag=True ")
    st_unflag = outcome_stats(unflagged, "flag=False")

    print(f"\n  {'Group':<12} {'n_entries':>10} {'n_resolved':>11} {'avg_pnl%':>10} {'win%':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*11} {'─'*10} {'─'*8}")
    for label, n_entries, n_res, avg, win_pct, _ in [st_flag, st_unflag]:
        avg_s  = f"{avg:+.3f}" if avg is not None else "—"
        win_s  = f"{win_pct:.1f}" if win_pct is not None else "—"
        print(f"  {label:<12} {n_entries:>10} {n_res:>11} {avg_s:>10} {win_s:>8}")

    # ── Sufficiency check ─────────────────────────────────────────────────
    MIN_RESOLVED = 30
    n_flag_res   = st_flag[2]
    n_unflag_res = st_unflag[2]

    print("\n" + "─" * 70)
    print("SUFFICIENCY CHECK (threshold: ≥30 resolved per group for bandit shadow-replay)")
    print(f"  flag=True  resolved: {n_flag_res}  {'✅' if n_flag_res >= MIN_RESOLVED else '❌ INSUFFICIENT'}")
    print(f"  flag=False resolved: {n_unflag_res}  {'✅' if n_unflag_res >= MIN_RESOLVED else '❌ INSUFFICIENT'}")

    # ── Verdict ───────────────────────────────────────────────────────────
    shadow_live = len(entries_with_decouple) > 0
    enough_data = n_flag_res >= MIN_RESOLVED and n_unflag_res >= MIN_RESOLVED

    print("\n" + "=" * 70)
    print("VERDICT")
    print("─" * 70)
    print(f"(a) Shadow logging active: {'YES' if shadow_live else 'NO ⚠️'}")
    print(f"    {len(entries_with_decouple)} of {total_entries_recent} recent entries carry decoupling_score ({coverage_pct:.0f}%)")
    print(f"    Flagged rate: {len(flagged)}/{len(entries_with_decouple)} = {len(flagged)/max(1,len(entries_with_decouple))*100:.1f}%")
    print()
    print(f"(b) Enough data for bandit shadow-replay: {'YES' if enough_data else 'NOT YET'}")
    if not enough_data:
        need_more = MAX(MIN_RESOLVED - n_flag_res, 0) if n_flag_res < MIN_RESOLVED else 0
        need_more_u = max(MIN_RESOLVED - n_unflag_res, 0) if n_unflag_res < MIN_RESOLVED else 0
        if need_more > 0:
            print(f"    flag=True group needs {need_more} more resolved entries")
        if need_more_u > 0:
            print(f"    flag=False group needs {need_more_u} more resolved entries")
    print()
    if enough_data:
        print("(c) Next step: run bandit shadow-replay adding decoupling_score to")
        print("    LinUCB context. Must show NS (early top-20 capture) lift without")
        print("    recall loss before enabling DECOUPLING_GATE_ENABLED=True.")
    else:
        print("(c) Next step: WAIT. Continue shadow logging for more resolved entries.")
        print("    Re-run this check in ~7 days.")
    print("=" * 70)

    # ── Write report ──────────────────────────────────────────────────────
    _write_report(
        total_entries_recent=total_entries_recent,
        n_with_decouple=len(entries_with_decouple),
        n_flagged=len(flagged),
        st_flag=st_flag,
        st_unflag=st_unflag,
        shadow_live=shadow_live,
        enough_data=enough_data,
        flagged_entries=flagged,
        unflagged_entries=unflagged,
        all_entries=entries_with_decouple,
    )


def MAX(a, b):
    return a if a > b else b


def _write_report(total_entries_recent, n_with_decouple, n_flagged,
                  st_flag, st_unflag, shadow_live, enough_data=False,
                  flagged_entries=None, unflagged_entries=None, all_entries=None):
    report_dir = Path(__file__).parent.parent / "docs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "2026-06-28-decoupling-shadow-check.md"

    flagged_entries = flagged_entries or []
    unflagged_entries = unflagged_entries or []
    all_entries = all_entries or []

    if st_flag is not None:
        _, n_fe, n_fr, avg_flag, win_flag, _ = st_flag
        _, n_ue, n_ur, avg_unflag, win_unflag, _ = st_unflag
    else:
        n_fe = n_fr = n_ue = n_ur = 0
        avg_flag = win_flag = avg_unflag = win_unflag = None

    coverage_pct = n_with_decouple / max(1, total_entries_recent) * 100
    flag_rate = n_flagged / max(1, n_with_decouple) * 100

    avg_flag_s   = f"{avg_flag:+.3f}%" if avg_flag is not None else "—"
    win_flag_s   = f"{win_flag:.1f}%"  if win_flag is not None else "—"
    avg_unflag_s = f"{avg_unflag:+.3f}%" if avg_unflag is not None else "—"
    win_unflag_s = f"{win_unflag:.1f}%"  if win_unflag is not None else "—"

    shadow_status = "YES" if shadow_live else "NO ⚠️ (feature may be disabled or bot not restarted)"
    data_status   = "YES — ready for bandit shadow-replay" if enough_data else "NOT YET — need more resolved entries"
    next_step = (
        "Run bandit shadow-replay: add `decoupling_score` to LinUCB context vector "
        "(dim-migrate saved A/b like RM-22 Step B). Must show NS early top-20 "
        "capture lift without recall loss before enabling `DECOUPLING_GATE_ENABLED=True`."
    ) if enough_data else (
        "Continue shadow logging. Re-run this check in ~7 days (next scheduled run: 2026-07-05). "
        "Do NOT change `DECOUPLING_GATE_ENABLED` — keep `False`."
    )

    lines = [
        "# Decoupling Shadow Feature Check — 2026-06-28",
        "",
        "**Scheduled audit.** Read-only analysis; no config changes.",
        f"**Window analyzed:** 2026-06-21 00:00 UTC → 2026-06-28 (today)",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Recent entry events (since 2026-06-21) | {total_entries_recent} |",
        f"| With `decoupling_score` logged | {n_with_decouple} ({coverage_pct:.0f}%) |",
        f"| `decoupling_flag=True` (flagged) | {n_flagged} ({flag_rate:.1f}% of logged) |",
        "",
        "## Outcome comparison (entry → exit P&L)",
        "",
        f"| Group | n entries | n resolved | avg pnl% | win% |",
        f"|-------|-----------|------------|----------|------|",
        f"| flag=True | {n_fe} | {n_fr} | {avg_flag_s} | {win_flag_s} |",
        f"| flag=False | {n_ue} | {n_ur} | {avg_unflag_s} | {win_unflag_s} |",
        "",
        "## Sufficiency (threshold: ≥30 resolved per group)",
        "",
        f"- `flag=True` resolved: {n_fr}  {'✅' if n_fr >= 30 else '❌ INSUFFICIENT'}",
        f"- `flag=False` resolved: {n_ur}  {'✅' if n_ur >= 30 else '❌ INSUFFICIENT'}",
        "",
        "## Verdict",
        "",
        f"**(a) Shadow logging active:** {shadow_status}",
        "",
        f"**(b) Enough data for bandit shadow-replay:** {data_status}",
        "",
        f"**(c) Recommended next step:** {next_step}",
        "",
        "## Notes",
        "",
        "- `DECOUPLING_GATE_ENABLED` remains `False` — no gate wired, pure shadow.",
        "- Acceptance criteria per spec: ≥7d shadow + bandit shadow-replay showing",
        "  NS (early top-20 capture) lift without recall loss.",
        "- Generated by `files/_check_decoupling_shadow.py` (read-only, no side-effects).",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
