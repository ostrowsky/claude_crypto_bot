"""North-star: EarlyCapture@<label> = coverage * capture_ratio * time_lead_score

Two parallel north-stars (P1.1 2026-05-07):
  EarlyCapture@top20      — historic ground truth (label_top20 from dataset)
  EarlyCapture@sustained  — clean ground truth (label_sustained_uptrend from
                            dataset_v2, see sustained-uptrend-label-spec.md)

Per winner-day:
  coverage_flag    = 1 if entered, else 0
  capture_ratio    = clamp(realized_pnl / eod_return_pct, 0, 1)
  time_lead_score  = 1 - (entry_hour_UTC / 24)   (early UTC = higher)
EarlyCapture = mean(coverage * capture * time_lead) across all winner-days.
"""
from __future__ import annotations
import argparse, json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import config
except Exception:                                  # reporting must not need it
    config = None
NOW = datetime.now(timezone.utc)


def load_watchlist() -> set:
    """The tradeable universe. The canonical North Star (CLAUDE.md s1) is
    '#(top-20 in watchlist)' — top_gainer_dataset now spans a broader learning
    universe (volume-ranked, ~3-4x the watchlist), so an unfiltered denominator
    counts coins the bot CANNOT trade and understates real coverage ~3-4x."""
    try:
        return set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
    except Exception:
        return set()


def load_winners(dataset_path: Path, label_field: str, cut_dt: datetime,
                 watchlist: set | None = None):
    """Returns (winners_set, eod_ret_dict). If `watchlist` is given, only
    winners whose symbol is tradeable are counted (canonical NS definition)."""
    winners = set()
    eod_ret = {}
    with io.open(dataset_path, encoding="utf-8") as f:
        for ln in f:
            try: e = json.loads(ln)
            except: continue
            ts_ms = e.get("ts");
            if not ts_ms: continue
            dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
            if dt < cut_dt: continue
            sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
            if watchlist is not None and sym not in watchlist:
                continue
            if e.get(label_field) == 1:
                winners.add((d, sym))
            eod_ret[(d, sym)] = e.get("eod_return_pct")
    return winners, eod_ret


MIN_ACTIVE_HOURS = 18   # a full trading day for the bot; below this it was down


def load_uptime(cut_dt: datetime):
    """Which days did the bot actually run?

    The metric is a ratio over top-20 of the window, and a day the bot was down
    contributes only misses — so an outage reads exactly like a collapse in
    performance. On 2026-07-23..07-31 the bot was dead for 8 days and the report
    announced "~2 of 100" while the live days were at their best-ever level.

    A running day emits events almost every hour (400-2500/day); a dead day emits
    none and a partial day only covers part of the clock. Days that cover fewer
    than MIN_ACTIVE_HOURS are reported separately and excluded from the ratio.
    """
    hours: dict[str, set] = {}
    with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if '"event"' not in ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            ts = str(e.get("ts", ""))
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if dt < cut_dt:
                continue
            hours.setdefault(dt.strftime("%Y-%m-%d"), set()).add(dt.hour)
    full = {d for d, hh in hours.items() if len(hh) >= MIN_ACTIVE_HOURS}
    partial = {d: len(hh) for d, hh in hours.items() if len(hh) < MIN_ACTIVE_HOURS}
    return full, partial, hours


def load_entries(cut_dt: datetime):
    first_entry = {}; pnl_pairs = {}; entries = {}
    with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
        for ln in f:
            if '"event"' not in ln: continue
            try: e = json.loads(ln)
            except: continue
            ev = e.get("event","")
            if ev not in ("entry","exit"): continue
            ts = e.get("ts","")
            try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
            except: continue
            if dt < cut_dt: continue
            sym = e.get("sym") or e.get("symbol") or ""
            if not sym: continue
            d = dt.strftime("%Y-%m-%d")
            if ev == "entry":
                ep = float(e.get("price") or e.get("entry_price") or 0)
                entries[sym] = (dt, d, ep)
                prev = first_entry.get((d, sym))
                if prev is None or dt < prev[0]:
                    first_entry[(d, sym)] = (dt, ep)
            else:
                ent = entries.pop(sym, None)
                if not ent: continue
                ex_p = float(e.get("exit_price") or e.get("price") or 0)
                if ent[2] <= 0 or ex_p <= 0: continue
                pnl = (ex_p - ent[2]) / ent[2] * 100
                pnl_pairs[(ent[1], sym)] = pnl
    return first_entry, pnl_pairs


def move_relative_lead(entry_dt, *, open_ts, deadline_ts):
    """How early the alert was AGAINST THE MOVE, in [0, 1].

    `lead = 1 - (entry - open) / (deadline - open)`, where the deadline is the
    first crossing of +2.5% from the UTC open. 1.0 = at or before the open,
    0.0 = at or after the move was already underway.

    The clock-hour version this replaces (`1 - hour/24`) measured earliness in
    the CALENDAR: a coin that started moving at 20:00 and was alerted at 20:05 —
    the best the bot could do — scored 0.17, while an idle 02:00 buy scored 0.92.
    That is backwards for the stated objective, so a change that genuinely
    improved entry timing could have lowered the North Star.

    Returns **None** when there is no deadline to measure against (a
    daily-resolution label carries no crossing time). Scoring 0.0 there would
    assert the bot alerted late, which is a claim about the bot rather than
    about the data (TH-05).
    """
    if deadline_ts is None or open_ts is None or entry_dt is None:
        return None
    window = (deadline_ts - open_ts).total_seconds()
    if window <= 0:
        return None
    elapsed = (entry_dt - open_ts).total_seconds()
    return max(0.0, min(1.0, 1.0 - elapsed / window))


def compute_north_star(winners, eod_ret, first_entry, pnl_pairs, label_name: str,
                       *, lead_mode: str = "clock", deadlines: dict | None = None):
    """`lead_mode="move"` scores earliness against the coin's own move instead of
    the clock. A winner-day whose label carries no +2.5% crossing time cannot be
    scored that way and is EXCLUDED and counted, never scored 0.0 — a zero would
    assert the bot alerted late when the truth is that we cannot tell (TH-05)."""
    ec_scores = []; breakdown = []; skipped_no_deadline = 0
    deadlines = deadlines or {}
    for key in winners:
        d, sym = key
        ent = first_entry.get(key)
        coverage = 1.0 if ent else 0.0
        if ent:
            edt, ep = ent
            if lead_mode == "move":
                open_ts, dl_ts = deadlines.get(key, (None, None))
                time_lead = move_relative_lead(edt, open_ts=open_ts,
                                               deadline_ts=dl_ts)
                if time_lead is None:
                    skipped_no_deadline += 1
                    continue
            else:
                time_lead = 1.0 - (edt.hour / 24.0)
            pnl = pnl_pairs.get(key)
            eod = eod_ret.get(key)
            if pnl is not None and eod is not None:
                eod_p = float(eod)
                if abs(eod_p) <= 5: eod_p *= 100
                if abs(eod_p) >= 1.0:
                    cap = max(0.0, min(1.0, pnl / eod_p))
                else:
                    cap = 0.0
            else:
                cap = 0.0
        else:
            time_lead = 0.0; cap = 0.0
        score = coverage * cap * time_lead
        ec_scores.append(score)
        breakdown.append({"d": d, "sym": sym, "cov": coverage, "cap": cap,
                          "lead": time_lead, "score": score})
    n = len(ec_scores)
    mean_ec = sum(ec_scores)/max(1, n)
    mean_cov = sum(b["cov"] for b in breakdown)/max(1,n)
    entered = [b for b in breakdown if b["cov"] > 0]
    mean_cap = sum(b["cap"] for b in entered) / max(1, len(entered))
    mean_lead = sum(b["lead"] for b in entered) / max(1, len(entered))
    return {
        "label": label_name, "n": n,
        "lead_definition": ("move_relative" if lead_mode == "move" else "clock_hour"),
        "winners_without_deadline": skipped_no_deadline,
        "early_capture": mean_ec,
        "decomp_coverage": mean_cov,
        "decomp_capture_mean": mean_cap,
        "decomp_time_lead_mean": mean_lead,
        "_breakdown": breakdown,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    cut_dt = NOW - timedelta(days=args.days)

    first_entry, pnl_pairs = load_entries(cut_dt)
    watchlist = load_watchlist()  # canonical NS: tradeable universe only
    full_days, partial_days, _hours = load_uptime(cut_dt)

    # Top-20 (existing) — filtered to watchlist per the canonical definition
    top20_all, eod_ret = load_winners(ROOT/"files"/"top_gainer_dataset.jsonl",
                                      label_field="label_top20", cut_dt=cut_dt,
                                      watchlist=watchlist)
    # Uptime-adjusted: only days the bot could actually act on.
    top20 = {k for k in top20_all if k[0] in full_days}
    res_top20 = compute_north_star(top20, eod_ret, first_entry, pnl_pairs, "top20")
    res_raw = compute_north_star(top20_all, eod_ret, first_entry, pnl_pairs, "top20_raw")

    # Immutable later-EOD ground truth (TH-03). Published BESIDE the old value,
    # never in place of it: the historical series was computed on the snapshot
    # label, and substituting the loader would turn a change of provenance into
    # what looks like a change in the bot (TH-04).
    res_imm = None
    if getattr(config, "NS_IMMUTABLE_LABELS_ENABLED", False):
        try:
            import immutable_labels as IL
            # rank_before_filter=True reproduces the North Star's own
            # denominator, `watchlist INTERSECT global-top20` — the same
            # question label_top20 asks. Ranking inside the watchlist answers
            # an easier one and mints 20 winners a day regardless of the market.
            imm_all, imm_eod = IL.winners_by_day(top_n=20, watchlist=watchlist,
                                                 rank_before_filter=True)
            imm_all = {k for k in imm_all if k[0] >= cut_dt.strftime("%Y-%m-%d")}
            imm = {k for k in imm_all if k[0] in full_days}
            if imm:
                res_imm = compute_north_star(imm, imm_eod, first_entry,
                                             pnl_pairs, "top20_immutable")
        except Exception as exc:                      # never break the daily run
            print(f"[immutable labels unavailable: {exc}]")

    # Goal 2 (signal entry as early as possible) measured against the MOVE.
    # Published beside v2, never instead of it: the two answer different
    # questions and the value is EXPECTED to fall (TH-04).
    res_move = None
    if res_imm is not None and getattr(config, "NS_MOVE_RELATIVE_LEAD_ENABLED", False):
        try:
            import label_store as LS
            deadlines = {}
            for r in LS.LabelStore().records():
                if LS.resolution_of(r) != "1h" or not r.get("early_deadline_ts"):
                    continue                    # only intraday labels can time
                day = r["utc_day"]
                open_ts = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                dl = datetime.fromtimestamp(r["early_deadline_ts"] / 1000, timezone.utc)
                deadlines[(day, r["symbol"])] = (open_ts, dl)
            res_move = compute_north_star(imm, imm_eod, first_entry, pnl_pairs,
                                          "top20_move_lead", lead_mode="move",
                                          deadlines=deadlines)
        except Exception as exc:
            print(f"[move-relative lead unavailable: {exc}]")

    # Sustained (P1.1 — try v2 dataset, fall back if absent)
    res_sustained = None
    v2_path = ROOT/"files"/"top_gainer_dataset_v2.jsonl"
    if v2_path.exists():
        sustained, eod_ret_s = load_winners(v2_path,
                                            label_field="label_sustained_uptrend",
                                            cut_dt=cut_dt, watchlist=watchlist)
        res_sustained = compute_north_star(sustained, eod_ret_s, first_entry,
                                           pnl_pairs, "sustained")

    # Output
    print(f"=== NORTH-STAR · last {args.days}d ===\n")
    down = args.days - len(full_days)
    print(f"uptime: {len(full_days)}/{args.days} full days"
          + (f"  ·  DOWN/partial: {down} "
             f"({', '.join(sorted(partial_days)) or 'no data at all'})" if down else ""))
    if down:
        print(f"  metric below counts only the {len(full_days)} full days "
              f"(down/partial days are excluded, not scored as misses).")
        print(f"  for reference, counting all days: "
              f"EC={res_raw['early_capture']:.3f} cov={res_raw['decomp_coverage']:.2f} "
              f"(n={res_raw['n']}) — DO NOT read as performance")
    print()
    for r in [res_top20, res_imm, res_move, res_sustained]:
        if r is None: continue
        print(f"EarlyCapture@{r['label']:<16}  {r['early_capture']:.3f}  "
              f"(n={r['n']}, cov={r['decomp_coverage']:.2f}, "
              f"cap={r['decomp_capture_mean']:.2f}, "
              f"lead={r['decomp_time_lead_mean']:.2f})")
    if res_imm:
        # Both lines now ask the same question — `watchlist INTERSECT
        # global-top20` — since the label store covers the global universe.
        # What still differs is the window and which pairs exist in it.
        print(f"\n  both lines use the SAME rule now: global top-20 INTERSECT "
              f"watchlist  ({res_top20['n']} vs {res_imm['n']} winners over "
              f"{len(full_days)} full days)")
        print("    top20           = ranked from the snapshot's rolling-24h return")
        print("    top20_immutable = ranked from exchange klines at the day's close")
        print("    Residual gap is the window and the universe, not the rule: the")
        print("    store holds ~497 currently-listed pairs a day and cannot rank a")
        print("    pair delisted since (TH-05).")
    if res_sustained is None:
        print("\nEarlyCapture@sustained: dataset_v2 not found — run "
              "files/_backfill_sustained_uptrend.py first")

    # Top-5 winners by score (top20 — existing behaviour)
    bd = sorted(res_top20["_breakdown"], key=lambda x: -x["score"])
    print(f"\nTop-5 top-20 winners (highest EC):")
    for b in bd[:5]:
        print(f"  {b['d']} {b['sym']:<10} score={b['score']:.3f}  "
              f"(cov={b['cov']:.0f}, cap={b['cap']:.2f}, lead={b['lead']:.2f})")

    # METRIC_JSON for daily aggregator.
    #
    # The immutable value becomes PRIMARY once it is available, and the metric
    # is versioned rather than silently redefined: `_v2` so a reader comparing
    # today against last month sees two names, not one number that changed
    # meaning. The leaky value keeps travelling under `legacy_*` so the old
    # series stays reconstructable (TH-04).
    primary = res_imm or res_top20
    versioned = "NS_EarlyCapture_top20_v2" if res_imm else "NS_EarlyCapture_top20"
    metric = {
        "metric": versioned,
        # Which top-20 `n` counts. The recall denominator changed once already
        # (April), and PROJECT_CONTEXT records the two methodologies as
        # "несопоставимы напрямую"; naming it here makes the next silent
        # redefinition visible in the artifact itself.
        "denominator": ("global_top20_intersect_watchlist_from_label_store"
                        if res_imm else
                        "top20_within_watchlist_from_top_gainer_dataset"),
        "label_provenance": ("immutable_later_eod_klines" if res_imm
                             else "rolling_24h_same_snapshot"),
        "days_window": args.days,
        "days_full": len(full_days),
        "days_down_or_partial": args.days - len(full_days),
        "raw_early_capture_all_days": res_raw["early_capture"],
        "n": primary["n"],
        "early_capture": primary["early_capture"],
        "decomp_coverage": primary["decomp_coverage"],
        "decomp_capture_mean": primary["decomp_capture_mean"],
        "decomp_time_lead_mean": primary["decomp_time_lead_mean"],
        # The old series, kept so it stays reconstructable rather than lost.
        "legacy_metric": "NS_EarlyCapture_top20",
        "legacy_label_provenance": "rolling_24h_same_snapshot",
        "legacy_n": res_top20["n"],
        "legacy_early_capture": res_top20["early_capture"],
        "legacy_coverage": res_top20["decomp_coverage"],
    }
    if res_imm:
        # A second value, not a replacement. The two are computed on the same
        # rule (day's top-20 by EOD return) and differ only in where the return
        # comes from, so a gap between them is a labelling gap and nothing else.
        metric["immutable_label_provenance"] = "immutable_later_eod_klines"
        metric["immutable_n"] = res_imm["n"]
        metric["immutable_early_capture"] = res_imm["early_capture"]
        metric["immutable_coverage"] = res_imm["decomp_coverage"]
        metric["immutable_capture_mean"] = res_imm["decomp_capture_mean"]
        metric["immutable_time_lead_mean"] = res_imm["decomp_time_lead_mean"]
        # ~497 currently-listed pairs a day over a 240-day window. Pairs
        # delisted since cannot be ranked, so a past day's global top-20 may be
        # missing a coin that was genuinely in it (TH-05).
        metric["immutable_universe_note"] = (
            "global daily-kline universe ~497 pairs/day; delisted pairs absent")
        # Carried so a downstream reader cannot difference the two by accident.
        metric["immutable_denominator"] = "global_top20_intersect_watchlist_from_label_store"
        # The same denominator as the primary since the store went global. The
        # residual difference is the window (closed UTC day vs rolling 24h) and
        # the universe (~497 currently-listed pairs, no delisted ones), not the
        # rule — so the two may now be read against each other.
        metric["immutable_comparable_to_primary"] = True
    metric["lead_definition"] = primary.get("lead_definition", "clock_hour")
    if res_move:
        metric["move_lead_metric"] = "NS_EarlyCapture_top20_v3"
        metric["move_lead_early_capture"] = res_move["early_capture"]
        metric["move_lead_n"] = res_move["n"]
        metric["move_lead_mean"] = res_move["decomp_time_lead_mean"]
        # Winner-days whose label has no +2.5% crossing time. Counted, because a
        # metric must know what it cannot see rather than score it zero.
        metric["move_lead_winners_without_deadline"] = res_move["winners_without_deadline"]
    if res_sustained:
        metric["sustained_n"] = res_sustained["n"]
        metric["sustained_early_capture"] = res_sustained["early_capture"]
        metric["sustained_coverage"] = res_sustained["decomp_coverage"]
        metric["sustained_capture_mean"] = res_sustained["decomp_capture_mean"]
    print("\nMETRIC_JSON:" + json.dumps(metric))


if __name__ == "__main__":
    main()
