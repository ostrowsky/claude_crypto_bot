from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
REPORT_DIR = RUNTIME_DIR / "reports"

BOT_EVENTS_FILE = ROOT / "bot_events.jsonl"
CRITIC_FILE = ROOT / "critic_dataset.jsonl"
ML_FILE = ROOT / "ml_dataset.jsonl"
RL_STATUS_FILE = RUNTIME_DIR / "rl_worker_status.json"
TRAIN_REPORT_FILE = ROOT / "ml_candidate_ranker_report.json"
SHADOW_REPORT_FILE = ROOT / "ml_candidate_ranker_shadow_report.json"

LOCAL_TZ = ZoneInfo("Europe/Budapest")
BELGRADE_TZ = ZoneInfo("Europe/Belgrade")
TOP_GAINER_FILE = ROOT / "top_gainer_dataset.jsonl"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _parse_utc_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _local_day_bounds(target_day: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_day, datetime.min.time(), tzinfo=LOCAL_TZ)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _count_rows_in_day(path: Path, start_utc: datetime, end_utc: datetime) -> int:
    count = 0
    for rec in _iter_jsonl(path):
        ts = _parse_utc_ts(rec.get("ts_signal")) or _parse_utc_ts(rec.get("ts"))
        if ts and start_utc <= ts < end_utc:
            count += 1
    return count


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_report(
    target_day: date,
    *,
    bot_events_file: Path = BOT_EVENTS_FILE,
    critic_file: Path = CRITIC_FILE,
    ml_file: Path = ML_FILE,
    rl_status_file: Path = RL_STATUS_FILE,
    train_report_file: Path = TRAIN_REPORT_FILE,
    shadow_report_file: Path = SHADOW_REPORT_FILE,
) -> Dict[str, Any]:
    start_utc, end_utc = _local_day_bounds(target_day)
    bot_rows = list(_iter_jsonl(bot_events_file))
    day_events = []
    for rec in bot_rows:
        ts = _parse_utc_ts(rec.get("ts"))
        if ts and start_utc <= ts < end_utc:
            day_events.append(rec)

    ranker_shadow = [r for r in day_events if r.get("event") == "ranker_shadow"]
    bot_take_ranker_skip = [
        r for r in ranker_shadow
        if r.get("bot_action") == "take" and not bool(r.get("ranker_take"))
    ]
    bot_blocked_ranker_take = [
        r for r in ranker_shadow
        if r.get("bot_action") == "blocked" and bool(r.get("ranker_take"))
    ]

    worst_take_by_proba = sorted(
        bot_take_ranker_skip,
        key=lambda r: (
            float(r.get("ranker_proba", 0.0)),
            -float(r.get("candidate_score", 0.0)),
        ),
    )[:10]
    missed_by_proba = sorted(
        bot_blocked_ranker_take,
        key=lambda r: (
            -float(r.get("ranker_proba", 0.0)),
            -float(r.get("candidate_score", 0.0)),
        ),
    )[:10]

    blocked_counts = Counter(str(r.get("reason_code", r.get("signal_type", ""))) for r in bot_blocked_ranker_take)
    mode_counts = Counter(str(r.get("mode", "")) for r in ranker_shadow)
    symbol_counts = Counter(str(r.get("sym", "")) for r in ranker_shadow)

    report = {
        "target_day_local": target_day.isoformat(),
        "generated_at_local": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "datasets": {
            "critic_rows_total": sum(1 for _ in _iter_jsonl(critic_file)),
            "critic_rows_day": _count_rows_in_day(critic_file, start_utc, end_utc),
            "ml_rows_total": sum(1 for _ in _iter_jsonl(ml_file)),
            "ml_rows_day": _count_rows_in_day(ml_file, start_utc, end_utc),
        },
        "worker_status": _load_json(rl_status_file),
        "train_report": _load_json(train_report_file),
        "shadow_report": _load_json(shadow_report_file),
        "ranker_shadow": {
            "events_total": len(ranker_shadow),
            "bot_take_ranker_skip": len(bot_take_ranker_skip),
            "bot_blocked_ranker_take": len(bot_blocked_ranker_take),
            "top_symbols": symbol_counts.most_common(10),
            "top_modes": mode_counts.most_common(10),
            "blocked_reason_counts": blocked_counts.most_common(10),
            "worst_bot_takes": worst_take_by_proba,
            "missed_bot_blocks": missed_by_proba,
        },
    }
    return report


def render_text(report: Dict[str, Any]) -> str:
    ds = report["datasets"]
    rs = report["ranker_shadow"]
    train = report.get("train_report", {})
    lines = [
        f"RL daily report for {report['target_day_local']}",
        f"Generated: {report['generated_at_local']}",
        "",
        "Datasets:",
        f"  critic_dataset: +{ds['critic_rows_day']} rows today, total {ds['critic_rows_total']}",
        f"  ml_dataset: +{ds['ml_rows_day']} rows today, total {ds['ml_rows_total']}",
        "",
        "Training:",
        f"  chosen_model: {train.get('chosen_model', '') or 'n/a'}",
        f"  train_rows: {train.get('train_rows', 0)} val_rows: {train.get('val_rows', 0)} test_rows: {train.get('test_rows', 0)}",
        f"  test_ret5_delta: {train.get('improvement_delta', {}).get('ret5_avg_delta', 'n/a')}",
        f"  test_win_rate_delta: {train.get('improvement_delta', {}).get('win_rate_delta', 'n/a')}",
        "",
        "Shadow disagreements:",
        f"  total: {rs['events_total']}",
        f"  bot_take vs ranker_skip: {rs['bot_take_ranker_skip']}",
        f"  bot_blocked vs ranker_take: {rs['bot_blocked_ranker_take']}",
    ]
    if rs["top_symbols"]:
        lines.append("  top symbols: " + ", ".join(f"{sym}={cnt}" for sym, cnt in rs["top_symbols"]))
    if rs["blocked_reason_counts"]:
        lines.append("  blocked reasons: " + ", ".join(f"{reason}={cnt}" for reason, cnt in rs["blocked_reason_counts"]))

    if rs["worst_bot_takes"]:
        lines.append("")
        lines.append("Worst bot takes (ranker wanted skip):")
        for rec in rs["worst_bot_takes"][:5]:
            lines.append(
                "  "
                + f"{rec.get('sym')} {rec.get('tf')} {rec.get('mode')} "
                + f"score={rec.get('candidate_score')} proba={rec.get('ranker_proba')} reason={rec.get('reason')}"
            )

    if rs["missed_bot_blocks"]:
        lines.append("")
        lines.append("Missed bot blocks (ranker wanted take):")
        for rec in rs["missed_bot_blocks"][:5]:
            lines.append(
                "  "
                + f"{rec.get('sym')} {rec.get('tf')} {rec.get('mode')} "
                + f"score={rec.get('candidate_score')} proba={rec.get('ranker_proba')} reason={rec.get('reason')}"
            )
    return "\n".join(lines)


def build_top_gainer_daily_report(target_day: date) -> Dict[str, Any]:
    """
    Daily top-gainer performance report for target_day.

    Metrics:
      - precision:      of entries on target_day, what % hit top5/10/20
      - capture:        of top5/10/20 on target_day, what % we signaled
      - early_capture:  signals before 10:00 UTC
      - lead_time:      hours before EOD for correct top20 signals
      - missed:         top20 coins in watchlist we didn't signal
      - ml_zone_filter: entries blocked by ml_proba zone today, quality
      - rolling_7d:     precision / capture trend over last 7 days
      - regime:         regime distribution for today's entries
    """
    # ── Load raw data ────────────────────────────────────────────────────────
    events: List[Dict] = list(_iter_jsonl(BOT_EVENTS_FILE))
    tg_records: List[Dict] = list(_iter_jsonl(TOP_GAINER_FILE))

    # ── Build top-gainer label lookup: day → {sym → {tier: bool, eod_return}} ─
    tg_by_day: Dict[date, Dict[str, Dict]] = defaultdict(dict)
    for rec in tg_records:
        ts_ms = rec.get("ts", 0)
        if not ts_ms:
            continue
        d = datetime.utcfromtimestamp(ts_ms / 1000).date()
        sym = rec.get("symbol", "")
        tg_by_day[d][sym] = {
            "top5":  bool(rec.get("label_top5", 0)),
            "top10": bool(rec.get("label_top10", 0)),
            "top20": bool(rec.get("label_top20", 0)),
            "eod_return": float(rec.get("eod_return_pct", 0.0)),
        }

    # ── Parse entry events: extract UTC datetime + metadata ─────────────────
    def _parse_entry(e: Dict) -> Dict | None:
        ts = _parse_utc_ts(e.get("ts"))
        if ts is None:
            return None
        return {
            "sym": e.get("sym", ""),
            "tf": e.get("tf", ""),
            "mode": e.get("mode", ""),
            "ts": ts,
            "hour_utc": ts.hour,
            "day": ts.date(),
            "ml_proba": e.get("ml_proba"),
        }

    entries_today = [
        _parse_entry(e) for e in events
        if e.get("event") == "entry" and _parse_utc_ts(e.get("ts")) is not None
        and _parse_utc_ts(e.get("ts")).date() == target_day  # type: ignore[union-attr]
    ]
    entries_today = [e for e in entries_today if e]

    # ── Blocked by ml_proba_zone today ──────────────────────────────────────
    ml_zone_blocked_today = [
        e for e in events
        if e.get("event") == "blocked"
        and e.get("signal_type") == "ml_proba_zone"
        and _parse_utc_ts(e.get("ts")) is not None
        and _parse_utc_ts(e.get("ts")).date() == target_day  # type: ignore[union-attr]
    ]
    # How many of blocked would have hit top20
    ml_zone_would_hit = 0
    if tg_by_day.get(target_day):
        for e in ml_zone_blocked_today:
            sym = e.get("sym", "")
            info = tg_by_day[target_day].get(sym, {})
            if info.get("top20"):
                ml_zone_would_hit += 1

    # ── Precision & capture for target_day ──────────────────────────────────
    def _day_metrics(day: date, entries: List[Dict]) -> Dict[str, Any]:
        tg = tg_by_day.get(day, {})
        if not tg:
            return {}
        watchlist = set(tg.keys())
        top5_set  = {s for s, v in tg.items() if v["top5"]}
        top10_set = {s for s, v in tg.items() if v["top10"]}
        top20_set = {s for s, v in tg.items() if v["top20"]}

        signaled = {e["sym"] for e in entries if e["day"] == day}
        early    = {e["sym"] for e in entries if e["day"] == day and e["hour_utc"] < 10}

        def _prec(tier_set):
            hits = sum(1 for s in signaled if s in tier_set)
            return hits / len(signaled) if signaled else 0.0

        def _cap(tier_set):
            in_wl = tier_set & watchlist
            hits = signaled & in_wl
            return len(hits) / len(in_wl) if in_wl else 0.0

        def _early_cap(tier_set):
            in_wl = tier_set & watchlist
            hits = early & in_wl
            return len(hits) / len(in_wl) if in_wl else 0.0

        # Lead time for correct top20 signals (hours before midnight UTC)
        lead_times = []
        for e in entries:
            if e["day"] == day and e["sym"] in top20_set:
                lead_times.append(24 - e["hour_utc"])

        # Missed top20 in watchlist
        missed = []
        for sym in (top20_set & watchlist) - signaled:
            missed.append({"sym": sym, "eod_return": tg[sym]["eod_return"]})
        missed.sort(key=lambda x: x["eod_return"], reverse=True)

        return {
            "n_entries": len(signaled),
            "n_entries_early": len(early),
            "precision_top5":  round(_prec(top5_set), 4),
            "precision_top10": round(_prec(top10_set), 4),
            "precision_top20": round(_prec(top20_set), 4),
            "capture_top5":    round(_cap(top5_set), 4),
            "capture_top10":   round(_cap(top10_set), 4),
            "capture_top20":   round(_cap(top20_set), 4),
            "early_capture_top5":  round(_early_cap(top5_set), 4),
            "early_capture_top20": round(_early_cap(top20_set), 4),
            "avg_lead_time_h": round(sum(lead_times) / len(lead_times), 1) if lead_times else 0.0,
            "early_signal_pct": round(sum(1 for h in lead_times if h >= 14) / len(lead_times), 4) if lead_times else 0.0,
            "n_top20_watchlist": len(top20_set & watchlist),
            "n_top5_watchlist":  len(top5_set & watchlist),
            "missed_top20": missed[:5],
        }

    today_metrics = _day_metrics(
        target_day,
        [e for e in [_parse_entry(ev) for ev in events if ev.get("event") == "entry"] if e],
    )

    # ── Rolling 7-day ────────────────────────────────────────────────────────
    all_entries = [e for e in [_parse_entry(ev) for ev in events if ev.get("event") == "entry"] if e]
    rolling: List[Dict] = []
    for delta in range(7):
        d = target_day - timedelta(days=delta)
        m = _day_metrics(d, all_entries)
        if m:
            m["day"] = d.isoformat()
            rolling.append(m)

    def _avg(key: str) -> float:
        vals = [r[key] for r in rolling if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    rolling_7d = {
        "days": len(rolling),
        "avg_precision_top20": _avg("precision_top20"),
        "avg_capture_top20":   _avg("capture_top20"),
        "avg_early_capture_top20": _avg("early_capture_top20"),
        "avg_lead_time_h":     _avg("avg_lead_time_h"),
        "trend": rolling,
    }

    # ── ML zone filter stats (today) ─────────────────────────────────────────
    ml_zone = {
        "blocked_today": len(ml_zone_blocked_today),
        "would_hit_top20": ml_zone_would_hit,
        "block_quality_pct": round(
            ml_zone_would_hit / len(ml_zone_blocked_today) * 100, 1
        ) if ml_zone_blocked_today else 0.0,
    }

    # ── Mode distribution for today's entries ───────────────────────────────
    mode_counts = Counter(e["mode"] for e in entries_today)

    return {
        "target_day": target_day.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "today": today_metrics,
        "ml_zone_filter": ml_zone,
        "rolling_7d": rolling_7d,
        "mode_counts": dict(mode_counts.most_common()),
        "has_tg_data": bool(tg_by_day.get(target_day)),
    }


def render_top_gainer_text(report: Dict[str, Any]) -> str:
    """Format top-gainer daily report for Telegram."""
    t = report.get("today", {})
    ml = report.get("ml_zone_filter", {})
    r7 = report.get("rolling_7d", {})
    day = report.get("target_day", "")

    lines = [f"📊 Top Gainer Report — {day}"]

    if not report.get("has_tg_data") or not t:
        lines.append("⚠️ No top gainer label data for this day yet")
        lines.append("(labels are collected at 00:05 UTC next day)")
        return "\n".join(lines)

    # ── Capture & precision ──────────────────────────────────────────────────
    lines.append("")
    lines.append("capture (signaled / in watchlist):")
    lines.append(
        f"  top5:  {t.get('capture_top5', 0)*100:.0f}%"
        f"   top20: {t.get('capture_top20', 0)*100:.0f}%"
    )
    lines.append(
        f"  early(<10UTC): top5={t.get('early_capture_top5', 0)*100:.0f}%"
        f"  top20={t.get('early_capture_top20', 0)*100:.0f}%"
    )
    lines.append("")
    lines.append("precision (of our entries):")
    lines.append(
        f"  top5={t.get('precision_top5', 0)*100:.0f}%"
        f"  top10={t.get('precision_top10', 0)*100:.0f}%"
        f"  top20={t.get('precision_top20', 0)*100:.0f}%"
        f"  (base: 5/10/20%)"
    )

    # ── Lead time ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(
        f"lead_time: avg={t.get('avg_lead_time_h', 0):.1f}h"
        f"  early_signals={t.get('early_signal_pct', 0)*100:.0f}%"
        f"  entries={t.get('n_entries', 0)}"
    )

    # ── Missed ───────────────────────────────────────────────────────────────
    missed = t.get("missed_top20", [])
    if missed:
        lines.append("")
        lines.append(f"missed top20 ({t.get('n_top20_watchlist', 0)} in watchlist):")
        for m in missed[:5]:
            lines.append(f"  {m['sym']:<14} eod={m['eod_return']:+.1f}%")

    # ── ML zone filter ───────────────────────────────────────────────────────
    if ml.get("blocked_today", 0) > 0:
        lines.append("")
        lines.append(
            f"ml_zone_filter: blocked={ml['blocked_today']}"
            f"  would_hit_top20={ml['would_hit_top20']}"
            f"  ({ml['block_quality_pct']:.0f}% good blocks)"
        )

    # ── Rolling 7d ───────────────────────────────────────────────────────────
    if r7.get("days", 0) >= 3:
        lines.append("")
        lines.append(f"rolling {r7['days']}d avg:")
        lines.append(
            f"  capture_top20={r7['avg_capture_top20']*100:.0f}%"
            f"  precision_top20={r7['avg_precision_top20']*100:.0f}%"
        )
        lines.append(
            f"  early_capture={r7['avg_early_capture_top20']*100:.0f}%"
            f"  lead_time={r7['avg_lead_time_h']:.1f}h"
        )

        # Trend: today vs 7d avg
        today_cap = t.get("capture_top20", 0)
        avg_cap   = r7["avg_capture_top20"]
        delta = today_cap - avg_cap
        arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "→")
        lines.append(f"  trend: capture {arrow} ({delta*100:+.0f}% vs 7d avg)")

    # ── Mode ─────────────────────────────────────────────────────────────────
    modes = report.get("mode_counts", {})
    if modes:
        lines.append("")
        mode_str = "  ".join(f"{m}={c}" for m, c in list(modes.items())[:5])
        lines.append(f"modes: {mode_str}")

    return "\n".join(lines)


def build_bandit_accuracy_section() -> str:
    """Build bandit prediction accuracy section for the report."""
    try:
        from offline_rl import evaluate_bandit_accuracy
        ba = evaluate_bandit_accuracy(n_recent_days=7)
    except Exception as e:
        return f"Bandit accuracy: error ({e})"

    if ba.get("status") != "ok":
        return f"Bandit accuracy: {ba.get('status', 'unknown')}"

    lines = ["", "Bandit Prediction Accuracy:"]
    lines.append(
        f"  recall@20: {ba['overall_recall_top20']*100:.1f}%"
        f" ({ba['total_top20_enter']}/{ba['total_top20']})"
    )
    lines.append(
        f"  UCB separation: {ba['ucb_separation']:+.4f}"
        f" (top={ba['avg_ucb_gap_top_gainers']:+.4f}"
        f"  rest={ba['avg_ucb_gap_non_top']:+.4f})"
    )

    daily = ba.get("daily", [])
    if daily:
        lines.append("  Per day:")
        for d in daily[:7]:
            lines.append(
                f"    {d['day']}: recall={d['recall_top20']*100:.0f}%"
                f" ({d['n_top20_enter']}/{d['n_top20']})"
            )

    # Learning progress trend
    progress_path = WORKSPACE_ROOT / ".runtime" / "learning_progress.jsonl"
    if progress_path.exists():
        recent = []
        for line in progress_path.read_text(encoding="utf-8").strip().splitlines()[-7:]:
            try:
                recent.append(json.loads(line))
            except Exception:
                pass
        if len(recent) >= 2:
            lines.append("")
            lines.append("  Learning trend:")
            for r in recent[-5:]:
                recall = r.get("bandit_recall_top20")
                sep = r.get("bandit_ucb_separation")
                ts = r.get("ts", "")[:10]
                recall_s = f"{recall*100:.0f}%" if recall is not None else "n/a"
                sep_s = f"{sep:+.3f}" if sep is not None else "n/a"
                lines.append(f"    {ts}: recall={recall_s}  sep={sep_s}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a daily RL status report from local datasets and shadow-ranker disagreements.")
    parser.add_argument("--date", default="", help="Local date in YYYY-MM-DD.")
    parser.add_argument("--previous-day", action="store_true", help="Report the previous local day.")
    args = parser.parse_args()

    if args.date:
        target_day = date.fromisoformat(args.date)
    else:
        target_day = datetime.now(LOCAL_TZ).date()
        if args.previous_day:
            target_day = target_day - timedelta(days=1)

    report = build_report(target_day)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"rl_daily_{target_day.isoformat()}.json"
    txt_path = REPORT_DIR / f"rl_daily_{target_day.isoformat()}.txt"

    text = render_text(report)
    # Append bandit accuracy section
    text += "\n" + build_bandit_accuracy_section()

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")
    print(text)
    print("")
    print(f"JSON report saved to: {json_path}")
    print(f"Text report saved to: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
