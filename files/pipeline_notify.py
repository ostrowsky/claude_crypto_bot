"""Telegram delivery for pipeline daily reports.

Reads the freshest L1 health.tg.txt (HTML-formatted, ready to send), appends
a brief Attribution block from the latest pipeline_attribution run, and posts
to every chat_id in `.chat_ids`. Uses `.runtime/tg_send_dedup.json` to avoid
duplicate sends on the same day (matches the existing dedup discipline of
daily_learning.py / RL train notifications).

Designed to be:
  - Self-contained: only stdlib (urllib), so pyembed Python is sufficient.
  - Injectable: `http_post` and `now_iso` parameters allow tests to drive the
    full flow without touching the real Telegram API or system clock.
  - Quiet by default: if there's no token, no chat_ids, or no health report,
    the script logs the reason and returns gracefully — never raises.

CLI:
    pyembed\\python.exe files\\pipeline_notify.py                 # send today
    pyembed\\python.exe files\\pipeline_notify.py --dry-run        # build, don't POST
    pyembed\\python.exe files\\pipeline_notify.py --force          # bypass dedup
    pyembed\\python.exe files\\pipeline_notify.py --date 2026-05-12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pipeline_lib as PL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_IDS_FILE   = PL.FILES_DIR / ".chat_ids"
DEDUP_FILE      = PL.RUNTIME / "tg_send_dedup.json"
TELEGRAM_API    = "https://api.telegram.org"
TG_MAX_CHARS    = 4000      # Telegram limit is 4096; keep a margin
DEFAULT_TIMEOUT = 20

# Per-mode reports written by _weekly_signal_eval_with_tg.py / runsignalevaluator.
# Used to build the "Incidents" block. Keep schema-only here (no telegram
# concerns leak back into the evaluator).
EVAL_PER_MODE_DIR = PL.REPO_ROOT / "evaluation_output" / "per_mode"


# ---------------------------------------------------------------------------
# Config readers — all defensive, all return None / [] on failure
# ---------------------------------------------------------------------------


def get_telegram_token() -> str | None:
    """Token resolution: env var (matches files/config.py) > runtime cmd file.

    Never raises. Returns None if not configured."""
    tok = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if tok:
        return tok
    # Fallback: parse runtime cmd if present (bot_bg_runner sets it there)
    runner = PL.RUNTIME / "bot_bg_runner.cmd"
    if runner.exists():
        try:
            for line in runner.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                line = line.strip()
                if line.lower().startswith("set telegram_bot_token="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
        except OSError:
            return None
    return None


def load_chat_ids(path: Path | None = None) -> list[int]:
    """Read .chat_ids. Accepts both BOM-prefixed UTF-8 (PowerShell default)
    and plain JSON. Returns sorted unique ints. [] on any failure.

    `path=None` (the default) resolves to the module-level CHAT_IDS_FILE at
    CALL time, not at def time. That matters: tests patch the module
    attribute, and a captured default would defeat them."""
    if path is None:
        path = CHAT_IDS_FILE
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    out: list[int] = []
    for item in payload:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def read_health_tg(target_date: date) -> str | None:
    p = PL.HEALTH / f"health-{target_date.isoformat()}.tg.txt"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Attribution block
# ---------------------------------------------------------------------------


def _latest_attribution_report() -> dict | None:
    """Pick the most recent attribution-*.json (regardless of date) — these
    are append-only per orchestrator run, so 'latest' is correct."""
    d = PL.PIPELINE / "attribution"
    if not d.exists():
        return None
    candidates = sorted(d.glob("attribution-*.json"))
    if not candidates:
        return None
    return PL.read_json(candidates[-1])


def build_attribution_block(report: dict | None = None) -> str | None:
    """Render the Attribution block as Telegram-HTML. Returns None if there's
    nothing meaningful to show (no report, or all decisions are needs_data)."""
    if report is None:
        report = _latest_attribution_report()
    if not report:
        return None
    meta = report.get("pipeline_meta") or {}
    n_eval = int(meta.get("n_evaluated") or 0)
    if n_eval <= 0:
        return None
    by_verdict = meta.get("by_verdict") or {}
    # Skip the block entirely when every decision is needs_data — nothing
    # actionable yet for the operator.
    if set(by_verdict.keys()) <= {"needs_data", "skip", "no_baseline"}:
        return None

    hit_rate = meta.get("hit_rate")
    hits = int(by_verdict.get("hit", 0))
    partial = int(by_verdict.get("partial", 0))
    miss = int(by_verdict.get("miss", 0))
    regr = int(by_verdict.get("regression", 0))
    pending = int(by_verdict.get("needs_data", 0)) + int(by_verdict.get("no_baseline", 0))

    hit_rate_str = "n/a" if hit_rate is None else f"{hit_rate:.2f}"

    lines = [
        "📈 <b>Pipeline Attribution</b>",
        f"   hit_rate: <b>{hit_rate_str}</b> "
        f"(✅{hits}  ◐{partial}  ❌{miss}  🚨{regr}  ⏳{pending})",
    ]
    if regr:
        lines.append(f"   ⚠️ {regr} regression(s) — check attribution.jsonl")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rollback recommendations block — surfaces L7's "this didn't work" verdicts
# ---------------------------------------------------------------------------
#
# L7 monitor writes `.runtime/pipeline/monitor/rollback_recommendations.jsonl`
# whenever an approved decision aged ≥ check-after-days fails to hit its
# expected delta. We render those into the daily Telegram so the operator
# can act on them — copy-paste a rollback command.
#
# This is the SURFACING half of "auto-apply rollback". Actually editing
# config.py from the daemon is risky on a live trading bot, so this MVP
# stops at presentation. If you later want to auto-execute, add a config
# flag AUTO_ROLLBACK_ENABLED and wire it here.


ROLLBACK_LOG_PATH = PL.PIPELINE / "monitor" / "rollback_recommendations.jsonl"


def _rolled_back_decision_ids() -> set[str]:
    """Set of decision_ids for which a 'rolled_back' record exists.

    The rollback log appends one record every time L7 detects a miss; we
    must de-dup so a decision that's already been rolled back doesn't keep
    showing up in Telegram forever."""
    out: set[str] = set()
    for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
        if rec.get("stage") == "rolled_back":
            original = rec.get("rolling_back") or rec.get("decision_id")
            if original:
                out.add(original)
    return out


def collect_rollback_recommendations(*, max_items: int = 3) -> list[dict]:
    """Pending L7 recommendations, newest first, de-duplicated."""
    if not ROLLBACK_LOG_PATH.exists():
        return []
    already = _rolled_back_decision_ids()
    seen: set[str] = set()
    out: list[dict] = []
    # Newest first
    recs = list(PL.iter_jsonl(ROLLBACK_LOG_PATH))
    for rec in reversed(recs):
        did = rec.get("decision_id")
        if not did or did in already or did in seen:
            continue
        seen.add(did)
        out.append(rec)
        if len(out) >= max_items:
            break
    return out


def build_rollback_block(*, max_items: int = 3) -> str | None:
    items = collect_rollback_recommendations(max_items=max_items)
    if not items:
        return None
    lines = [f"🔄 <b>Pipeline предлагает откатить</b> ({len(items)}):"]
    for it in items:
        hid = it.get("hypothesis_id") or "?"
        rule = it.get("rule") or "?"
        key = it.get("config_key") or ""
        reason = ((it.get("outcome") or {}).get("reason")
                  or (it.get("outcome") or {}).get("verdict")
                  or "verdict=miss")
        # Truncate reason — full details in attribution.jsonl
        if len(reason) > 100:
            reason = reason[:97] + "…"
        lines.append("")
        lines.append(f"  <b>{rule}</b>  <i>({key})</i>")
        lines.append(f"  <i>{reason}</i>")
        lines.append(f"  ▶️ <code>pyembed\\python.exe files\\pipeline_approve.py "
                     f"--rollback {it.get('decision_id')} --reason \"auto: L7 miss\"</code>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Incidents block — premature exits, missed sustained trends, losing trades
# ---------------------------------------------------------------------------
#
# Reads the per-mode signal-evaluator reports (one JSON per entry mode).
# Folds three categories of incident into a single Telegram block so the
# operator gets them inside the unified daily report instead of as a
# separate "Skill daily check" message.
#
# Thresholds mirror what `_weekly_signal_eval_with_tg.py::_build_daily_digest`
# used so the operator sees the same items, just inline now.


# Empirical thresholds (matched to the standalone digest so operators see
# identical incident lists). Surface a few constants in case we want to tune.
INCIDENT_PREMATURE_CAPTURE_MAX = 0.30   # capture_ratio < this → premature
INCIDENT_PREMATURE_BARS_EARLY  = 3      # sold ≥ this many bars before peak
INCIDENT_LOSING_PNL_MAX_PCT    = -1.5   # captured_pnl_pct below this → loser
INCIDENT_MISSED_GAIN_MIN_PCT   = 5.0    # gain ≥ this → notable miss
INCIDENT_REPORT_FRESH_HOURS    = 36     # ignore reports older than this


def _load_per_mode_reports(per_mode_dir: Path | None = None) -> dict[str, dict]:
    """Return {mode_name: report_dict} for every mode subdir with a valid
    report.json. Skips silently when nothing is there (signal evaluator
    hasn't run yet)."""
    base = per_mode_dir if per_mode_dir is not None else EVAL_PER_MODE_DIR
    out: dict[str, dict] = {}
    if not base.exists():
        return out
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        rj = sub / "report.json"
        if not rj.exists():
            continue
        data = PL.read_json(rj)
        if isinstance(data, dict):
            out[sub.name] = data
    return out


def _report_age_hours(report: dict, *, now: datetime | None = None) -> float | None:
    ts = report.get("generated_at")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    n = now if now is not None else datetime.now(timezone.utc)
    return (n - dt).total_seconds() / 3600.0


def _collect_incidents(reports: dict[str, dict]) -> dict[str, list]:
    """Apply the same filters _weekly_signal_eval_with_tg.py uses so the
    operator sees the same list. De-dups missed opportunities across modes
    (same coin shows up in each per-mode report)."""
    premature, losers = [], []
    for mode, r in reports.items():
        for v in (r.get("trade_verdicts") or []):
            v = dict(v)
            v["_mode"] = mode
            bars_early = v.get("sell_lateness_bars")
            pnl        = v.get("captured_pnl_pct") or 0.0
            capture    = v.get("capture_ratio") or 1.0
            if (bars_early is not None and bars_early < -INCIDENT_PREMATURE_BARS_EARLY
                and pnl > 0 and capture < INCIDENT_PREMATURE_CAPTURE_MAX):
                premature.append(v)
            if pnl < INCIDENT_LOSING_PNL_MAX_PCT:
                losers.append(v)

    big_misses, seen = [], set()
    for mode, r in reports.items():
        for t in (r.get("missed_opportunities") or []):
            if (t.get("gain_pct") or 0) < INCIDENT_MISSED_GAIN_MIN_PCT:
                continue
            key = (t.get("symbol"), str(t.get("true_start_ts")))
            if key in seen:
                continue
            seen.add(key)
            big_misses.append({**t, "_mode": mode})

    return {"premature": premature, "losers": losers, "misses": big_misses}


def build_incidents_block(per_mode_dir: Path | None = None,
                          *, max_per_section: int = 3,
                          now: datetime | None = None) -> str | None:
    """Render the incidents block, or None when:
      - no per-mode reports exist yet
      - reports are all too old (signal evaluator hasn't run today)
      - there are no incidents at all
    """
    reports = _load_per_mode_reports(per_mode_dir)
    if not reports:
        return None
    fresh_enough = [r for r in reports.values()
                    if (_report_age_hours(r, now=now) or 1e9) <= INCIDENT_REPORT_FRESH_HOURS]
    if not fresh_enough:
        return None

    inc = _collect_incidents(reports)
    if not (inc["premature"] or inc["losers"] or inc["misses"]):
        return None

    lines = ["⚠️ <b>Инциденты за 24ч:</b>"]

    if inc["premature"]:
        lines.append(f"  ✂️ Premature exits ({len(inc['premature'])}):")
        for v in sorted(inc["premature"], key=lambda x: x.get("capture_ratio") or 0)[:max_per_section]:
            cr = (v.get("capture_ratio") or 0) * 100
            bars = abs(v.get("sell_lateness_bars", 0))
            lines.append(
                f"    · {v.get('symbol')} [{v.get('_mode')}] capture {cr:.0f}%, "
                f"sold {bars}b before peak"
            )

    if inc["misses"]:
        lines.append(f"  🚫 Missed sustained trends ({len(inc['misses'])}):")
        for t in sorted(inc["misses"], key=lambda x: -(x.get("gain_pct") or 0))[:max_per_section]:
            lines.append(
                f"    · {t.get('symbol')} [{t.get('_mode')}] gain {t.get('gain_pct'):+.1f}% — missed entry"
            )

    if inc["losers"]:
        lines.append(f"  🔴 Losing trades ({len(inc['losers'])}):")
        for v in sorted(inc["losers"], key=lambda x: x.get("captured_pnl_pct") or 0)[:max_per_section]:
            lines.append(
                f"    · {v.get('symbol')} [{v.get('_mode')}] P&L {v.get('captured_pnl_pct', 0):+.2f}%"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hypothesis review block — "you have N items to approve"
# ---------------------------------------------------------------------------


SEVERITY_ORDER = {"critical": 3, "red": 2, "yellow": 1}
DEFAULT_REVIEW_TOP_N = 3


def _latest_advisor(hypothesis_id: str) -> dict | None:
    """Read approve_advisory.jsonl and return the latest entry for this hyp."""
    advisory_log = PL.PIPELINE / "approve_advisory.jsonl"
    if not advisory_log.exists():
        return None
    latest = None
    for rec in PL.iter_jsonl(advisory_log):
        if rec.get("hypothesis_id") == hypothesis_id:
            latest = rec
    return latest


def _ready_for_review(hyp: dict, locked_keys: set[str]) -> bool:
    """Mirror of pipeline_approve.safety_checks PLUS some softer signals:
    only surface hypotheses where the operator can actually act."""
    if hyp.get("status") != "pending_validation":
        return False
    if hyp.get("config_key") in locked_keys:
        return False
    vr = (hyp.get("validation_report") or {}).get("result") or {}
    if vr.get("verdict") == "reject":
        return False
    # We allow pending_manual_validation through: many Claude-generated
    # hypotheses don't have a registered L3 validator yet. The operator
    # decides — but they need to see the option.
    return True


def collect_review_candidates(top_n: int = DEFAULT_REVIEW_TOP_N) -> list[dict]:
    """Return up to top_n hypotheses ready for review, sorted by severity
    then by persistence (days_red), most actionable first."""
    locked = set((PL.load_do_not_touch() or {}).get("config_keys_locked", []))
    out: list[dict] = []
    for p in sorted(PL.HYPOTHESES.glob("h-*.json")):
        h = PL.read_json(p) or {}
        if not _ready_for_review(h, locked):
            continue
        out.append(h)

    def sort_key(h: dict):
        sev = SEVERITY_ORDER.get(h.get("severity", "yellow"), 0)
        days = (h.get("persistence") or {}).get("days_red") or 0
        # higher first
        return (-sev, -days, h.get("hypothesis_id", ""))

    out.sort(key=sort_key)
    return out[:top_n]


# ----- Plain-language helpers ---------------------------------------------


def humanize_rule(hyp: dict) -> str:
    """Translate a rule_id + diff into a one-line Russian description that a
    non-technical operator can read without opening the JSON file."""
    rule = hyp.get("rule", "")
    diff = hyp.get("diff", {}) or {}
    frm, to = diff.get("from"), diff.get("to")

    if rule.startswith("disable_mode_"):
        mode = rule[len("disable_mode_"):]
        return f"Отключить режим входа «{mode}»"
    if rule == "widen_watchlist_match_tolerance":
        return f"Расширить score-фильтр watchlist ({frm} → {to})"
    if rule.startswith("relax_gate_"):
        gate = rule[len("relax_gate_"):]
        return f"Ослабить фильтр «{gate}» (+10%)"
    if rule.startswith("tighten_proba_"):
        mode = rule[len("tighten_proba_"):]
        return f"Поднять порог уверенности модели для «{mode}»"
    if rule == "entry_score_floor_relax":
        return f"Опустить минимальный entry score ({frm} → {to})"
    # Fallback — at least show the diff if we have it
    if frm is not None and to is not None:
        return f"{rule}: {frm} → {to}"
    return rule


# Hand-curated, per-rule caveat shown alongside the backtest numbers when the
# counterfactual is known to overstate the effect. Keep them short.
SIM_CAVEATS: dict[str, str] = {
    "widen_watchlist_match_tolerance":
        "Бэктест — upper-bound оценка: реальный эффект меньше, "
        "т.к. tolerance расширяет окно, а не отменяет фильтр.",
}


def _delta_emoji(delta: float | None, *, lower_is_better: bool = False,
                 noise_threshold: float = 0.005) -> str:
    """✅ for good direction, ❌ for bad, ◐ for no significant change.

    `delta` is in PERCENTAGE POINTS (the same scale as the inputs). A change
    smaller than 0.005pp (0.5 basis points) is treated as noise — the
    sample-size-dependent sampling error of median PnL on real shadow data
    is typically larger than that."""
    if delta is None or abs(delta) < noise_threshold:
        return "◐"
    is_positive = delta > 0
    good = (is_positive and not lower_is_better) or (not is_positive and lower_is_better)
    return "✅" if good else "❌"


def _fmt_pct(v: float | None) -> str:
    """Format a percentage value. Inputs are ALREADY in percent (because
    `labels.ret_5` in the critic dataset is the per-trade % return, not a
    fraction). Just append '%' — never multiply."""
    if v is None:
        return "—"
    return f"{v:+.2f}%"


def _extract_pnl_triple(stats: dict) -> tuple[float | None, float | None, float | None]:
    return (
        stats.get("prod_median_pnl_pct"),
        stats.get("shadow_median_pnl_pct"),
        stats.get("delta_median_pnl_pct"),
    )


def _measured_summary(hyp: dict) -> dict | None:
    """Pull measured numbers out of shadow_report and derive plain-language
    metric changes. Returns None when no measurement is attached.

    Output includes a `recent` sub-block when the shadow_report contains a
    recency split — operator wants to see if the historical effect still
    holds in the most recent window (fixes may have already addressed it)."""
    sr = hyp.get("shadow_report") or {}
    by_feat = ((sr.get("summary") or {}).get("by_feature") or {})
    if not by_feat:
        return None
    flag, stats = next(iter(by_feat.items()))
    window = sr.get("window_days") or 60
    n = int(stats.get("n_events") or 0)
    pnl_before, pnl_after, pnl_delta = _extract_pnl_triple(stats)

    rule = hyp.get("rule", "")
    trades_before, trades_after = None, None
    if rule.startswith("disable_mode_"):
        trades_before, trades_after = n, 0
    elif rule == "widen_watchlist_match_tolerance":
        trades_before, trades_after = 0, n

    out = {
        "feature_flag":   flag,
        "window_days":    window,
        "n_events":       n,
        "pnl_before":     pnl_before,
        "pnl_after":      pnl_after,
        "pnl_delta":      pnl_delta,
        "trades_before":  trades_before,
        "trades_after":   trades_after,
        "context":        sr.get("context") or {},
    }

    rec = sr.get("recency") or {}
    rec_by_feat = ((rec.get("summary") or {}).get("by_feature") or {})
    if rec_by_feat:
        rflag, rstats = next(iter(rec_by_feat.items()))
        rb, ra, rd = _extract_pnl_triple(rstats)
        out["recent"] = {
            "recent_days":  rec.get("recent_days") or (window // 2),
            "n_events":     int(rstats.get("n_events") or 0),
            "pnl_before":   rb,
            "pnl_after":    ra,
            "pnl_delta":    rd,
        }
    return out


def _volume_context_str(m: dict, days: int) -> str | None:
    """Render '−31% от всех сигналов' or similar. Returns None when we don't
    have the denominator (e.g. for non-simulated hypotheses)."""
    tb, ta = m.get("trades_before"), m.get("trades_after")
    if tb is None or ta is None or days <= 0:
        return None
    ctx = m.get("context") or {}
    total = int(ctx.get("total_takes_in_window") or 0)
    delta_per_day = (ta - tb) / days
    if total <= 0:
        # no denominator; just /day
        if abs(delta_per_day) < 0.5:
            return None
        return f"{delta_per_day:+.0f} трейдов/день"
    total_per_day = total / days
    pct = (delta_per_day / total_per_day) * 100.0 if total_per_day else 0.0
    sign = "+" if delta_per_day > 0 else "−"
    return (f"{sign}{abs(delta_per_day):.0f} трейдов/день "
            f"({pct:+.0f}% от всех сигналов бота)")


def _recency_drift_label(full_delta: float | None, recent_delta: float | None) -> str | None:
    """Plain-Russian label for how the recent half compares to the full window."""
    if full_delta is None or recent_delta is None:
        return None
    if abs(full_delta) < 0.05 and abs(recent_delta) < 0.05:
        return None
    if abs(full_delta) < 0.05:
        return "эффект появился недавно"
    ratio = abs(recent_delta) / abs(full_delta)
    if ratio > 1.5:
        return "эффект усиливается"
    if ratio < 0.5:
        return "эффект ослаб — возможно, недавние фиксы уже сработали"
    if (full_delta > 0) != (recent_delta > 0):
        return "направление сменилось — фиксы могли уже сработать"
    return "эффект стабилен"


def _freshness_label(generated_at: str | None) -> str | None:
    """Return 'обновлено N часов/дней назад' for a shadow_report timestamp.
    None if we can't parse it."""
    if not generated_at:
        return None
    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    age = datetime.now(timezone.utc) - dt
    hours = age.total_seconds() / 3600
    if hours < 1:
        return "только что"
    if hours < 24:
        return f"{int(hours)} ч назад"
    days = int(hours / 24)
    return f"{days} дн назад"


def _format_measurement_block(m: dict, rule: str, hyp: dict) -> list[str]:
    days = m["window_days"]
    n = m["n_events"]
    sr_gen = (hyp.get("shadow_report") or {}).get("generated_at")
    fresh = _freshness_label(sr_gen)
    fresh_tag = f"  <i>(обновлено {fresh})</i>" if fresh else ""
    out = [f"📊 <b>Бэктест:</b>{fresh_tag}"]
    if m["pnl_before"] is not None and m["pnl_after"] is not None:
        delta = m["pnl_delta"]
        emoji = _delta_emoji(delta)
        tail = "нет значимого эффекта" if emoji == "◐" else f"{delta:+.2f}pp"
        out.append(
            f"  • {days} дн ({n} трейдов): PnL/трейд "
            f"<b>{_fmt_pct(m['pnl_before'])}</b> → <b>{_fmt_pct(m['pnl_after'])}</b>   "
            f"{emoji} {tail}"
        )
    rec = m.get("recent")
    if rec and rec.get("pnl_before") is not None and rec.get("pnl_after") is not None:
        rd = rec["pnl_delta"]
        remoji = _delta_emoji(rd)
        rtail = "нет эффекта" if remoji == "◐" else f"{rd:+.2f}pp"
        rd_label = _recency_drift_label(m.get("pnl_delta"), rd)
        rd_suffix = f"  ({rd_label})" if rd_label else ""
        out.append(
            f"  • {rec['recent_days']} дн ({rec['n_events']} трейдов): PnL/трейд "
            f"<b>{_fmt_pct(rec['pnl_before'])}</b> → <b>{_fmt_pct(rec['pnl_after'])}</b>   "
            f"{remoji} {rtail}{rd_suffix}"
        )
    vol = _volume_context_str(m, days)
    if vol:
        out.append(f"  • Объём после apply: {vol}")
    cav = SIM_CAVEATS.get(rule)
    if cav:
        out.append(f"  ⚠️ <i>{cav}</i>")
    return out


# ----- Bottom-line verdict ------------------------------------------------


def _pipeline_verdict(hyp: dict,
                      measured: dict | None,
                      advisor: dict | None) -> tuple[str, str, str]:
    """Compute a deterministic plain-Russian verdict for the review block.

    Returns (emoji, headline, reason). Logic is conservative:
      - ❌ when there's no measurement at all — never apply blind
      - ❌ when the backtest shows a CLEAR regression
      - ⚠️ when the effect is noisy, weak, or based on an upper-bound proxy
      - ⚠️ when the recent window contradicts the full window (fixes
            may already have addressed the issue)
      - ✅ only when measurement is positive AND not undermined by caveats

    We do NOT defer to Claude's recommendation here: it's already shown
    separately. The verdict is for the case where the operator hasn't read
    Claude's reasoning and just wants a one-glance signal."""
    rule = hyp.get("rule", "")
    cav = SIM_CAVEATS.get(rule)

    if measured is None:
        return ("❌", "Не апрувить",
                "Бэктест не запущен — нет данных для решения.")

    delta = measured.get("pnl_delta")
    n     = measured.get("n_events", 0)

    if delta is None or n < 30:
        return ("⚠️", "Подождать",
                f"Недостаточно событий ({n}) для надёжной оценки.")

    if delta < -0.05:
        return ("❌", "Не апрувить",
                f"Бэктест показал ухудшение PnL ({delta:+.2f}pp).")

    if abs(delta) < 0.05:
        # No real signal one way or the other
        if cav:
            return ("⚠️", "Подождать",
                    "Бэктест не показал значимого эффекта; симуляция — "
                    "upper-bound proxy. Нужно better evidence.")
        return ("⚠️", "Подождать",
                "Бэктест не показал значимого эффекта.")

    # delta >= +0.05pp — positive signal
    recent = measured.get("recent") or {}
    rd = recent.get("pnl_delta")
    if rd is not None and abs(delta) >= 0.05:
        # Direction reversal: full says positive, recent says negative
        if (delta > 0) != (rd > 0) and abs(rd) >= 0.05:
            return ("⚠️", "Подождать",
                    "Полное окно показывает плюс, а последние "
                    f"{recent.get('recent_days', '?')} дн — минус. "
                    "Недавние фиксы могли уже исправить проблему.")
        # Effect weakened a lot
        if abs(rd) < abs(delta) * 0.5 and abs(delta) >= 0.10:
            return ("⚠️", "APPROVE с оговоркой",
                    "Минус существует, но эффект ослабевает — фиксы могли "
                    "уже сократить проблему.")

    if cav:
        return ("⚠️", "APPROVE с оговоркой",
                "Бэктест показал плюс, но это upper-bound оценка — "
                "реальный эффект меньше.")

    return ("✅", "APPROVE",
            f"Бэктест показал устойчивый плюс ({delta:+.2f}pp на {n} трейдах).")


def _format_expected_block(hyp: dict) -> list[str]:
    """Render Claude's projected delta — clearly marked as not measured."""
    exp = hyp.get("expected_delta") or {}
    if not exp:
        return []
    out = ["🎯 <b>Прогноз Claude</b> (не измерено бэктестом):"]
    for k, v in exp.items():
        out.append(f"  • {k}: <code>{v}</code>")
    return out


def _format_advisor_line(advisor: dict | None) -> str | None:
    if not advisor:
        return None
    rec = (advisor.get("recommendation") or "").lower()
    conf = advisor.get("confidence") or "?"
    label = {"approve": "✅ APPROVE",
             "reject":  "❌ REJECT",
             "needs_review": "⚠️ NEEDS REVIEW"}.get(rec, f"❓ {rec.upper()}")
    return f"🤖 Claude: {label} ({conf})"


def _format_no_measurement_note(hyp: dict) -> list[str]:
    """For hypotheses without an L4 sim handler — surface what little we know
    from L3 (rationale snippet) so the operator can still judge."""
    out = ["📊 <b>Бэктест:</b> не выполнен (нет sim handler для этого правила)"]
    rat = (hyp.get("rationale") or "").strip()
    if rat:
        rat = rat.replace("\n", " ")
        if len(rat) > 140:
            rat = rat[:137] + "…"
        out.append(f"  <i>{rat}</i>")
    return out


def build_hypothesis_review_block(top_n: int = DEFAULT_REVIEW_TOP_N) -> str | None:
    """Render the operator-facing review block. New format prioritises
    PLAIN-LANGUAGE titles and a measured before/after table — the explicit
    feedback from the operator was "I can't decide from rule IDs and verdicts;
    show me what backtest says metrics become."

    Layout per hypothesis:
        1. Plain-Russian title (humanize_rule)
        2. Severity + generator tag
        3. Measured before/after table from shadow_report (or "no backtest")
        4. Claude's projected delta (clearly marked as projection)
        5. Claude advisor recommendation (if any)
        6. Ready-to-paste approve command
    """
    items = collect_review_candidates(top_n)
    if not items:
        return None

    # Header counts by quick recommendation flavour
    quick: dict[str, int] = {"approve": 0, "needs_review": 0, "reject": 0,
                             "unknown": 0}
    for h in items:
        adv = _latest_advisor(h.get("hypothesis_id", "")) or {}
        rec = (adv.get("recommendation") or "unknown").lower()
        quick[rec if rec in quick else "unknown"] += 1
    summary_parts = []
    if quick["approve"]:
        summary_parts.append(f"✅{quick['approve']}")
    if quick["needs_review"]:
        summary_parts.append(f"⚠️{quick['needs_review']}")
    if quick["reject"]:
        summary_parts.append(f"❌{quick['reject']}")
    if quick["unknown"]:
        summary_parts.append(f"❓{quick['unknown']}")
    header_tail = f"  {' '.join(summary_parts)}" if summary_parts else ""

    lines = [f"📋 <b>Готовы к применению: {len(items)}</b>{header_tail}"]
    for i, hyp in enumerate(items, 1):
        hid  = hyp.get("hypothesis_id", "?")
        gen  = hyp.get("generator", "rule")
        sev  = hyp.get("severity", "?")
        title = humanize_rule(hyp)

        lines.append("")
        lines.append(f"<b>{i}. {title}</b>  <i>[{sev} · {gen}]</i>")

        m = _measured_summary(hyp)
        if m:
            lines.extend(_format_measurement_block(m, hyp.get("rule", ""), hyp))
        else:
            lines.extend(_format_no_measurement_note(hyp))

        # Bottom-line verdict — the answer to "is this good or bad?"
        adv = _latest_advisor(hid)
        v_emoji, v_head, v_reason = _pipeline_verdict(hyp, m, adv)
        lines.append(f"📌 <b>Итог:</b> {v_emoji} <b>{v_head}</b>")
        lines.append(f"   <i>{v_reason}</i>")

        # Recent incident evidence — surfaces "this hypothesis is reinforced
        # by 2 premature exits and 5 losing trades in the last 24h".
        ie = hyp.get("incident_evidence") or {}
        if ie:
            parts = []
            if ie.get("premature"):
                parts.append(f"{ie['premature']} premature exit(s)")
            if ie.get("losers"):
                parts.append(f"{ie['losers']} losing trade(s)")
            if ie.get("missed"):
                parts.append(f"{ie['missed']} missed trend(s)")
            if parts:
                lines.append("📎 Подкреплено за 24ч: " + ", ".join(parts))

        # Claude advisor — optional, one line, after our deterministic verdict
        adv_line = _format_advisor_line(adv)
        if adv_line:
            lines.append(adv_line)

        # Hypothesis target (Claude prediction) — kept terse, single line
        exp = hyp.get("expected_delta") or {}
        if exp:
            headline_metric = next(iter(exp.keys()))
            lines.append(
                f"🎯 Цель: <code>{headline_metric}: {exp[headline_metric]}</code>"
                + (" …" if len(exp) > 1 else "")
            )

        # Full PowerShell-paste-ready command. Without "pyembed\python.exe"
        # the user gets a CommandNotFoundException — .py files aren't directly
        # executable on Windows by default.
        lines.append(
            f"▶️ <code>pyembed\\python.exe files\\pipeline_approve.py --hypothesis {hid}</code>"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------


def build_full_message(
    target_date: date,
    *,
    health_text: str | None = None,
    attribution_report: dict | None = None,
    review_block: str | None | type = ...,
    incidents_block: str | None | type = ...,
    rollback_block: str | None | type = ...,
) -> str | None:
    """Compose the final Telegram payload. Returns None if there's no health
    report (we don't send naked attribution blocks — they have no context).

    `*_block=...` (sentinel) means "compute from disk". Pass an explicit
    str or None to override — useful for tests."""
    if health_text is None:
        health_text = read_health_tg(target_date)
    if not health_text:
        return None
    parts = [health_text.rstrip()]

    if incidents_block is ...:
        incidents_block = build_incidents_block()
    if incidents_block:
        parts.append("")
        parts.append(incidents_block)

    attribution = build_attribution_block(attribution_report)
    if attribution:
        parts.append("")
        parts.append(attribution)

    # Rollback recommendations sit between attribution and review block:
    # they need action TODAY (something previously approved didn't work),
    # which is higher priority than new approval candidates.
    if rollback_block is ...:
        rollback_block = build_rollback_block()
    if rollback_block:
        parts.append("")
        parts.append(rollback_block)

    if review_block is ...:
        review_block = build_hypothesis_review_block()
    if review_block:
        parts.append("")
        parts.append(review_block)

    msg = "\n".join(parts)
    if len(msg) > TG_MAX_CHARS:
        msg = msg[: TG_MAX_CHARS - 50] + "\n…[truncated]"
    return msg


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _load_dedup() -> dict[str, Any]:
    if not DEDUP_FILE.exists():
        return {}
    try:
        return json.loads(DEDUP_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_dedup(state: dict[str, Any]) -> None:
    DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEDUP_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def dedup_key(target_date: date) -> str:
    return f"pipeline_health_{target_date.isoformat()}"


def is_dedup_blocked(target_date: date, state: dict | None = None) -> bool:
    """A key for today already in dedup → block second send.

    Caller can pass `state` for testability (otherwise loaded from disk)."""
    s = state if state is not None else _load_dedup()
    return dedup_key(target_date) in s


def mark_dedup(target_date: date, *, now_iso: str | None = None,
               state: dict | None = None) -> dict:
    s = state if state is not None else _load_dedup()
    s[dedup_key(target_date)] = now_iso or PL.utc_now_iso()
    if state is None:
        _save_dedup(s)
    return s


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


def _real_http_post(url: str, payload: dict, *, timeout: int) -> tuple[int, str]:
    """Default HTTP POST used in production. Returns (status, body)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, body
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"network: {e!r}"


def send_to_chat(
    token: str,
    chat_id: int,
    text: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    http_post: Callable[..., tuple[int, str]] = _real_http_post,
) -> dict:
    """Post one message. Returns {"ok": bool, "status": int, "body": str}."""
    url = f"{TELEGRAM_API}/bot{token}/sendMessage"
    payload = {
        "chat_id":                   chat_id,
        "text":                      text[:TG_MAX_CHARS + 96],   # API limit 4096
        "parse_mode":                "HTML",
        "disable_web_page_preview":  True,
    }
    status, body = http_post(url, payload, timeout=timeout)
    return {"ok": 200 <= status < 300, "status": status, "body": body, "chat_id": chat_id}


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def notify(
    target_date: date | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    http_post: Callable[..., tuple[int, str]] = _real_http_post,
    now_iso: str | None = None,
) -> dict:
    """Run end-to-end. Returns a result dict suitable for logging.

    `dry_run=True` builds the message and skips POST + dedup write.
    `force=True` bypasses the dedup check (but still records on success).
    """
    target_date = target_date or datetime.now(timezone.utc).date()
    result: dict[str, Any] = {
        "date":     target_date.isoformat(),
        "dry_run":  dry_run,
        "force":    force,
        "skipped":  None,
        "sent":     [],
        "errors":   [],
    }

    if not force and is_dedup_blocked(target_date):
        result["skipped"] = "already_sent_today"
        return result

    token = get_telegram_token()
    if not token:
        result["skipped"] = "no_token"
        return result

    chat_ids = load_chat_ids()
    if not chat_ids:
        result["skipped"] = "no_chat_ids"
        return result

    msg = build_full_message(target_date)
    if not msg:
        result["skipped"] = "no_health_report"
        return result

    result["message_chars"] = len(msg)

    if dry_run:
        result["skipped"] = "dry_run"
        result["message_preview"] = msg[:300]
        return result

    any_ok = False
    for cid in chat_ids:
        r = send_to_chat(token, cid, msg, http_post=http_post)
        if r["ok"]:
            any_ok = True
            result["sent"].append(cid)
        else:
            result["errors"].append({"chat_id": cid, "status": r["status"],
                                     "body": r["body"][:300]})

    if any_ok:
        mark_dedup(target_date, now_iso=now_iso)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build message and exit; no POST, no dedup write")
    ap.add_argument("--force", action="store_true",
                    help="ignore dedup and re-send")
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    d = date.fromisoformat(args.date) if args.date else None
    res = notify(d, dry_run=args.dry_run, force=args.force)
    if args.do_print:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        if res.get("skipped"):
            print(f"[notify] skipped: {res['skipped']}")
        elif res.get("sent"):
            print(f"[notify] sent to {len(res['sent'])} chat(s); errors={len(res['errors'])}")
        else:
            print(f"[notify] no chats reached; errors={len(res['errors'])}")


if __name__ == "__main__":
    main()
