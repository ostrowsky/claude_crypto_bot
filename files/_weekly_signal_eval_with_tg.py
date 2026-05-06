"""Signal-evaluator runner with TG digest at three cadences.

Three modes triggered by Windows Scheduled Tasks:
  daily   — last 24h, silent unless incidents (premature exits, big misses)
  weekly  — last 7d (default), full digest + scout auto-apply
  monthly — last 30d, regime / trend / model decision summary

Flow:
  1) Run skill via wrapper (--per-mode, --window-days N).
  2) Parse evaluation_output/per_mode/<mode>/report.json for each mode.
  3) Build cadence-specific TG digest.
  4) Send to TG admin chat(s).
  5) Trigger scout (weekly / monthly only — daily is too noisy).

Spec: docs/specs/features/signal-evaluator-integration-spec.md
"""
from __future__ import annotations
import asyncio, io, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PYEMBED = ROOT / "pyembed" / "python.exe"
WRAPPER = ROOT / "files" / "_run_signal_evaluator.py"
EVAL_DIR = ROOT / "evaluation_output"
PER_MODE_DIR = EVAL_DIR / "per_mode"

sys.path.insert(0, str(ROOT / "files"))
import config  # noqa: E402
import aiohttp  # noqa: E402


def _load_chat_ids() -> list[int]:
    """Reuse same .chat_ids JSON file the bot writes via /start.

    Schema: JSON array of int chat_ids. Empty file '[]' -> no recipients.
    """
    p = ROOT / ".chat_ids"
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    out: list[int] = []
    for item in payload:
        try:
            out.append(int(item))
        except Exception:
            continue
    return sorted(set(out))


def _resolve_token() -> str:
    """Resolve TG token. Tries config (env var) first, falls back to
    parsing .runtime/bot_bg_runner.cmd (where bot wrapper persists it).
    Scheduled tasks invoke this script without env var, so fallback is
    required for cron-like execution.
    """
    tok = str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    if tok:
        return tok
    runner = ROOT / ".runtime" / "bot_bg_runner.cmd"
    if runner.exists():
        import re
        for ln in runner.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"set\s+TELEGRAM_BOT_TOKEN\s*=\s*(\S+)", ln, re.IGNORECASE)
            if m:
                return m.group(1).strip()
    return ""


def _md_to_html(text: str) -> str:
    """Convert our digest's lightweight Markdown to HTML.
    Telegram MarkdownV1 trips on `[mode]` (treated as link), `*` in tokens, etc.
    HTML is more forgiving for our use case.
    Substitutions:
      *bold*        → <b>bold</b>
      _italic_      → <i>italic</i>
      `code`        → <code>code</code>
      <,>,&         → escaped first
    """
    import re
    # 1) Escape HTML metachars first (without breaking already-emitted tags — we
    #    have no pre-existing tags in our digests).
    out = (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
    # 2) Inline replacements. Use non-greedy so multiple per line work.
    out = re.sub(r"\*([^\*\n]+?)\*",  r"<b>\1</b>",    out)
    out = re.sub(r"_([^_\n]+?)_",     r"<i>\1</i>",    out)
    out = re.sub(r"`([^`\n]+?)`",     r"<code>\1</code>", out)
    return out


async def send_tg(text: str) -> None:
    token = _resolve_token()
    if not token:
        print("[skill-eval] no TELEGRAM_BOT_TOKEN (env + .runtime fallback) — printing instead:")
        print(text); return
    chat_ids = _load_chat_ids()
    if not chat_ids:
        print("[weekly-eval] no chat IDs — printing instead:")
        print(text); return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    html_text = _md_to_html(text)
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for cid in chat_ids:
            payload = {
                "chat_id": cid, "text": html_text[:4000],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            try:
                async with session.post(url, json=payload) as r:
                    if r.status != 200:
                        body = await r.text()
                        print(f"[skill-eval] tg send {cid} HTTP {r.status}: {body[:200]}")
                        # Fallback: plain text (no formatting)
                        plain = {"chat_id": cid, "text": text[:4000],
                                 "disable_web_page_preview": True}
                        async with session.post(url, json=plain) as r2:
                            if r2.status != 200:
                                print(f"[skill-eval] plain-text fallback also failed {cid}: HTTP {r2.status}")
            except Exception as e:
                print(f"[skill-eval] tg send failed {cid}: {e}")


def run_evaluator(window_days: int) -> int:
    cmd = [str(PYEMBED), str(WRAPPER),
           "--window-days", str(window_days), "--per-mode"]
    print(f"[skill-eval] running: {' '.join(cmd[1:])}")
    return subprocess.call(cmd, cwd=str(ROOT))


def _verdict_emoji(alpha: float) -> str:
    if alpha >= 5: return "🟢"
    if alpha >= 1: return "🟡"
    if alpha >= -1: return "⚪"
    return "🔴"


def parse_per_mode_reports() -> dict[str, dict]:
    """Read per_mode/<mode>/report.json for each mode dir."""
    out: dict[str, dict] = {}
    if not PER_MODE_DIR.exists():
        return out
    for mode_dir in PER_MODE_DIR.iterdir():
        if not mode_dir.is_dir(): continue
        rj = mode_dir / "report.json"
        if not rj.exists(): continue
        try:
            with io.open(rj, encoding="utf-8") as f:
                out[mode_dir.name] = json.load(f)
        except Exception as e:
            print(f"[weekly-eval] parse fail {rj}: {e}")
    return out


def _collect_verdicts(reports: dict[str, dict]) -> list[dict]:
    out = []
    for mode, r in reports.items():
        for v in r.get("trade_verdicts", []) or []:
            v = dict(v); v["_mode"] = mode
            out.append(v)
    return out


def build_digest(reports: dict[str, dict], cadence: str = "weekly") -> str | None:
    """Returns digest string, or None if cadence == daily and nothing notable."""
    if cadence == "daily":
        return _build_daily_digest(reports)
    if cadence == "monthly":
        return _build_monthly_digest(reports)
    return _build_weekly_digest(reports)


def _build_daily_digest(reports: dict[str, dict]) -> str | None:
    """Silent unless incidents:
      - premature exits where capture < 0.3 AND pnl > 0
      - missed sustained trends with gain >= 5%
      - losing trades < -1.5%
    """
    if not reports:
        return None

    verdicts = _collect_verdicts(reports)
    premature = [v for v in verdicts
                 if v.get("sell_lateness_bars") is not None
                 and v.get("sell_lateness_bars") < -3
                 and (v.get("captured_pnl_pct") or 0) > 0
                 and (v.get("capture_ratio") or 1.0) < 0.3]
    losers = [v for v in verdicts if (v.get("captured_pnl_pct") or 0) < -1.5]

    # De-dup missed opps across modes (same coin appears in each per-mode report)
    big_misses = []
    seen_misses = set()
    for mode, r in reports.items():
        for t in r.get("missed_opportunities", []) or []:
            if (t.get("gain_pct") or 0) < 5.0:
                continue
            key = (t.get("symbol"), str(t.get("true_start_ts")))
            if key in seen_misses:
                continue
            seen_misses.add(key)
            big_misses.append({**t, "_mode": mode})

    if not (premature or losers or big_misses):
        return None  # silent — nothing to alert on

    lines = [f"⚠️ *Skill daily check (24h)*",
             f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_",
             ""]

    if premature:
        lines.append(f"✂️ *Premature exits* ({len(premature)}):")
        for v in sorted(premature, key=lambda x: x.get("capture_ratio") or 0)[:3]:
            cr = (v.get("capture_ratio") or 0) * 100
            lines.append(f"  · {v.get('symbol')} `[{v.get('_mode')}]` "
                         f"capture {cr:.0f}%, sold {abs(v.get('sell_lateness_bars',0))}b before peak")

    if big_misses:
        lines.append("")
        lines.append(f"🚫 *Missed sustained trends* ({len(big_misses)}):")
        for t in sorted(big_misses, key=lambda x: -(x.get("gain_pct") or 0))[:3]:
            lines.append(f"  · {t.get('symbol')} `[{t.get('_mode')}]` "
                         f"gain {t.get('gain_pct'):+.1f}% — bot missed entry")

    if losers:
        lines.append("")
        lines.append(f"🔴 *Losing trades* ({len(losers)}):")
        for v in sorted(losers, key=lambda x: x.get("captured_pnl_pct") or 0)[:3]:
            lines.append(f"  · {v.get('symbol')} `[{v.get('_mode')}]` "
                         f"P&L {v.get('captured_pnl_pct',0):+.2f}%")

    lines.append("")
    lines.append("_run skill manually for details: pyembed/python.exe files/_run_signal_evaluator.py --window-days 1 --per-mode_")
    return "\n".join(lines)


def _build_monthly_digest(reports: dict[str, dict]) -> str:
    """30d strategic summary: alpha trend, top recurring blind-spots,
    architectural recommendations.
    """
    if not reports:
        return "⚠️ *Monthly skill review*: no reports"

    lines = [f"📅 *Monthly skill review (30d)*",
             f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d')} UTC_",
             ""]

    # Per-mode alpha rank
    lines.append("*Mode performance over 30d:*")
    rows = []
    for mode, r in reports.items():
        s = r.get("summary", {})
        rows.append((float(s.get("alpha_vs_buy_and_hold_pct") or 0),
                     mode, int(s.get("total_buy_signals") or 0),
                     float(s.get("median_capture_ratio") or 0)))
    rows.sort(key=lambda x: -x[0])
    for alpha, mode, buys, capture in rows:
        emoji = _verdict_emoji(alpha)
        lines.append(f"{emoji} `{mode}`: alpha *{alpha:+.2f}%*, buys={buys}, "
                     f"cap={capture*100:.0f}%")

    # Top recurring missed coins (3+ separate trends in 30d)
    coin_misses: dict[str, list[float]] = {}
    # De-dup across modes (same trend appears in each per-mode report)
    seen_pairs: set[tuple[str, str]] = set()
    for mode, r in reports.items():
        for t in r.get("missed_opportunities", []) or []:
            sym = t.get("symbol")
            if not sym: continue
            key = (sym, str(t.get("true_start_ts")))
            if key in seen_pairs: continue
            seen_pairs.add(key)
            coin_misses.setdefault(sym, []).append(float(t.get("gain_pct") or 0))
    recurring = [(s, len(g), sum(g)) for s, g in coin_misses.items() if len(g) >= 3]
    recurring.sort(key=lambda x: -x[2])
    if recurring:
        lines.append("")
        lines.append(f"🎯 *Recurring blind-spot coins (3+ missed trends):*")
        for sym, n, total_gain in recurring[:5]:
            lines.append(f"  · {sym}: {n} missed trends, total gain {total_gain:+.0f}%")

    # Total verdict-class breakdown
    verdicts = _collect_verdicts(reports)
    if verdicts:
        from collections import Counter
        vcounts = Counter(v.get("verdict","unknown") for v in verdicts)
        lines.append("")
        lines.append(f"📋 *Trade verdicts (n={len(verdicts)}):*")
        for vk in ("optimal", "late_entry_optimal_exit", "optimal_entry_late_exit",
                   "late_entry_late_exit", "optimal_entry_premature_exit",
                   "late_entry_premature_exit", "losing_trade"):
            if vcounts.get(vk):
                lines.append(f"  · {vk.replace('_',' ')}: {vcounts[vk]}")

    lines.append("")
    lines.append("_strategic decisions: architecture review · model retrain · regime audit_")
    return "\n".join(lines)


def _build_weekly_digest(reports: dict[str, dict]) -> str:
    if not reports:
        return "⚠️ *Weekly signal eval*: no reports generated"
    lines = [f"📊 *Weekly signal evaluation*",
             f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_",
             ""]

    # Per-mode summary
    rows = []
    for mode, r in reports.items():
        s = r.get("summary", {})
        alpha = float(s.get("alpha_vs_buy_and_hold_pct") or 0.0)
        miss_rate = float(s.get("miss_rate") or 0) * 100
        capture = float(s.get("median_capture_ratio") or 0)
        n_buys = int(s.get("total_buy_signals") or 0)
        n_trends = int(s.get("total_trends_in_period") or 0)
        rows.append((alpha, mode, n_buys, n_trends, miss_rate, capture))

    rows.sort(key=lambda x: -x[0])  # highest alpha first
    lines.append("*По режимам (alpha):*")
    for alpha, mode, n_buys, n_trends, miss_rate, capture in rows:
        emoji = _verdict_emoji(alpha)
        cap_str = f"cap={capture*100:.0f}%" if capture else "cap=—"
        lines.append(f"{emoji} `{mode}`: alpha *{alpha:+.2f}%*, "
                     f"buys={n_buys}, trends={n_trends}, miss={miss_rate:.0f}%, {cap_str}")

    # Coaching: top issues across all modes
    all_verdicts = []
    for mode, r in reports.items():
        for v in r.get("trade_verdicts", []) or []:
            v["_mode"] = mode
            all_verdicts.append(v)

    late_buys = [v for v in all_verdicts if (v.get("buy_lateness_bars") or 0) >= 8]
    early_sells = [v for v in all_verdicts
                   if (v.get("sell_lateness_bars") is not None
                       and v.get("sell_lateness_bars") < -3
                       and (v.get("captured_pnl_pct") or 0) > 0)]
    losers = [v for v in all_verdicts if (v.get("captured_pnl_pct") or 0) < -1]

    if late_buys:
        lines.append("")
        lines.append(f"⏰ *Late entries* ({len(late_buys)}):")
        for v in sorted(late_buys, key=lambda x: -(x.get("buy_lateness_bars") or 0))[:3]:
            lines.append(f"  · {v.get('symbol','?')} `[{v.get('_mode')}]` "
                         f"late {v.get('buy_lateness_bars')} bars "
                         f"({(v.get('buy_lateness_pct_of_move') or 0):.0f}% of move)")

    if early_sells:
        lines.append("")
        lines.append(f"✂️ *Premature exits* ({len(early_sells)}):")
        for v in sorted(early_sells, key=lambda x: x.get("sell_lateness_bars", 0))[:3]:
            lines.append(f"  · {v.get('symbol','?')} `[{v.get('_mode')}]` "
                         f"sold {abs(v.get('sell_lateness_bars',0))} bars "
                         f"before peak (capture {v.get('captured_pnl_pct',0):+.2f}%)")

    if losers:
        lines.append("")
        lines.append(f"🔴 *Losing trades* ({len(losers)}):")
        for v in sorted(losers, key=lambda x: x.get("captured_pnl_pct") or 0)[:3]:
            lines.append(f"  · {v.get('symbol','?')} `[{v.get('_mode')}]` "
                         f"P&L {v.get('captured_pnl_pct',0):+.2f}%")

    lines.append("")
    lines.append(f"_full reports: `evaluation_output/per_mode/<mode>/report.md`_")
    return "\n".join(lines)


def export_missed_trends_for_scout(reports: dict[str, dict]) -> Path:
    """Hybrid architecture (2026-05-05): write skill-found missed trends
    to JSON for trend_scout consumption. Each missed trend = a coin where
    ZigZag confirmed sustainable uptrend but bot did NOT enter.

    File: evaluation_output/skill_missed_trends.json

    Spec: docs/specs/features/signal-evaluator-integration-spec.md
          (Hybrid section: skill = oracle, scout = controller)
    """
    out_path = EVAL_DIR / "skill_missed_trends.json"
    # De-dup by (symbol, true_start_ts) — same coin can appear in multiple
    # mode reports (since per-mode runs filter events but evaluate same trends).
    seen: set[tuple[str, str]] = set()
    missed: list[dict] = []
    for mode, r in reports.items():
        for trend in r.get("missed_opportunities", []) or []:
            sym = (trend.get("symbol") or "").upper()
            start_ts = str(trend.get("true_start_ts") or "")
            key = (sym, start_ts)
            if not sym or key in seen:
                continue
            seen.add(key)
            missed.append({
                "symbol": sym,
                "mode_seen_in": mode,
                "tf": (r.get("config", {}) or {}).get("timeframe", "15m"),
                "start_ts": trend.get("true_start_ts"),
                "end_ts": trend.get("true_end_ts"),
                "start_price": trend.get("true_start_price"),
                "end_price": trend.get("true_end_price"),
                "gain_pct": float(trend.get("gain_pct") or 0.0),
                "duration_bars": int(trend.get("duration_bars") or 0),
                # Score: high gain_pct + reasonable duration → high trend_score
                # Maps to scout's TREND_SCORE_MIN (default 60).
                "trend_score": min(100.0, 50.0 + float(trend.get("gain_pct") or 0.0) * 4),
            })
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skill_version": "signal-efficiency-evaluator-v1",
        "missed_trends": missed,
        "n": len(missed),
    }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[weekly-eval] wrote {len(missed)} missed trends to {out_path}")
    return out_path


def trigger_scout_with_skill_recs(missed_path: Path, dry_run: bool = False) -> None:
    """Phase 5 hybrid: invoke trend_scout in skill-fed mode.

    Scout reads missed_trends.json instead of running its heuristic scan,
    then runs its existing propose → validate → apply pipeline.
    dry_run=True (monthly) → proposals computed but NOT applied to config.
    """
    cmd = [str(PYEMBED), str(ROOT / "files" / "trend_scout.py"),
           "--source", "skill", "--missed-file", str(missed_path),
           "--auto-apply-risk", "low"]
    if dry_run:
        cmd.append("--dry-run")
    label = "dry-run" if dry_run else "auto-apply"
    print(f"[skill-eval] running scout with skill recs ({label})...")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    print(f"[skill-eval] scout exit code {rc}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cadence", choices=["daily", "weekly", "monthly"],
                   default="weekly",
                   help="daily=1d silent-on-no-news; weekly=7d full digest+scout; "
                        "monthly=30d strategic review")
    args = p.parse_args()

    cadence_to_days = {"daily": 1, "weekly": 7, "monthly": 30}
    window_days = cadence_to_days[args.cadence]
    print(f"[skill-eval] cadence={args.cadence}, window={window_days}d")

    rc = run_evaluator(window_days)
    if rc != 0:
        print(f"[skill-eval] evaluator exit code {rc}")
    reports = parse_per_mode_reports()
    digest = build_digest(reports, cadence=args.cadence)

    if digest is None:
        print(f"[skill-eval] {args.cadence}: no incidents — silent (TG not posted)")
    else:
        print("\n" + "=" * 60)
        print(f"[skill-eval] {args.cadence.upper()} DIGEST:")
        print("=" * 60)
        print(digest)
        print("=" * 60 + "\n")
        asyncio.run(send_tg(digest))

    # Hybrid scout integration:
    #   - weekly: full skill→scout→auto-apply (low risk)
    #   - monthly: skill→scout proposals only (no auto-apply, manual review)
    #   - daily: SKIP (sample too small, oscillation risk)
    if args.cadence in ("weekly", "monthly"):
        try:
            missed_path = export_missed_trends_for_scout(reports)
            if missed_path.exists():
                # Monthly = dry-run; weekly = real auto-apply
                trigger_scout_with_skill_recs(
                    missed_path,
                    dry_run=(args.cadence == "monthly"),
                )
        except Exception as e:
            print(f"[skill-eval] scout integration failed (non-fatal): {e}")

    print(f"[skill-eval] {args.cadence} done")


if __name__ == "__main__":
    main()
