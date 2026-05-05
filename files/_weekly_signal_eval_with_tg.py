"""Weekly signal-evaluator runner with TG digest.

Designed to be invoked by Windows Scheduled Task on Sundays 03:30 local.

Flow:
  1) Run skill via wrapper (--per-mode, last 7 d).
  2) Parse evaluation_output/per_mode/<mode>/report.json for each mode.
  3) Build compact TG digest (top-3 highlights per mode).
  4) Send to TG admin chat(s) via existing infrastructure.

Spec: docs/specs/features/signal-evaluator-integration-spec.md
      (operational follow-up §8: scheduled CryptoBot_SignalEvaluator_Weekly).
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


async def send_tg(text: str) -> None:
    token = str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    if not token:
        print("[weekly-eval] no TELEGRAM_BOT_TOKEN — printing instead:")
        print(text); return
    chat_ids = _load_chat_ids()
    if not chat_ids:
        print("[weekly-eval] no chat IDs — printing instead:")
        print(text); return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for cid in chat_ids:
            payload = {
                "chat_id": cid, "text": text[:4000],
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
            try:
                async with session.post(url, json=payload) as r:
                    r.raise_for_status()
            except Exception as e:
                print(f"[weekly-eval] tg send failed {cid}: {e}")


def run_evaluator() -> int:
    cmd = [str(PYEMBED), str(WRAPPER), "--window-days", "7", "--per-mode"]
    print(f"[weekly-eval] running: {' '.join(cmd[1:])}")
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


def build_digest(reports: dict[str, dict]) -> str:
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


def trigger_scout_with_skill_recs(missed_path: Path) -> None:
    """Phase 5 hybrid: invoke trend_scout in skill-fed mode.

    Scout reads missed_trends.json instead of running its heuristic scan,
    then runs its existing propose → validate → apply pipeline.
    """
    cmd = [str(PYEMBED), str(ROOT / "files" / "trend_scout.py"),
           "--source", "skill", "--missed-file", str(missed_path),
           "--auto-apply-risk", "low"]
    print(f"[weekly-eval] running scout with skill recs...")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    print(f"[weekly-eval] scout exit code {rc}")


def main():
    rc = run_evaluator()
    if rc != 0:
        print(f"[weekly-eval] evaluator exit code {rc}")
    reports = parse_per_mode_reports()
    digest = build_digest(reports)
    print("\n" + "=" * 60)
    print("[weekly-eval] DIGEST:")
    print("=" * 60)
    print(digest)
    print("=" * 60 + "\n")
    asyncio.run(send_tg(digest))

    # Hybrid: export skill missed trends + trigger scout (best-effort)
    try:
        missed_path = export_missed_trends_for_scout(reports)
        if missed_path.exists():
            trigger_scout_with_skill_recs(missed_path)
    except Exception as e:
        print(f"[weekly-eval] scout integration failed (non-fatal): {e}")

    print("[weekly-eval] done")


if __name__ == "__main__":
    main()
