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
    print("[weekly-eval] done")


if __name__ == "__main__":
    main()
