"""
rl_top_gainers_critic.py — Daily Top-Gainers Critic Agent

Runs automatically at 00:00 UTC and 12:00 UTC.
At each run:
  1. Fetches Binance USDT spot top-gainers (24h %)
  2. Reconstructs bot's signals for the review window from bot_events.jsonl
  3. Computes coverage / precision / miss metrics
  4. Calls Claude API critic for structured feedback
  5. Saves report to rl_gainers_log.jsonl for trend analysis

Target state (what the critic optimises for):
  - Bot gives BUY signals for coins that will be in top-gainers by EOD
  - Signals arrive as EARLY as possible (max time to profit)
  - Precision: fraction of bot signals that land in top-gainers
  - Coverage: fraction of top-gainers that received a signal
  - Lead time: average hours between first signal and peak gain

Deployment:
  - Scheduled by rl_headless_worker._gainers_loop()  (12h interval)
  - Or run standalone:  python rl_top_gainers_critic.py --mode midnight
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

log = logging.getLogger(__name__)

# ── Output files ───────────────────────────────────────────────────────────────
GAINERS_LOG_FILE   = Path("rl_gainers_log.jsonl")
GAINERS_REPORT_DIR = Path("rl_gainers_reports")

# ── Critic constants ───────────────────────────────────────────────────────────
TOP_N_GAINERS   = 20    # consider top-20 by 24h % change
MIN_GAIN_PCT    = 3.0   # only coins that gained >= 3% qualify as "top gainer"
BINANCE_API_URL = "https://api.binance.com"

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GainerInfo:
    sym:        str
    change_24h: float     # % change
    price_open: float
    price_now:  float
    volume_usdt: float
    rank:       int = 0   # filled by caller after sorting


@dataclass
class BotSignal:
    sym:       str
    ts:        str
    mode:      str
    tf:        str
    price:     float
    vol_x:     float
    adx:       float
    slope:     float
    rsi:       float
    exit_pnl:  Optional[float]   = None
    exit_ts:   Optional[str]     = None
    exit_reason: str             = ""


@dataclass
class DayReview:
    """Complete review for one evaluation window."""
    review_id:    str
    window_start: str    # ISO UTC
    window_end:   str
    session:      str    # "midnight" | "noon"

    # Binance data
    top_gainers:  List[GainerInfo] = field(default_factory=list)

    # Bot signals in window
    signals:      List[BotSignal]  = field(default_factory=list)

    # Overlap analysis
    hit_syms:     List[str] = field(default_factory=list)   # signalled AND gainer
    miss_syms:    List[str] = field(default_factory=list)   # gainer but NOT signalled
    noise_syms:   List[str] = field(default_factory=list)   # signalled but NOT gainer

    # Metrics
    precision:    float = 0.0   # hit / total_signals
    coverage:     float = 0.0   # hit / total_gainers
    avg_lead_h:   float = 0.0   # avg hours from signal to window_end
    score:        float = 0.0   # composite

    # Claude feedback
    critic_score:    Optional[float] = None
    critic_summary:  Optional[str]   = None
    critic_fixes:    Optional[List[dict]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['top_gainers'] = [asdict(g) for g in self.top_gainers]
        d['signals']     = [asdict(s) for s in self.signals]
        return d


# ── Binance API ────────────────────────────────────────────────────────────────

async def fetch_top_gainers(
    session: aiohttp.ClientSession,
    top_n: int = TOP_N_GAINERS,
    min_gain_pct: float = MIN_GAIN_PCT,
) -> List[GainerInfo]:
    """
    Fetch 24h top gainers from Binance /api/v3/ticker/24hr.
    Filters: USDT pairs only, minimum gain threshold, minimum USDT volume.
    """
    try:
        async with session.get(
            f"{BINANCE_API_URL}/api/v3/ticker/24hr",
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as e:
        log.error("Binance API error fetching 24hr tickers: %s", e)
        return []

    gainers = []
    for t in data:
        sym = str(t.get("symbol", ""))
        if not sym.endswith("USDT"):
            continue
        try:
            change = float(t.get("priceChangePercent", 0))
            volume = float(t.get("quoteVolume", 0))     # USDT volume
            price  = float(t.get("lastPrice", 0))
            open_p = float(t.get("openPrice", 0))
        except (ValueError, TypeError):
            continue

        if change < min_gain_pct:
            continue
        if volume < 500_000:        # skip illiquid
            continue
        if price <= 0:
            continue

        gainers.append(GainerInfo(
            sym=sym,
            change_24h=round(change, 3),
            price_open=round(open_p, 8),
            price_now=round(price, 8),
            volume_usdt=round(volume, 0),
        ))

    gainers.sort(key=lambda g: g.change_24h, reverse=True)
    for i, g in enumerate(gainers[:top_n]):
        g.rank = i + 1

    return gainers[:top_n]


async def fetch_klines_open(
    session: aiohttp.ClientSession,
    sym: str,
    start_ts_ms: int,
) -> Optional[float]:
    """Fetch open price of the first 1h candle at start_ts_ms for lead-time calc."""
    try:
        url = f"{BINANCE_API_URL}/api/v3/klines"
        params = {"symbol": sym, "interval": "1h", "startTime": start_ts_ms, "limit": 1}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if data:
                return float(data[0][1])  # open price
    except Exception:
        pass
    return None


# ── Bot events reader ──────────────────────────────────────────────────────────

def load_bot_signals(
    window_start: datetime,
    window_end: datetime,
    events_file: Path = Path("bot_events.jsonl"),
) -> List[BotSignal]:
    """
    Read bot_events.jsonl and return BotSignal list for the given time window.
    Pairs entry↔exit events for PnL annotation.
    """
    if not events_file.exists():
        return []

    raw_entries: List[dict] = []
    raw_exits:   List[dict] = []

    ws = window_start.strftime("%Y-%m-%dT%H:%M")
    we = window_end.strftime("%Y-%m-%dT%H:%M")

    for line in events_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = str(ev.get("ts", ""))[:16]
        if ev.get("event") == "entry" and ws <= ts <= we:
            raw_entries.append(ev)
        elif ev.get("event") == "exit":
            raw_exits.append(ev)

    # Build exit lookup: (sym, mode) → exit event
    exit_map: Dict[Tuple[str, str], dict] = {}
    for x in raw_exits:
        key = (x.get("sym", ""), x.get("mode", ""))
        if key not in exit_map or x.get("ts", "") > exit_map[key].get("ts", ""):
            exit_map[key] = x

    signals = []
    for ev in raw_entries:
        sym  = ev.get("sym", "")
        mode = ev.get("mode", "")
        x    = exit_map.get((sym, mode), {})
        signals.append(BotSignal(
            sym=sym,
            ts=ev.get("ts", ""),
            mode=mode,
            tf=ev.get("tf", "15m"),
            price=float(ev.get("price", 0)),
            vol_x=float(ev.get("vol_x", 0)),
            adx=float(ev.get("adx", 0)),
            slope=float(ev.get("slope_pct", 0)),
            rsi=float(ev.get("rsi", 0)),
            exit_pnl=x.get("pnl_pct"),
            exit_ts=x.get("ts"),
            exit_reason=x.get("reason", ""),
        ))

    return signals


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(review: DayReview) -> DayReview:
    """Compute precision, coverage, lead-time and composite score."""
    gainer_syms = {g.sym for g in review.top_gainers}
    signal_syms = {s.sym for s in review.signals}

    hits   = gainer_syms & signal_syms
    misses = gainer_syms - signal_syms
    noise  = signal_syms - gainer_syms

    review.hit_syms   = sorted(hits)
    review.miss_syms  = sorted(misses)
    review.noise_syms = sorted(noise)

    n_sig  = len(signal_syms)
    n_gain = len(gainer_syms)

    review.precision = round(len(hits) / max(1, n_sig),  4)
    review.coverage  = round(len(hits) / max(1, n_gain), 4)

    # Lead-time: hours from first signal to window_end
    end_dt = datetime.fromisoformat(review.window_end.replace("Z", "+00:00"))
    lead_hours = []
    for sig in review.signals:
        if sig.sym in hits:
            try:
                sig_dt = datetime.fromisoformat(sig.ts.replace("Z", "+00:00"))
                lead = (end_dt - sig_dt).total_seconds() / 3600
                if lead > 0:
                    lead_hours.append(lead)
            except Exception:
                pass

    review.avg_lead_h = round(sum(lead_hours) / len(lead_hours), 2) if lead_hours else 0.0

    # Composite score: balance coverage and precision, bonus for early signals
    cov   = review.coverage
    prec  = review.precision
    lead  = min(review.avg_lead_h / 12.0, 1.0)   # normalise to 12h window
    review.score = round(0.45 * cov + 0.35 * prec + 0.20 * lead, 4)

    return review


# ── Claude critic prompt ───────────────────────────────────────────────────────

def _build_gainers_critic_prompt(review: DayReview) -> str:
    gainer_lines = "\n".join(
        f"  #{g.rank} {g.sym}: +{g.change_24h:.1f}%  vol=${g.volume_usdt/1e6:.1f}M"
        for g in review.top_gainers[:15]
    )

    hit_signals = [s for s in review.signals if s.sym in set(review.hit_syms)]
    hit_lines   = "\n".join(
        f"  ✅ {s.sym} [{s.mode}] {s.ts[11:16]} vol={s.vol_x:.2f}x ADX={s.adx:.0f} "
        f"PnL={s.exit_pnl:+.2f}%" if s.exit_pnl is not None else
        f"  ✅ {s.sym} [{s.mode}] {s.ts[11:16]} vol={s.vol_x:.2f}x ADX={s.adx:.0f}"
        for s in hit_signals
    ) or "  (none)"

    miss_lines = "\n".join(f"  ❌ {sym}" for sym in review.miss_syms[:10]) or "  (none)"

    noise_lines = "\n".join(
        f"  ⚠️  {s.sym} [{s.mode}] {s.ts[11:16]} vol={s.vol_x:.2f}x → "
        f"PnL={s.exit_pnl:+.2f}%" if s.exit_pnl is not None else
        f"  ⚠️  {s.sym} [{s.mode}] {s.ts[11:16]} vol={s.vol_x:.2f}x"
        for s in review.signals if s.sym in set(review.noise_syms)
    )[:1000] or "  (none)"

    return f"""You are a professional cryptocurrency trading analyst evaluating a momentum trading bot.

Review window: {review.window_start[:16]} → {review.window_end[:16]} UTC  ({review.session})

OBJECTIVE: The bot should detect early signs of coins that will be in Binance's top-gainers list by end of day.
Measure: how many top-gainers did the bot catch, how early, and how many false signals.

TOP-20 GAINERS THIS PERIOD (Binance 24h):
{gainer_lines}

BOT PERFORMANCE:
  Total signals in window: {len(review.signals)}
  Hit (signalled AND gainer): {len(review.hit_syms)} → {review.hit_syms}
  Missed gainers (no signal): {len(review.miss_syms)}
  Noise signals (not gainer): {len(review.noise_syms)}
  Precision: {review.precision*100:.1f}%  Coverage: {review.coverage*100:.1f}%  Avg lead: {review.avg_lead_h:.1f}h

CORRECT SIGNALS (hit gainers):
{hit_lines}

MISSED GAINERS (bot gave no signal):
{miss_lines}

NOISE SIGNALS (bot signalled, coin did NOT gain):
{noise_lines}

Analyse:
1. What patterns distinguish the missed gainers from caught ones? (volume spike? early ADX rise? breakout structure?)
2. What caused the false/noise signals? (weak ADX? low volume? wrong market context?)
3. What specific parameter or filter changes would improve the catch rate?

Respond ONLY with valid JSON (no markdown):
{{
  "score": <float 0..1, overall bot performance vs gainers today>,
  "session_verdict": "<excellent|good|acceptable|poor|terrible>",
  "coverage_analysis": "<what patterns the caught signals shared>",
  "miss_analysis": "<why the bot missed these gainers - what early signals existed>",
  "noise_analysis": "<why noise signals fired - what went wrong>",
  "top3_fixes": [
    {{"priority": 1, "param": "<CONFIG_PARAM or strategy change>", "direction": "<increase|decrease|add|remove>", "reason": "<specific reason>"}},
    {{"priority": 2, "param": "...", "direction": "...", "reason": "..."}},
    {{"priority": 3, "param": "...", "direction": "...", "reason": "..."}}
  ],
  "early_signal_patterns": "<what indicators appear 1-4 hours BEFORE a coin becomes a top gainer>",
  "market_context_note": "<any structural market observation relevant to today>"
}}"""


async def call_gainers_critic(
    session: aiohttp.ClientSession,
    review: DayReview,
    model: str = "claude-sonnet-4-5-20251022",
) -> DayReview:
    """Call Claude API critic with the day review and annotate review with feedback."""
    prompt = _build_gainers_critic_prompt(review)

    payload = {
        "model": model,
        "max_tokens": 1200,
        "temperature": 0.15,
        "system": (
            "You are a professional crypto market analyst and trading system evaluator. "
            "Respond only with valid JSON, no prose outside JSON."
        ),
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(3):
        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=40),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                text = data["content"][0]["text"].strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                result = json.loads(text.strip())
                review.critic_score   = float(result.get("score", 0.5))
                review.critic_summary = (
                    f"[{result.get('session_verdict','?').upper()}] "
                    f"Coverage: {review.coverage*100:.0f}% | "
                    f"{result.get('coverage_analysis','')[:120]} | "
                    f"MISS: {result.get('miss_analysis','')[:120]}"
                )
                review.critic_fixes = result.get("top3_fixes", [])
                log.info(
                    "Gainers critic: score=%.2f verdict=%s coverage=%.0f%% precision=%.0f%%",
                    review.critic_score,
                    result.get("session_verdict", "?"),
                    review.coverage * 100,
                    review.precision * 100,
                )
                # Log early-signal patterns for research
                pattern = result.get("early_signal_patterns", "")
                if pattern:
                    log.info("Early signal patterns: %s", pattern[:200])
                return review
        except json.JSONDecodeError as e:
            log.warning("Critic JSON parse error (attempt %d): %s", attempt + 1, e)
        except aiohttp.ClientError as e:
            log.warning("Critic API error (attempt %d): %s", attempt + 1, e)
            await asyncio.sleep(2 ** attempt)

    review.critic_score   = None
    review.critic_summary = "Critic unavailable"
    return review


# ── Main evaluation function ───────────────────────────────────────────────────

async def run_gainers_review(
    session_name: str,                # "midnight" | "noon"
    *,
    events_file: Path = Path("bot_events.jsonl"),
    http_session: Optional[aiohttp.ClientSession] = None,
    now: Optional[datetime] = None,
) -> DayReview:
    """
    Main entry point for one evaluation run.

    midnight (00:00 UTC): reviews previous 24h (yesterday 00:00 → today 00:00)
    noon    (12:00 UTC): reviews morning session (today 00:00 → today 12:00)

    Returns completed DayReview with metrics and critic feedback.
    """
    now = now or datetime.now(timezone.utc)
    now = now.replace(minute=0, second=0, microsecond=0)

    if session_name == "midnight":
        window_end   = now
        window_start = now - timedelta(hours=24)
    else:  # noon
        window_end   = now
        window_start = now - timedelta(hours=12)

    review_id = f"{session_name}_{window_end.strftime('%Y%m%d_%H%M')}"

    log.info(
        "Starting %s gainers review: %s → %s",
        session_name,
        window_start.strftime("%Y-%m-%d %H:%M"),
        window_end.strftime("%Y-%m-%d %H:%M"),
    )

    review = DayReview(
        review_id=review_id,
        window_start=window_start.strftime("%Y-%m-%dT%H:%M:00Z"),
        window_end=window_end.strftime("%Y-%m-%dT%H:%M:00Z"),
        session=session_name,
    )

    close_session = http_session is None
    if close_session:
        http_session = aiohttp.ClientSession()

    try:
        # 1. Fetch top gainers from Binance
        gainers = await fetch_top_gainers(http_session)
        if not gainers:
            log.warning("No top gainers fetched — Binance API may be unavailable")
        review.top_gainers = gainers
        log.info("Top gainers fetched: %d coins (top: %s +%.1f%%)",
                 len(gainers),
                 gainers[0].sym if gainers else "?",
                 gainers[0].change_24h if gainers else 0)

        # 2. Load bot signals for the window
        signals = load_bot_signals(window_start, window_end, events_file)
        review.signals = signals
        log.info("Bot signals in window: %d", len(signals))

        # 3. Compute metrics
        review = compute_metrics(review)
        log.info(
            "Metrics: precision=%.0f%% coverage=%.0f%% lead=%.1fh score=%.3f "
            "hits=%d misses=%d noise=%d",
            review.precision * 100, review.coverage * 100,
            review.avg_lead_h, review.score,
            len(review.hit_syms), len(review.miss_syms), len(review.noise_syms),
        )

        # 4. Call Claude critic
        review = await call_gainers_critic(http_session, review)

        # 5. Persist to log
        _save_review(review)

        return review

    finally:
        if close_session:
            await http_session.close()


# ── Persistence ────────────────────────────────────────────────────────────────

def _save_review(review: DayReview) -> None:
    """Append compact review summary to rl_gainers_log.jsonl."""
    record = {
        "review_id":   review.review_id,
        "session":     review.session,
        "window_start": review.window_start,
        "window_end":   review.window_end,
        "top_gainers":  [(g.sym, g.change_24h) for g in review.top_gainers[:10]],
        "n_signals":    len(review.signals),
        "hit_syms":     review.hit_syms,
        "miss_syms":    review.miss_syms,
        "n_noise":      len(review.noise_syms),
        "precision":    review.precision,
        "coverage":     review.coverage,
        "avg_lead_h":   review.avg_lead_h,
        "score":        review.score,
        "critic_score": review.critic_score,
        "critic_summary": (review.critic_summary or "")[:300],
        "top_fixes":    review.critic_fixes or [],
    }
    with GAINERS_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Also save full report for deep analysis
    GAINERS_REPORT_DIR.mkdir(exist_ok=True)
    report_path = GAINERS_REPORT_DIR / f"{review.review_id}.json"
    report_path.write_text(
        json.dumps(review.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Review saved: %s", report_path)


def load_recent_reviews(n: int = 14) -> list:
    """Load last n review summaries from log for trend analysis."""
    if not GAINERS_LOG_FILE.exists():
        return []
    lines = GAINERS_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    result = []
    for line in lines[-n:]:
        try:
            result.append(json.loads(line))
        except Exception:
            pass
    return result


def render_trend_report(reviews: list) -> str:
    """Render multi-day trend summary from recent reviews."""
    if not reviews:
        return "No reviews yet."

    lines = [f"Top-Gainers Critic — last {len(reviews)} sessions", ""]
    for r in reviews:
        score_str = f"{r.get('critic_score', 0):.2f}" if r.get('critic_score') is not None else "n/a"
        top5 = [f"{s}+{p:.0f}%" for s, p in r.get("top_gainers", [])[:5]]
        lines.append(
            f"  {r['review_id']:30s}  "
            f"cov={r.get('coverage',0)*100:.0f}%  prec={r.get('precision',0)*100:.0f}%  "
            f"lead={r.get('avg_lead_h',0):.1f}h  score={score_str}"
        )
        if r.get("hit_syms"):
            lines.append(f"    Hits: {r['hit_syms']}")
        if r.get("miss_syms"):
            lines.append(f"    Missed: {r.get('miss_syms',[])[:5]}")
        top5_str = ", ".join(top5)
        lines.append(f"    Market top: {top5_str}")
        lines.append("")

    # Trend stats
    scores   = [r["score"] for r in reviews if r.get("score") is not None]
    cov_list = [r["coverage"] for r in reviews if r.get("coverage") is not None]
    if scores:
        lines.append(f"Trend: avg_score={sum(scores)/len(scores):.3f}  "
                     f"avg_coverage={sum(cov_list)/len(cov_list)*100:.0f}%")

    return "\n".join(lines)


# ── Standalone runner ──────────────────────────────────────────────────────────

async def _amain() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run top-gainers critic review")
    parser.add_argument("--mode", choices=["midnight", "noon", "now"],
                        default="now",
                        help="Session mode: midnight=24h review, noon=12h, now=auto-detect")
    parser.add_argument("--events", type=Path, default=Path("bot_events.jsonl"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch gainers and compute metrics but skip Claude API call")
    parser.add_argument("--report", action="store_true",
                        help="Print trend report from historical reviews and exit")
    args = parser.parse_args()

    if args.report:
        reviews = load_recent_reviews(30)
        print(render_trend_report(reviews))
        return

    mode = args.mode
    if mode == "now":
        hour = datetime.now(timezone.utc).hour
        mode = "midnight" if hour < 6 or hour >= 20 else "noon"

    if args.dry_run:
        # Override critic call
        async def _noop_critic(http_session, review, **kw):
            review.critic_score   = None
            review.critic_summary = "[DRY RUN] Critic skipped"
            return review
        import rl_top_gainers_critic as _self
        _self.call_gainers_critic = _noop_critic

    review = await run_gainers_review(mode, events_file=args.events)

    print(f"\n=== Review: {review.review_id} ===")
    print(f"Window:    {review.window_start[:16]} → {review.window_end[:16]} UTC")
    print(f"Gainers:   {len(review.top_gainers)} (top: "
          f"{review.top_gainers[0].sym if review.top_gainers else '?'} "
          f"+{review.top_gainers[0].change_24h:.1f}%)")
    print(f"Signals:   {len(review.signals)}")
    print(f"Coverage:  {review.coverage*100:.1f}%  Precision: {review.precision*100:.1f}%")
    print(f"Lead time: {review.avg_lead_h:.1f}h  Score: {review.score:.3f}")
    if review.hit_syms:
        print(f"Hits:      {review.hit_syms}")
    if review.miss_syms:
        print(f"Missed:    {review.miss_syms[:10]}")
    if review.critic_summary:
        print(f"Critic:    {review.critic_summary[:200]}")
    if review.critic_fixes:
        print("Top fixes:")
        for fix in review.critic_fixes:
            print(f"  #{fix.get('priority','?')} {fix.get('param','?')} "
                  f"→ {fix.get('direction','?')}: {fix.get('reason','?')[:80]}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    asyncio.run(_amain())
