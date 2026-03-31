"""
rl_bootstrap.py — Offline RL Training Bootstrap

Processes existing bot_events.jsonl to pre-populate the experience
replay buffer WITHOUT calling Claude API for every trade.

Instead uses a fast heuristic critic for historical trades:
  - Computes reward from PnL + forward accuracy
  - Skips expensive Claude calls (use for historical backfill)
  - Then runs initial CMA-ES optimization

After bootstrap, the live rl_agent.py will call real Claude critic
for all new trades going forward.

Usage:
    python rl_bootstrap.py [--max-trades 500] [--optimize]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def heuristic_critic_score(
    pnl_pct: float,
    vol_x: float,
    adx: float,
    macd: float,
    slope: float,
    mode: str,
    bars_held: int,
    exit_reason: str,
) -> tuple[float, float, float, float]:
    """
    Fast heuristic critic score for historical trades.
    Returns (entry_quality, exit_quality, timing, overall).
    Range: -1.0 … +1.0
    """
    # Entry quality heuristics
    entry = 0.0
    if vol_x >= 1.5:   entry += 0.3
    elif vol_x >= 1.0: entry += 0.1
    elif vol_x < 0.5:  entry -= 0.5

    if adx >= 30:      entry += 0.2
    elif adx >= 20:    entry += 0.1
    elif adx < 15:     entry -= 0.3

    if macd > 0:       entry += 0.15
    else:              entry -= 0.20

    if slope >= 0.3:   entry += 0.15
    elif slope >= 0.1: entry += 0.05
    else:              entry -= 0.10

    # Exit quality heuristics
    exit_q = 0.0
    if "RSI перекуплен" in exit_reason:         exit_q = 0.4    # captured overbought exit
    elif "WEAK: RSI дивергенция" in exit_reason: exit_q = 0.1   # slightly early but signal-based
    elif "ATR-трейл" in exit_reason:             exit_q = -0.1  # ATR stop = price moved against
    elif "время" in exit_reason:                 exit_q = -0.2  # time exit = didn't find direction
    elif "EMA20" in exit_reason:                 exit_q = -0.3  # fast exit = entry was bad

    # Timing: did we hold appropriate duration?
    if bars_held <= 1:  exit_q -= 0.4    # too fast — never had a chance
    elif bars_held <= 3: exit_q -= 0.1

    # Adjust for PnL reality
    if pnl_pct > 2.0:   exit_q += 0.3
    elif pnl_pct > 0.5: exit_q += 0.1
    elif pnl_pct < -1.5: exit_q -= 0.2

    # Timing score: mode-appropriate duration
    target_bars = {"impulse_speed": 8, "breakout": 6, "retest": 10, "trend": 20, "alignment": 16}
    target = target_bars.get(mode, 12)
    bar_ratio = bars_held / target
    timing = 0.0
    if 0.5 <= bar_ratio <= 1.5:  timing = 0.2    # within reasonable range
    elif bar_ratio < 0.3:        timing = -0.3   # way too fast
    elif bar_ratio > 2.0:        timing = -0.1   # held too long

    overall = 0.4 * entry + 0.35 * exit_q + 0.25 * timing
    return (
        max(-1.0, min(1.0, entry)),
        max(-1.0, min(1.0, exit_q)),
        max(-1.0, min(1.0, timing)),
        max(-1.0, min(1.0, overall)),
    )


def bootstrap(
    events_file: str = "bot_events.jsonl",
    max_trades: Optional[int] = None,
    run_optimize: bool = True,
) -> int:
    """
    Process historical events to bootstrap RL memory.
    Returns number of experiences created.
    """
    import uuid
    from rl_critic import TradeExperience, compute_reward, extract_state
    from rl_memory import save_experience, memory_stats, load_experiences

    # Check what's already in memory
    existing = load_experiences()
    existing_ids = {e.trade_id for e in existing}
    log.info("Existing experiences: %d", len(existing))

    # Load events
    events = []
    with open(events_file, encoding="utf-8") as f:
        for line in f:
            try: events.append(json.loads(line))
            except: pass

    log.info("Loaded %d events from %s", len(events), events_file)

    # Build entry→exit pairs
    entries = [e for e in events if e['event'] == 'entry']
    exits   = [e for e in events if e['event'] == 'exit']
    forwards = [e for e in events if e['event'] == 'forward']

    # Forward accuracy lookup: (sym, mode) → {3: bool, 5: bool, 10: bool}
    fwd_map = defaultdict(dict)
    for f in forwards:
        key = (f['sym'], f.get('mode',''))
        h = f.get('horizon')
        if h:
            fwd_map[key][h] = {
                'correct': f.get('correct'),
                'ret': f.get('pnl_pct'),
            }

    # Exit lookup
    exit_map = defaultdict(list)
    for x in exits:
        key = (x['sym'], x.get('mode',''), x.get('tf',''))
        exit_map[key].append(x)

    # BTC context lookup
    bull_events = {e['ts'][:13]: e for e in events if e['event'] == 'bull_day'}

    created = 0
    skipped = 0

    entries_to_process = entries[:max_trades] if max_trades else entries

    for entry in entries_to_process:
        sym  = entry.get('sym', '?')
        mode = entry.get('mode', '?')
        tf   = entry.get('tf', '15m')
        key  = (sym, mode, tf)

        # Find matching exit
        matched_exits = exit_map.get(key, [])
        exit_event = None
        for x in matched_exits:
            if x.get('ts', '') > entry.get('ts', ''):
                exit_event = x
                break

        if not exit_event:
            skipped += 1
            continue

        # Skip if already in memory
        trade_id = f"{sym}_{entry['ts'][:16]}_{mode}"[:20]

        # BTC context at entry time
        hour_key = entry['ts'][:13]
        bull_ev  = bull_events.get(hour_key, {})

        # Build state
        state = extract_state(entry)
        state['ml_proba'] = entry.get('ml_proba', 0.5)

        # Forward accuracy
        fwd = fwd_map.get((sym, mode), {})

        pnl      = float(exit_event.get('pnl_pct', 0))
        vol      = float(entry.get('vol_x', 1))
        adx_v    = float(entry.get('adx', 20))
        macd_v   = float(entry.get('macd_hist', 0))
        slope_v  = float(entry.get('slope_pct', 0.1))
        bars     = int(exit_event.get('bars_held', 0))
        reason   = exit_event.get('reason', '')

        entry_q, exit_q, timing, overall = heuristic_critic_score(
            pnl, vol, adx_v, macd_v, slope_v, mode, bars, reason
        )

        exp = TradeExperience(
            trade_id     = trade_id,
            sym          = sym,
            tf           = tf,
            mode         = mode,
            ts_entry     = entry.get('ts', ''),
            ts_exit      = exit_event.get('ts', ''),
            btc_vs_ema50 = float(bull_ev.get('btc_vs_ema50', 0)),
            is_bull_day  = bool(bull_ev.get('is_bull', False)),
            market_regime = "bull_trend" if bull_ev.get('is_bull') else "neutral",
            state        = state,
            trail_k      = float(entry.get('trail_k', 2.0)),
            max_hold_bars = int(entry.get('max_hold_bars', 16)),
            entry_price  = float(exit_event.get('entry_price', 0)),
            exit_price   = float(exit_event.get('exit_price', 0)),
            pnl_pct      = pnl,
            bars_held    = bars,
            exit_reason  = reason,
            t3_correct   = fwd.get(3, {}).get('correct'),
            t5_correct   = fwd.get(5, {}).get('correct'),
            t10_correct  = fwd.get(10, {}).get('correct'),
            t3_ret       = fwd.get(3, {}).get('ret'),
            t5_ret       = fwd.get(5, {}).get('ret'),
            t10_ret      = fwd.get(10, {}).get('ret'),
            critic_score = overall,
            critic_entry = entry_q,
            critic_exit  = exit_q,
            critic_timing = timing,
            critic_feedback = f"[HEURISTIC] {'good' if overall > 0 else 'poor'} decision",
        )
        exp.reward = compute_reward(exp)

        save_experience(exp)
        created += 1

        if created % 100 == 0:
            log.info("  Created %d / %d experiences...", created, len(entries_to_process))

    log.info("Bootstrap complete: %d created, %d skipped (no exit)", created, skipped)

    stats = memory_stats(load_experiences())
    log.info("Memory stats: total=%d WR=%.1f%% avg_pnl=%.2f%%",
             stats.get('total',0), stats.get('win_rate',0)*100, stats.get('avg_pnl',0))

    for mode_name, ms in stats.get('by_mode', {}).items():
        log.info("  [%s] n=%d avg_pnl=%.2f%%", mode_name, ms['n'], ms['avg_pnl'])

    if run_optimize and created > 0:
        log.info("\nRunning initial CMA-ES optimization...")
        from rl_optimizer import run_optimization_step
        best = run_optimization_step(batch_size=256, n_generations=20, population=30, verbose=True)
        if best:
            from pathlib import Path
            Path("rl_params.json").write_text(json.dumps(best, indent=2))
            log.info("Initial optimized params saved to rl_params.json")
            log.info("Top 10 params:")
            for k, v in list(best.items())[:10]:
                log.info("  %s = %.4f", k, v)

    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap RL from historical events")
    parser.add_argument("--events",     default="bot_events.jsonl")
    parser.add_argument("--max-trades", type=int, default=None)
    parser.add_argument("--optimize",   action="store_true", default=True)
    parser.add_argument("--no-optimize",dest="optimize", action="store_false")
    args = parser.parse_args()

    n = bootstrap(args.events, args.max_trades, args.optimize)
    print(f"\nDone: {n} experiences created")
