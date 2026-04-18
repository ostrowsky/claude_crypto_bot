"""
offline_rl.py — Offline RL Training Pipeline

Batch training on accumulated data for four RL components:

  1. Entry Bandit     — enter/skip decision from critic_dataset labels
  2. Trail Bandit     — trail_k/hold selection from trade outcomes
  3. Exit Policy      — fits Q-values from trade trajectories
  4. CMA-ES Optimizer — runs parameter search (existing)

Data sources:
  - critic_dataset.jsonl  -> rich signal records with features + forward labels
  - bot_events.jsonl      -> entry/exit/blocked events from live bot
  - rl_memory.jsonl       -> TradeExperience records (entry context + outcome)

Trigger: called from rl_agent after N trades or on schedule.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rl_critic import TradeExperience, compute_reward
from rl_memory import load_experiences, sample_batch

log = logging.getLogger(__name__)

OFFLINE_STATE_FILE = Path("offline_rl_state.json")
OFFLINE_LOG_FILE = Path("offline_rl_log.jsonl")
CRITIC_DATASET_FILE = Path("critic_dataset.jsonl")
BOT_EVENTS_FILE = Path("bot_events.jsonl")
TOP_GAINER_DATASET_FILE = Path("top_gainer_dataset.jsonl")


# ── Offline state tracking ────────────────────────────────────────────────────

def _load_offline_state() -> dict:
    if OFFLINE_STATE_FILE.exists():
        try:
            return json.loads(OFFLINE_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "last_n_experiences": 0,
        "last_n_critic": 0,
        "last_run_ts": "",
        "runs": 0,
    }


def _save_offline_state(state: dict) -> None:
    OFFLINE_STATE_FILE.write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_critic_dataset(
    max_records: int = 10000,
    since_n: int = 0,
) -> List[dict]:
    """
    Load records from critic_dataset.jsonl.

    Each record has:
      - f: dict of features (slope, rsi, adx, vol_x, daily_range, btc_vs_ema50, ...)
      - decision.action: "take" or "blocked"
      - labels: ret_3, ret_5, ret_10, label_3/5/10, trade_taken, trade_exit_pnl
      - signal_type, tf, sym, is_bull_day
    """
    if not CRITIC_DATASET_FILE.exists():
        return []

    records = []
    n = 0
    with CRITIC_DATASET_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            n += 1
            if n <= since_n:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("f") and rec.get("labels"):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
            if len(records) >= max_records:
                break
    return records


def _load_bot_events_entries(max_records: int = 5000) -> List[dict]:
    """Load entry events from bot_events.jsonl (have features at entry time)."""
    if not BOT_EVENTS_FILE.exists():
        return []

    entries = []
    with BOT_EVENTS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("event") == "entry":
                    entries.append(rec)
            except json.JSONDecodeError:
                continue
            if len(entries) >= max_records:
                break
    return entries


def _load_bot_events_exits() -> Dict[str, dict]:
    """Load exit events keyed by symbol for pairing with entries."""
    if not BOT_EVENTS_FILE.exists():
        return {}

    exits: Dict[str, list] = {}
    with BOT_EVENTS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("event") == "exit":
                    sym = rec.get("sym", "")
                    exits.setdefault(sym, []).append(rec)
            except json.JSONDecodeError:
                continue
    return exits


# ── Top-gainer dataset loader ────────────────────────────────────────────────

def _load_top_gainer_dataset(max_records: int = 50000) -> List[dict]:
    """Load records from top_gainer_dataset.jsonl (ALL watchlist symbols with EOD labels)."""
    if not TOP_GAINER_DATASET_FILE.exists():
        return []
    records = []
    with TOP_GAINER_DATASET_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("features") and rec.get("symbol"):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
            if len(records) >= max_records:
                break
    return records


def _tg_features_to_context(features: dict) -> Tuple[dict, float]:
    """
    Map top_gainer_dataset features to bandit context state dict.

    Returns (state_dict, btc_vs_ema50_approx).
    """
    state = {
        "slope_pct": features.get("tg_ema20_slope", 0.0),
        "adx": features.get("tg_adx", 20.0),
        "rsi": features.get("tg_rsi", 50.0),
        "vol_x": features.get("tg_volume_ratio_1h", 1.0),
        "ml_proba": 0.5,  # no signal-level ML prediction for universal samples
        "daily_range": features.get("tg_daily_range_pct", 3.0),
        "macd_hist": features.get("tg_ema20_slope", 0.0),  # slope as proxy
    }
    btc_vs_ema50 = features.get("tg_btc_return_4h", 0.0)
    return state, btc_vs_ema50


# ── 1. Entry bandit batch training (UNIVERSAL — all watchlist symbols) ───────

def train_entry_bandit(
    *,
    min_samples: int = 30,
    top_n_per_day: int = 20,
    top_gainer_threshold_pct: float = 3.0,
    use_earliest_snapshot: bool = True,
) -> dict:
    """
    Train the enter/skip bandit from TWO data sources:

    PRIMARY: top_gainer_dataset.jsonl — features + EOD labels for ALL watchlist
             symbols at multiple daily snapshots. This ensures the bandit sees
             every top gainer every day, regardless of whether a signal fired.

    SECONDARY: critic_dataset.jsonl — signal-originated records with richer
               features and actual bot decisions. Provides signal-specific context.

    For universal samples (top_gainer_dataset):
      - Each symbol is treated as if bandit must decide ENTER or SKIP
      - Arm is set to ENTER(1) for top gainers, SKIP(0) for others
      - This teaches the bandit what features predict top gainers

    For signal samples (critic_dataset):
      - Arm comes from actual bot decision (take=ENTER, blocked=SKIP)
      - Reward uses asymmetric scheme (penalizes missed top gainers)
    """
    from contextual_bandit import (
        get_entry_bandit, extract_context, ENTRY_STATE_FILE,
        N_ENTRY_ARMS,
    )

    bandit = get_entry_bandit()

    # ── Source 1: Universal samples from top_gainer_dataset ─────────────────
    tg_records = _load_top_gainer_dataset()
    universal_samples = []
    n_universal_top = 0
    tg_days = set()

    # Group by (date, symbol) to pick earliest snapshot per day per symbol
    by_day_sym: Dict[str, Dict[str, list]] = {}
    for rec in tg_records:
        ts_ms = rec.get("ts", 0)
        if not ts_ms:
            continue
        dt = datetime.utcfromtimestamp(ts_ms / 1000)
        day_key = dt.strftime("%Y-%m-%d")
        sym = rec.get("symbol", "")
        by_day_sym.setdefault(day_key, {}).setdefault(sym, []).append(rec)

    for day_key, sym_recs in by_day_sym.items():
        tg_days.add(day_key)
        for sym, recs in sym_recs.items():
            # Pick earliest snapshot if configured, else latest
            if use_earliest_snapshot:
                rec = min(recs, key=lambda r: r.get("ts", 0))
            else:
                rec = max(recs, key=lambda r: r.get("ts", 0))

            features = rec.get("features", {})
            is_top20 = bool(rec.get("label_top20", 0))
            is_top10 = bool(rec.get("label_top10", 0))

            state, btc_ema50 = _tg_features_to_context(features)
            # Infer bull day from BTC context
            is_bull = btc_ema50 > 0.3

            x = extract_context(
                state, mode="trend", tf="15m",
                is_bull_day=is_bull,
                market_regime="bull" if is_bull else "neutral",
                btc_vs_ema50=btc_ema50,
            )

            # For universal samples: train BOTH arms per sample.
            # This creates proper discrimination — the bandit learns
            # which contexts predict top gainers by seeing rewards
            # for both ENTER and SKIP in the same context.
            if is_top20:
                # Top gainer: strong signal that ENTER was correct
                universal_samples.append((x, 1, 1.0))    # ENTER rewarded
                universal_samples.append((x, 0, -0.8))   # SKIP penalized heavily
                n_universal_top += 1
            else:
                # Not top gainer: mild signal that SKIP was correct
                # Weaker penalty for ENTER to maintain high recall
                universal_samples.append((x, 0, 0.10))   # SKIP mildly rewarded
                universal_samples.append((x, 1, -0.12))  # ENTER mildly penalized

    log.info("Universal samples: %d from %d days (%d top gainers)",
             len(universal_samples), len(tg_days), n_universal_top)

    # ── Source 2: Signal samples from critic_dataset ────────────────────────
    try:
        import config as _cfg  # type: ignore
        _max_critic = int(getattr(_cfg, "BANDIT_CRITIC_MAX_RECORDS", 25_000))
    except Exception:
        _max_critic = 25_000
    critic_records = _load_critic_dataset(max_records=_max_critic)
    signal_samples = []
    n_signal_top = 0

    # Group critic by date to find per-day top gainers
    critic_by_date: Dict[str, List[dict]] = {}
    for rec in critic_records:
        ts = rec.get("ts_signal", "")[:10]
        if ts:
            critic_by_date.setdefault(ts, []).append(rec)

    critic_top_ids = set()
    for dt, day_recs in critic_by_date.items():
        scored = [
            (r, r.get("labels", {}).get("ret_10", 0.0) or 0.0)
            for r in day_recs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        for rank, (r, ret) in enumerate(scored):
            if rank < top_n_per_day or ret >= top_gainer_threshold_pct:
                critic_top_ids.add(r.get("id", ""))

    for rec in critic_records:
        f = rec.get("f", {})
        decision = rec.get("decision", {})

        state = {
            "slope_pct": f.get("slope", 0.0),
            "adx": f.get("adx", 20.0),
            "rsi": f.get("rsi", 50.0),
            "vol_x": f.get("vol_x", 1.0),
            "ml_proba": 0.5,
            "daily_range": f.get("daily_range", 3.0),
            "macd_hist": f.get("macd_hist_norm", 0.0),
        }

        mode = rec.get("signal_type", "trend")
        tf = rec.get("tf", "15m")
        is_bull = rec.get("is_bull_day", False)
        btc_ema50 = f.get("btc_vs_ema50", 0.0)

        x = extract_context(
            state, mode=mode, tf=tf,
            is_bull_day=is_bull,
            market_regime="bull" if is_bull else "neutral",
            btc_vs_ema50=btc_ema50,
        )

        action = decision.get("action", "take")
        arm = 1 if action == "take" else 0
        is_top = rec.get("id", "") in critic_top_ids

        if arm == 1:
            reward = 1.0 if is_top else -0.05
        else:
            reward = -1.0 if is_top else 0.0

        signal_samples.append((x, arm, reward))
        if is_top:
            n_signal_top += 1

    log.info("Signal samples: %d (%d top gainers)", len(signal_samples), n_signal_top)

    # ── Combine and train ──────────────────────────────────────────────────
    all_samples = universal_samples + signal_samples
    if len(all_samples) < min_samples:
        log.info("Entry bandit: only %d total samples, need %d", len(all_samples), min_samples)
        return {"status": "skipped", "n_samples": len(all_samples)}

    count = bandit.batch_update(all_samples)
    bandit.save(ENTRY_STATE_FILE)

    arm_names = ["skip", "enter"]
    stats = bandit.arm_stats(arm_names)

    log.info("Entry bandit trained: %d updates (universal=%d + signal=%d), "
             "top_gainers: universal=%d signal=%d, total=%d",
             count, len(universal_samples), len(signal_samples),
             n_universal_top, n_signal_top, bandit.total_updates)
    return {
        "status": "ok",
        "n_samples": count,
        "n_universal_samples": len(universal_samples),
        "n_signal_samples": len(signal_samples),
        "n_universal_top_gainers": n_universal_top,
        "n_signal_top_gainers": n_signal_top,
        "n_days": len(tg_days),
        "total_updates": bandit.total_updates,
        "arm_stats": stats,
    }


# ── Bandit prediction accuracy (backtest) ───────────────────────────────────

def evaluate_bandit_accuracy(n_recent_days: int = 7) -> dict:
    """
    Backtest the current entry bandit on recent top_gainer_dataset records.

    For each day, simulate: would the bandit choose ENTER for actual top gainers?

    Returns dict with recall@20, avg UCB gap, per-day breakdown.
    """
    from contextual_bandit import get_entry_bandit, extract_context

    bandit = get_entry_bandit()
    if bandit.total_updates < 50:
        return {"status": "untrained", "total_updates": bandit.total_updates}

    tg_records = _load_top_gainer_dataset()
    if not tg_records:
        return {"status": "no_data"}

    # Group by (date, symbol) — pick earliest snapshot
    by_day_sym: Dict[str, Dict[str, dict]] = {}
    for rec in tg_records:
        ts_ms = rec.get("ts", 0)
        if not ts_ms:
            continue
        dt = datetime.utcfromtimestamp(ts_ms / 1000)
        day_key = dt.strftime("%Y-%m-%d")
        sym = rec.get("symbol", "")
        if sym not in by_day_sym.get(day_key, {}):
            by_day_sym.setdefault(day_key, {})[sym] = rec
        else:
            existing_ts = by_day_sym[day_key][sym].get("ts", 0)
            if ts_ms < existing_ts:
                by_day_sym[day_key][sym] = rec

    # Keep only most recent N days
    sorted_days = sorted(by_day_sym.keys(), reverse=True)[:n_recent_days]

    daily_results = []
    total_top20 = 0
    total_top20_enter = 0
    all_ucb_gaps_top = []
    all_ucb_gaps_nontop = []

    for day_key in sorted_days:
        sym_recs = by_day_sym[day_key]
        day_top20 = 0
        day_top20_enter = 0

        for sym, rec in sym_recs.items():
            features = rec.get("features", {})
            is_top20 = bool(rec.get("label_top20", 0))

            state, btc_ema50 = _tg_features_to_context(features)
            is_bull = btc_ema50 > 0.3

            x = extract_context(
                state, mode="trend", tf="15m",
                is_bull_day=is_bull,
                market_regime="bull" if is_bull else "neutral",
                btc_vs_ema50=btc_ema50,
            )

            arm, info = bandit.select_arm(x)
            ucbs = info.get("ucbs", [0, 0])
            ucb_gap = ucbs[1] - ucbs[0] if len(ucbs) >= 2 else 0.0

            if is_top20:
                day_top20 += 1
                total_top20 += 1
                if arm == 1:  # ENTER
                    day_top20_enter += 1
                    total_top20_enter += 1
                all_ucb_gaps_top.append(ucb_gap)
            else:
                all_ucb_gaps_nontop.append(ucb_gap)

        recall = day_top20_enter / day_top20 if day_top20 > 0 else 0.0
        daily_results.append({
            "day": day_key,
            "n_symbols": len(sym_recs),
            "n_top20": day_top20,
            "n_top20_enter": day_top20_enter,
            "recall_top20": round(recall, 4),
        })

    overall_recall = total_top20_enter / total_top20 if total_top20 > 0 else 0.0
    avg_ucb_gap_top = sum(all_ucb_gaps_top) / len(all_ucb_gaps_top) if all_ucb_gaps_top else 0.0
    avg_ucb_gap_nontop = sum(all_ucb_gaps_nontop) / len(all_ucb_gaps_nontop) if all_ucb_gaps_nontop else 0.0

    return {
        "status": "ok",
        "n_days": len(daily_results),
        "overall_recall_top20": round(overall_recall, 4),
        "total_top20": total_top20,
        "total_top20_enter": total_top20_enter,
        "avg_ucb_gap_top_gainers": round(avg_ucb_gap_top, 4),
        "avg_ucb_gap_non_top": round(avg_ucb_gap_nontop, 4),
        "ucb_separation": round(avg_ucb_gap_top - avg_ucb_gap_nontop, 4),
        "daily": daily_results,
    }


# ── 2. Trail bandit batch update ────────────────────────────────────────────

def train_trail_bandit(
    experiences: List[TradeExperience],
    *,
    min_samples: int = 20,
) -> dict:
    """
    Batch-train trail_k bandit from closed trade experiences.

    For each trade:
      - Reconstruct context from entry state
      - Map the actual trail_k used to the closest arm
      - Use trade reward as bandit feedback
    """
    from contextual_bandit import (
        get_trail_bandit, extract_context, map_trail_k_to_arm,
        TRAIL_ARMS, STATE_FILE,
    )

    bandit = get_trail_bandit()
    samples = []

    for exp in experiences:
        if not exp.state or exp.reward is None:
            continue

        x = extract_context(
            exp.state,
            mode=exp.mode,
            tf=exp.tf,
            is_bull_day=exp.is_bull_day,
            market_regime=exp.market_regime,
            btc_vs_ema50=exp.btc_vs_ema50,
        )

        arm = getattr(exp, "bandit_arm", None)
        if arm is None:
            base_trail_k = _base_trail_k_for_mode(exp.mode)
            if base_trail_k > 0:
                mult = exp.trail_k / base_trail_k
                arm = map_trail_k_to_arm(mult)
            else:
                arm = 2  # default arm

        samples.append((x, arm, exp.reward))

    if len(samples) < min_samples:
        log.info("Trail bandit: only %d samples, need %d", len(samples), min_samples)
        return {"status": "skipped", "n_samples": len(samples)}

    count = bandit.batch_update(samples)
    bandit.save(STATE_FILE)
    arm_names = [a["name"] for a in TRAIL_ARMS]
    stats = bandit.arm_stats(arm_names)
    log.info("Trail bandit trained: %d updates, total=%d", count, bandit.total_updates)
    return {
        "status": "ok",
        "n_samples": count,
        "total_updates": bandit.total_updates,
        "arm_stats": stats,
    }


# Backward compat alias
def train_bandit(
    experiences: List[TradeExperience],
    *,
    min_samples: int = 20,
) -> dict:
    """Backward compat: trains trail bandit."""
    return train_trail_bandit(experiences, min_samples=min_samples)


# ── 3. Exit policy batch update ──────────────────────────────────────────────

def train_exit_policy(
    experiences: List[TradeExperience],
    *,
    min_samples: int = 20,
) -> dict:
    """
    Batch-train exit Q-learning from trade trajectories.

    For each closed trade, we create training samples:
      - Terminal state (at exit): action depends on exit reason
      - Intermediate states: reconstructed from bars_held progression
    """
    from exit_rl import (
        ExitPolicy, extract_exit_state, get_exit_policy,
        ACTION_HOLD, ACTION_TIGHTEN, ACTION_EXIT,
    )

    policy = get_exit_policy()
    samples = []

    for exp in experiences:
        if not exp.state or exp.reward is None:
            continue

        entry_adx = exp.state.get("adx", 20.0)
        max_hold = exp.max_hold_bars if exp.max_hold_bars > 0 else 16

        # Terminal state at exit
        terminal_state = extract_exit_state(
            current_pnl=exp.pnl_pct,
            bars_held=exp.bars_held,
            max_hold_bars=max_hold,
            rsi=exp.state.get("rsi", 50.0),
            slope=exp.state.get("slope_pct", 0.0),
            adx=exp.state.get("adx", 20.0),
            entry_adx=entry_adx,
            vol_x=exp.state.get("vol_x", 1.0),
            macd_hist=exp.state.get("macd_hist", 0.0),
            is_bull_day=exp.is_bull_day,
            market_regime=exp.market_regime,
            mode=exp.mode,
        )

        terminal_reward = exp.reward

        exit_reason = (exp.exit_reason or "").lower()
        if "atr" in exit_reason or "trail" in exit_reason:
            terminal_action = ACTION_EXIT
        elif "weak" in exit_reason:
            terminal_action = ACTION_TIGHTEN
        else:
            terminal_action = ACTION_EXIT

        samples.append((terminal_state, terminal_action, terminal_reward, None, True))

        # Synthetic intermediate hold states for good trades
        if exp.pnl_pct > 0 and exp.bars_held >= 3:
            for bar_frac in [0.25, 0.50, 0.75]:
                bar_i = max(1, int(exp.bars_held * bar_frac))
                pnl_est = exp.pnl_pct * bar_frac * 0.8
                mid_state = extract_exit_state(
                    current_pnl=pnl_est,
                    bars_held=bar_i,
                    max_hold_bars=max_hold,
                    rsi=exp.state.get("rsi", 50.0),
                    slope=max(0, exp.state.get("slope_pct", 0.1)),
                    adx=exp.state.get("adx", 20.0),
                    entry_adx=entry_adx,
                    vol_x=exp.state.get("vol_x", 1.0),
                    is_bull_day=exp.is_bull_day,
                    market_regime=exp.market_regime,
                    mode=exp.mode,
                )
                samples.append((mid_state, ACTION_HOLD, 0.1, terminal_state, False))

        # Synthetic: early exit on bad trade = good decision
        if exp.pnl_pct < -0.5 and exp.bars_held >= 2:
            early_state = extract_exit_state(
                current_pnl=exp.pnl_pct * 0.3,
                bars_held=max(1, exp.bars_held // 3),
                max_hold_bars=max_hold,
                rsi=exp.state.get("rsi", 50.0),
                slope=min(0, exp.state.get("slope_pct", -0.1)),
                adx=exp.state.get("adx", 20.0) * 0.8,
                entry_adx=entry_adx,
                vol_x=exp.state.get("vol_x", 1.0) * 0.7,
                is_bull_day=exp.is_bull_day,
                market_regime=exp.market_regime,
                mode=exp.mode,
            )
            avoided_loss = abs(exp.pnl_pct) * 0.5
            samples.append((early_state, ACTION_EXIT, avoided_loss, None, True))

    if len(samples) < min_samples:
        log.info("Exit policy training: only %d samples, need %d",
                 len(samples), min_samples)
        return {"status": "skipped", "n_samples": len(samples)}

    result = policy.batch_update(samples)
    policy.save()
    log.info("Exit policy trained: %s", result)
    return {
        "status": "ok",
        **result,
        "weight_summary": policy.weight_summary(),
    }


# ── 4. Full offline training cycle ──────────────────────────────────────────

def run_offline_training(
    *,
    batch_size: int = 256,
    min_new_trades: int = 10,
) -> dict:
    """
    Run full offline RL training cycle.
    Called from rl_agent after enough new trades, or on schedule.

    Trains:
      1. Entry bandit from critic_dataset.jsonl (enter/skip, top gainer reward)
      2. Trail bandit from rl_memory.jsonl (trail_k selection, PnL reward)
      3. Exit policy from rl_memory.jsonl (hold/tighten/exit decisions)
      4. CMA-ES optimizer from rl_memory.jsonl (parameter search)

    Returns summary of all training results.
    """
    state = _load_offline_state()
    experiences = load_experiences()
    n_total_exp = len(experiences)

    results = {}

    # 1. Train entry bandit from top_gainer_dataset + critic_dataset (universal)
    try:
        results["entry_bandit"] = train_entry_bandit()
    except Exception as e:
        log.error("Entry bandit training failed: %s", e)
        results["entry_bandit"] = {"status": "error", "error": str(e)}

    # 1b. Evaluate bandit prediction accuracy on recent data
    try:
        results["bandit_accuracy"] = evaluate_bandit_accuracy(n_recent_days=7)
    except Exception as e:
        log.error("Bandit accuracy evaluation failed: %s", e)
        results["bandit_accuracy"] = {"status": "error", "error": str(e)}

    # Check if enough new rl_memory experiences for trail/exit/cmaes
    n_new = n_total_exp - state.get("last_n_experiences", 0)
    if n_new < min_new_trades:
        log.info("Offline RL: %d new trades, need %d — skipping trail/exit/cmaes",
                 n_new, min_new_trades)
        results["trail_bandit"] = {"status": "skipped", "n_new": n_new}
        results["exit_policy"] = {"status": "skipped", "n_new": n_new}
        results["cmaes"] = {"status": "skipped", "n_new": n_new}
    else:
        batch = sample_batch(experiences, batch_size, mode_balanced=True)
        log.info("Offline RL: training on %d samples (%d new trades)", len(batch), n_new)

        # 2. Train trail bandit
        try:
            results["trail_bandit"] = train_trail_bandit(batch)
        except Exception as e:
            log.error("Trail bandit training failed: %s", e)
            results["trail_bandit"] = {"status": "error", "error": str(e)}

        # 3. Train exit policy
        try:
            results["exit_policy"] = train_exit_policy(batch)
        except Exception as e:
            log.error("Exit policy training failed: %s", e)
            results["exit_policy"] = {"status": "error", "error": str(e)}

        # 4. Run CMA-ES optimizer
        try:
            from rl_optimizer import run_optimization_step
            best_params = run_optimization_step(
                batch_size=batch_size,
                n_generations=10,
                population=20,
            )
            results["cmaes"] = {
                "status": "ok" if best_params else "insufficient_data",
                "n_params": len(best_params) if best_params else 0,
            }
        except Exception as e:
            log.error("CMA-ES training failed: %s", e)
            results["cmaes"] = {"status": "error", "error": str(e)}

    # Update state
    state["last_n_experiences"] = n_total_exp
    state["last_run_ts"] = datetime.now(timezone.utc).isoformat()
    state["runs"] = state.get("runs", 0) + 1
    _save_offline_state(state)

    _log_offline_event(results, n_total_exp, n_new)

    log.info("Offline RL training complete: %s", {
        k: v.get("status", "?") for k, v in results.items()
        if isinstance(v, dict)
    })
    return results


def _log_offline_event(results: dict, n_total: int, n_new: int) -> None:
    eb = results.get("entry_bandit", {})
    ba = results.get("bandit_accuracy", {})
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_total": n_total,
        "n_new": n_new,
        "entry_bandit_status": eb.get("status"),
        "entry_bandit_n_universal": eb.get("n_universal_samples", 0),
        "entry_bandit_n_signal": eb.get("n_signal_samples", 0),
        "entry_bandit_n_top_gainers": eb.get("n_universal_top_gainers", 0),
        "bandit_recall_top20": ba.get("overall_recall_top20"),
        "bandit_ucb_separation": ba.get("ucb_separation"),
        "trail_bandit_status": results.get("trail_bandit", {}).get("status"),
        "exit_status": results.get("exit_policy", {}).get("status"),
        "cmaes_status": results.get("cmaes", {}).get("status"),
    }
    with OFFLINE_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_trail_k_for_mode(mode: str) -> float:
    """Return base trail_k for a signal mode (from config defaults)."""
    try:
        import config
        if mode == "breakout":
            return float(getattr(config, "ATR_TRAIL_K_BREAKOUT", 1.5))
        elif mode == "retest":
            return float(getattr(config, "ATR_TRAIL_K_RETEST", 1.8))
        elif mode in ("strong_trend", "impulse_speed"):
            return float(getattr(config, "ATR_TRAIL_K_STRONG", 2.5))
        else:
            return float(getattr(config, "ATR_TRAIL_K", 2.0))
    except ImportError:
        return 2.0
