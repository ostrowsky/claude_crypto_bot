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
    """Load the MOST RECENT records from top_gainer_dataset.jsonl.

    This used to `break` after the first `max_records` lines, which on a
    118_625-line file meant the bandit trained on the OLDEST 50k rows and its
    data window sat frozen at 2026-06-05 for 69 days. Keeping a bounded tail
    (deque) costs the same single pass and always ends at today.
    """
    if not TOP_GAINER_DATASET_FILE.exists():
        return []
    from collections import deque
    keep: deque = deque(maxlen=max(1, int(max_records)))
    with TOP_GAINER_DATASET_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("features") and rec.get("symbol"):
                keep.append(rec)
    return list(keep)


def _forward_move_pct(rec: dict) -> Optional[float]:
    """Move still AHEAD of the snapshot, in percent, or None if unknowable.

    `eod_return_pct` covers open->close; `tg_return_since_open` covers
    open->snapshot. What an ENTER decision can still capture is the ratio of
    the two. This is the only part of the day the bandit can actually earn.
    """
    f = rec.get("features") or {}
    since_open = f.get("tg_return_since_open")
    eod = rec.get("eod_return_pct")
    if not isinstance(since_open, (int, float)) or not isinstance(eod, (int, float)):
        return None
    den = 1.0 + float(since_open) / 100.0
    if den <= 0.01:
        return None
    return ((1.0 + float(eod) / 100.0) / den - 1.0) * 100.0


def _label_params() -> dict:
    """Entry-label settings, read once so training and evaluation agree.

    They must share this: if the bandit is trained on the remaining move but
    graded against `label_top20`, the fix looks like a regression.
    """
    try:
        import config as _cfg
    except Exception:                                    # pragma: no cover
        _cfg = None

    def g(name, default):
        return getattr(_cfg, name, default) if _cfg else default

    return {
        "forward": bool(g("BANDIT_FORWARD_REWARD_ENABLED", True)),
        "top_n": int(g("BANDIT_FORWARD_TOP_N", 10)),
        "min_pct": float(g("BANDIT_FORWARD_MIN_PCT", 3.0)),
        "eps": float(g("BANDIT_DECIDED_EPS_PCT", 0.5)),
        "rebuild": bool(g("BANDIT_REBUILD_ON_TRAIN", True)),
        "max_records": int(g("BANDIT_TG_MAX_RECORDS", 50_000)),
    }


def _build_universal_rows(tg_records: List[dict], lp: dict, *,
                          use_earliest_snapshot: bool = True) -> Tuple[List[dict], int]:
    """One graded row per (day, symbol): {day, x, positive}.

    Positive = among the day's top-N by the move STILL AHEAD of the snapshot,
    and that move clears the floor. Rank alone would mint exactly N winners a
    day whatever the market did (measured: lift 1.02x, ENTER on 98% of rows);
    the floor is what makes the label mean anything.

    Returns (rows, n_dropped_decided).
    """
    from contextual_bandit import extract_context

    by_day_sym: Dict[str, Dict[str, list]] = {}
    for rec in tg_records:
        ts_ms = rec.get("ts", 0)
        if not ts_ms:
            continue
        day_key = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
        by_day_sym.setdefault(day_key, {}).setdefault(rec.get("symbol", ""), []).append(rec)

    rows: List[dict] = []
    n_dropped = 0
    for day_key, sym_recs in by_day_sym.items():
        picked: List[dict] = []
        for _sym, recs in sym_recs.items():
            rec = (min(recs, key=lambda r: r.get("ts", 0)) if use_earliest_snapshot
                   else max(recs, key=lambda r: r.get("ts", 0)))
            if lp["forward"]:
                # A snapshot taken after the day resolved has the whole move in
                # its own features; there was never a decision to grade.
                if _day_already_decided(rec, lp["eps"]):
                    n_dropped += 1
                    continue
                fwd = _forward_move_pct(rec)
                if fwd is None:
                    continue
                rec = dict(rec, _fwd_pct=fwd)
            picked.append(rec)

        if lp["forward"]:
            ranked = sorted(picked, key=lambda r: -r["_fwd_pct"])
            winners = {id(r) for r in ranked[:max(0, lp["top_n"])]
                       if r["_fwd_pct"] >= lp["min_pct"]}
        else:
            winners = {id(r) for r in picked if bool(r.get("label_top20", 0))}

        for rec in picked:
            state, btc_ema50 = _tg_features_to_context(rec.get("features", {}))
            is_bull = btc_ema50 > 0.3
            rows.append({
                "day": day_key,
                "x": extract_context(state, mode="trend", tf="15m",
                                     is_bull_day=is_bull,
                                     market_regime="bull" if is_bull else "neutral",
                                     btc_vs_ema50=btc_ema50),
                "positive": id(rec) in winners,
            })
    return rows, n_dropped


def _universal_samples_from_rows(rows: List[dict]) -> List[tuple]:
    """Both arms per row, so the bandit sees ENTER and SKIP in one context."""
    out: List[tuple] = []
    for r in rows:
        if r["positive"]:
            out.append((r["x"], 1, 1.0))     # ENTER rewarded
            out.append((r["x"], 0, -0.8))    # SKIP penalised heavily
        else:
            out.append((r["x"], 0, 0.10))    # SKIP mildly rewarded
            out.append((r["x"], 1, -0.12))   # ENTER mildly penalised
    return out


def _day_already_decided(rec: dict, eps_pct: float) -> bool:
    """True when the snapshot is the EOD resolution of a day already over.

    Such a row carries no decision — the whole move is in the features and in
    the label at once — so training on it teaches hindsight, not prediction.
    """
    f = rec.get("features") or {}
    since_open = f.get("tg_return_since_open")
    eod = rec.get("eod_return_pct")
    if not isinstance(since_open, (int, float)) or not isinstance(eod, (int, float)):
        return True
    return abs(float(eod) - float(since_open)) < eps_pct


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

# ── Step 5.2: fast-reversal proba feature/reward helpers ────────────────────
_FR_FEATS = [
    "close_vs_ema20", "close_vs_ema50", "close_vs_ema200", "ema20_vs_ema50",
    "ema50_vs_ema200", "slope", "rsi", "adx", "vol_x", "macd_hist_norm",
    "atr_pct", "daily_range", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "btc_vs_ema50", "btc_momentum_4h", "market_vol_24h",
]


def _load_fast_reversal_model():
    """Return (model, enabled). Fails OPEN: any error -> (None, False) so the
    bandit trains exactly as before (feature inert, reward term dormant)."""
    try:
        import config as _cfg
        if not bool(getattr(_cfg, "FAST_REVERSAL_LEARNING_ENABLED", False)):
            return None, False
        from catboost import CatBoostClassifier
        path = Path(getattr(_cfg, "FAST_REVERSAL_MODEL_FILE", "fast_reversal_catboost.cbm"))
        if not path.exists():
            log.warning("fast_reversal model missing (%s) — feature stays 0", path.name)
            return None, True
        m = CatBoostClassifier()
        m.load_model(str(path))
        return m, True
    except Exception as e:
        log.warning("fast_reversal model load failed: %s", e)
        return None, False


def _fr_predict(model, f: dict, tf: str, mode: str) -> float:
    if model is None:
        return 0.0
    try:
        row = [float(f.get(k, 0.0) or 0.0) for k in _FR_FEATS] + [str(tf), str(mode)]
        return float(model.predict_proba([row])[0][1])
    except Exception:
        return 0.0


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

    lp = _label_params()
    fwd_reward_on = lp["forward"]
    fwd_top_n = lp["top_n"]
    fwd_min_pct = lp["min_pct"]
    rebuild = lp["rebuild"]

    if rebuild:
        # LinUCB's A/b are sums over the whole history, so batch_update on the
        # saved state would carry the old (leaky) label forever — and re-adding
        # the same rows each run had inflated it to 8.39M updates from ~44.6k
        # unique samples. Rebuild from scratch instead, keeping a one-time
        # backup of what the leaky label produced.
        import contextual_bandit as _cb
        backup = ENTRY_STATE_FILE.with_suffix(".pre_leakfix.json")
        if ENTRY_STATE_FILE.exists() and not backup.exists():
            backup.write_bytes(ENTRY_STATE_FILE.read_bytes())
            log.warning("Entry bandit: pre-leakfix state backed up -> %s", backup.name)
        from contextual_bandit import LinUCBBandit as _LinUCB, N_FEATURES as _NF
        bandit = _LinUCB(n_arms=N_ENTRY_ARMS, n_features=_NF, alpha=2.0)
        # the live process holds the old matrix in memory; drop it so the
        # rebuilt state is what actually decides
        _cb._entry_bandit = bandit
    else:
        bandit = get_entry_bandit()
    _fr_model, _fr_on = _load_fast_reversal_model()
    try:
        import config as _cfg_fr
        _FR_PEN = float(getattr(_cfg_fr, "FAST_REVERSAL_ENTER_PENALTY", -0.6))
        _FR_BON = float(getattr(_cfg_fr, "FAST_REVERSAL_SKIP_BONUS", 0.30))
    except Exception:
        _FR_PEN, _FR_BON = -0.6, 0.30
    n_fr_pos = 0

    # ── Source 1: Universal samples from top_gainer_dataset ─────────────────
    tg_records = _load_top_gainer_dataset(lp["max_records"])
    universal_rows, n_dropped_decided = _build_universal_rows(
        tg_records, lp, use_earliest_snapshot=use_earliest_snapshot)
    universal_samples = _universal_samples_from_rows(universal_rows)
    n_universal_top = sum(1 for r in universal_rows if r["positive"])
    tg_days = {r["day"] for r in universal_rows}

    n_universal_rows = len(universal_samples) // 2
    universal_base_rate = (n_universal_top / n_universal_rows) if n_universal_rows else 0.0
    # §0a rule 1: the positive rate travels with the count, so no reader can
    # mistake "the bandit learned N winners" for skill.
    log.info("Universal samples: %d rows from %d days (%d positive = %.1f%% base "
             "rate, %d dropped as already-decided, label=%s)",
             n_universal_rows, len(tg_days), n_universal_top,
             100.0 * universal_base_rate, n_dropped_decided,
             f"forward top-{fwd_top_n}/>={fwd_min_pct:g}%" if fwd_reward_on else "label_top20")

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

        fr_proba = _fr_predict(_fr_model, f, tf, mode) if _fr_on else 0.0
        x = extract_context(
            state, mode=mode, tf=tf,
            is_bull_day=is_bull,
            market_regime="bull" if is_bull else "neutral",
            btc_vs_ema50=btc_ema50,
            fast_reversal_proba=fr_proba,
        )

        action = decision.get("action", "take")
        arm = 1 if action == "take" else 0
        is_top = rec.get("id", "") in critic_top_ids

        if arm == 1:
            reward = 1.0 if is_top else -0.05
        else:
            reward = -1.0 if is_top else 0.0

        # Step 5.2 (§4a): asymmetric fast-reversal term — penalise ENTER into a
        # realised fast-flip, reward SKIP of one. Only on labelled critic records.
        if _fr_on and (rec.get("labels", {}) or {}).get("label_fast_reversal") == 1:
            reward += _FR_PEN if arm == 1 else _FR_BON
            reward = float(max(-1.5, min(1.5, reward)))
            n_fr_pos += 1

        signal_samples.append((x, arm, reward))
        if is_top:
            n_signal_top += 1

    log.info("Signal samples: %d (%d top gainers)", len(signal_samples), n_signal_top)
    if _fr_on:
        log.info("Fast-reversal (§4a): model=%s, %d fast-flip-labelled samples got reward term",
                 "loaded" if _fr_model is not None else "MISSING", n_fr_pos)

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
        # leak-fix bookkeeping — a positive count is meaningless without these
        "label": (f"forward_top{fwd_top_n}_min{fwd_min_pct:g}" if fwd_reward_on
                  else "label_top20"),
        "n_universal_rows": n_universal_rows,
        "n_dropped_decided": n_dropped_decided,
        "universal_base_rate": round(universal_base_rate, 4),
        "rebuilt_from_scratch": rebuild,
        "tg_window": (f"{min(tg_days)}..{max(tg_days)}" if tg_days else ""),
    }


# ── Bandit prediction accuracy (backtest) ───────────────────────────────────

def _binary_policy_ratio_context(*, total_rows: int, total_enter: int,
                                 total_positive: int,
                                 true_positive_enter: int) -> dict:
    """Return the minimum context required to interpret policy recall."""
    recall = true_positive_enter / total_positive if total_positive else 0.0
    action_rate = total_enter / total_rows if total_rows else 0.0
    base_rate = total_positive / total_rows if total_rows else 0.0
    precision = true_positive_enter / total_enter if total_enter else 0.0
    return {
        "recall": round(recall, 4),
        "action_rate": round(action_rate, 4),
        "base_rate": round(base_rate, 4),
        "precision": round(precision, 4),
        "recall_lift": round(recall / action_rate, 4) if action_rate else 0.0,
        "precision_lift": round(precision / base_rate, 4) if base_rate else 0.0,
    }

def _score_policy(bandit, rows: List[dict]) -> dict:
    """Run a bandit over graded rows; ratio context travels with the recall."""
    total = enter = pos = tp = 0
    gaps_pos: List[float] = []
    gaps_neg: List[float] = []
    per_day: Dict[str, List[int]] = {}
    for r in rows:
        arm, info = bandit.select_arm(r["x"])
        ucbs = info.get("ucbs", [0, 0])
        gap = ucbs[1] - ucbs[0] if len(ucbs) >= 2 else 0.0
        total += 1
        d = per_day.setdefault(r["day"], [0, 0, 0, 0])   # rows, enter, pos, tp
        d[0] += 1
        if arm == 1:
            enter += 1
            d[1] += 1
        if r["positive"]:
            pos += 1
            d[2] += 1
            gaps_pos.append(gap)
            if arm == 1:
                tp += 1
                d[3] += 1
        else:
            gaps_neg.append(gap)

    ratio = _binary_policy_ratio_context(total_rows=total, total_enter=enter,
                                         total_positive=pos, true_positive_enter=tp)
    g_pos = sum(gaps_pos)/len(gaps_pos) if gaps_pos else 0.0
    g_neg = sum(gaps_neg)/len(gaps_neg) if gaps_neg else 0.0
    return {
        "n_days": len(per_day),
        "total_rows": total,
        "total_enter": enter,
        "total_positive": pos,
        "total_positive_enter": tp,
        "recall": ratio["recall"],
        "action_rate": ratio["action_rate"],
        "base_rate": ratio["base_rate"],
        "precision": ratio["precision"],
        "lift": ratio["recall_lift"],
        "precision_lift": ratio["precision_lift"],
        "avg_ucb_gap_top_gainers": round(g_pos, 4),
        "avg_ucb_gap_non_top": round(g_neg, 4),
        "ucb_separation": round(g_pos - g_neg, 4),
        "daily": [
            {"day": day, "n_symbols": v[0], "n_enter": v[1], "n_top20": v[2],
             "n_top20_enter": v[3],
             "recall_top20": round(v[3]/v[2], 4) if v[2] else 0.0,
             "action_rate": round(v[1]/v[0], 4) if v[0] else 0.0}
            for day, v in sorted(per_day.items(), reverse=True)
        ],
    }


def evaluate_bandit_accuracy(n_recent_days: int = 7) -> dict:
    """Grade the entry bandit on the LAST n_recent_days, out of sample.

    The headline numbers come from a bandit trained only on the days BEFORE the
    evaluation window, so they are a genuine time holdout rather than a
    post-fit echo — the live bandit has already seen every row of this dataset,
    and reading its own training data back was what produced "recall@20 = 100%"
    while ENTER fired on 73% of everything.

    The live bandit's in-sample numbers are still returned under
    `in_sample_post_fit`, because the gap between the two IS the diagnostic
    (§0 rule 3) and collapsing them into one number hides it.

    Grading uses the same label as training (`_build_universal_rows`), so the
    two cannot drift apart.
    """
    from contextual_bandit import get_entry_bandit, LinUCBBandit, N_FEATURES

    live = get_entry_bandit()
    if live.total_updates < 50:
        return {"status": "untrained", "total_updates": live.total_updates}

    lp = _label_params()
    tg_records = _load_top_gainer_dataset(lp["max_records"])
    if not tg_records:
        return {"status": "no_data"}

    rows, _dropped = _build_universal_rows(tg_records, lp)
    if not rows:
        return {"status": "no_data"}

    days = sorted({r["day"] for r in rows}, reverse=True)
    eval_days = set(days[:n_recent_days])
    train_rows = [r for r in rows if r["day"] not in eval_days]
    eval_rows = [r for r in rows if r["day"] in eval_days]

    in_sample = _score_policy(live, eval_rows)

    out: dict = {
        "status": "ok",
        "label": (f"forward_top{lp['top_n']}_min{lp['min_pct']:g}" if lp["forward"]
                  else "label_top20"),
        "in_sample_post_fit": dict(in_sample, evaluation_scope="in_sample_post_fit",
                                   diagnostic_only=True),
    }

    if len(train_rows) < 200 or not eval_rows:
        # Not enough earlier days to train an honest holdout — say so rather
        # than quietly promoting the in-sample figure (§0a rule 10).
        out.update(in_sample)
        out.update({
            "evaluation_scope": "insufficient_history_for_holdout",
            "diagnostic_only": True,
            "n_train_rows": len(train_rows),
            "overall_recall_top20": in_sample["recall"],
            "total_top20": in_sample["total_positive"],
            "total_top20_enter": in_sample["total_positive_enter"],
            "total_non_top_enter": in_sample["total_enter"] - in_sample["total_positive_enter"],
        })
        return out

    holdout = LinUCBBandit(n_arms=2, n_features=N_FEATURES, alpha=live.alpha)
    holdout.batch_update(_universal_samples_from_rows(train_rows))
    oos = _score_policy(holdout, eval_rows)

    out.update(oos)
    out.update({
        "evaluation_scope": "out_of_sample_time_holdout",
        "diagnostic_only": False,
        "n_train_rows": len(train_rows),
        "train_days": len({r["day"] for r in train_rows}),
        # kept for backwards compatibility with existing report fields
        "overall_recall_top20": oos["recall"],
        "total_top20": oos["total_positive"],
        "total_top20_enter": oos["total_positive_enter"],
        "total_non_top_enter": oos["total_enter"] - oos["total_positive_enter"],
    })
    return out


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
        "bandit_evaluation_scope": ba.get("evaluation_scope"),
        "bandit_action_rate": ba.get("action_rate"),
        "bandit_top20_base_rate": ba.get("base_rate"),
        "bandit_recall_lift": ba.get("lift"),
        "bandit_precision": ba.get("precision"),
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
