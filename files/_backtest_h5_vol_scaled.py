"""H-CP1 backtest: volatility-scaled H5 break-even threshold.

Hypothesis (from docs/reports/2026-05-07-ns-hypotheses-roadmap.md):
  TON Trade #2 (2026-05-06): pnl=+0.24% at WEAK exit, mode=impulse_speed/15m.
  Current per-mode H5 threshold is 0.3% — did NOT fire. Coin continued to
  +16%, gave up 14pp of capture.

  Proposed: effective_H5 = base_H5 * clip(5.0 / max(DR, 1.0), 0.5, 1.5)
  On high-vol days (DR=15%), threshold drops to ~0.1%. On quiet days
  (DR=3%), threshold rises to ~0.45%.

Backtest scope: MAX PERIOD = full bot_events.jsonl (108 days).
Methodology:
  1. Pair entries with exits, identify WEAK/EMA-pattern exits on profitable
     trades.
  2. For each such exit, compute:
     - Current H5 verdict (did fire / didn't fire)
     - Vol-scaled H5 verdict at same pnl
  3. Where vol-scaled would NEWLY suppress (current didn't, scaled does):
     - Estimate counterfactual continuation via top_gainer_dataset eod_return_pct
     - Net delta = (eod_return - actual_exit_return) if continued
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = ROOT / "files" / "bot_events.jsonl"
TG_DATASET_PATH = ROOT / "files" / "top_gainer_dataset.jsonl"

# Per-mode base thresholds matching production v2.20.0
PER_MODE_BASE = {
    "impulse_speed": 0.3,
    "impulse": 0.3,
    "trend_surge": 0.3,
    "breakout": 0.3,
    "retest": 0.5,
    "alignment": 0.5,
    "trend": 0.5,
    "strong_trend": 0.7,
}
GLOBAL_BASE = 0.5


def vol_scaled_threshold(base: float, daily_range: float) -> float:
    """clip(5 / max(DR, 1), 0.5, 1.5) — quiet days raise, vol days lower."""
    if daily_range is None or daily_range <= 0:
        return base
    ratio = 5.0 / max(daily_range, 1.0)
    ratio = max(0.5, min(1.5, ratio))
    return base * ratio


SOFT_EMA_MARKERS = (
    "2 закрытия подряд ниже ema20",
    "ниже ema20",
    "ema20 разворачивается",
    "adx ослабевает",
)
WEAK_PREFIX = "⚠️"


def is_h5_candidate_reason(reason: str) -> bool:
    """Soft EMA-pattern or WEAK reason — eligible for H5 suppression in spec."""
    if not reason: return False
    r = reason.lower()
    if r.startswith(WEAK_PREFIX) or "weak" in r:
        return True  # WEAK signals are also in H5 scope
    if "rsi" in r or "macd" in r:
        return False
    if "лимит" in r or "max_hold" in r or "время" in r:
        return False
    if "atr" in r or "трейл" in r:
        return False  # ATR-trail is hard exit, NOT eligible
    return any(m in r for m in SOFT_EMA_MARKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0,
                    help="Filter to last N days (0=ALL available, max period)")
    ap.add_argument("--floor", type=float, default=0.1,
                    help="Hard minimum on scaled threshold")
    args = ap.parse_args()

    cut_dt = None
    if args.days > 0:
        cut_dt = datetime.now(timezone.utc) - timedelta(days=args.days)

    # 1) Load eod_return_pct per (date, sym) from top_gainer_dataset
    eod_by_sym_date: dict = {}
    with io.open(TG_DATASET_PATH, encoding="utf-8") as f:
        for ln in f:
            try: e = json.loads(ln)
            except: continue
            ts_ms = e.get("ts")
            if not ts_ms: continue
            dt = datetime.fromtimestamp(int(ts_ms)/1000, tz=timezone.utc)
            sym = e.get("symbol")
            if not sym: continue
            d = dt.strftime("%Y-%m-%d")
            eod = e.get("eod_return_pct")
            if eod is None: continue
            # Normalize: some rows have % > 5, others decimal < 5
            try: eod = float(eod)
            except: continue
            # NOTE: eod_return_pct is ALREADY in percent throughout dataset
            # (verified via _diag_eod_units 2026-05-07). Do NOT multiply.
            eod_by_sym_date[(d, sym)] = eod

    print(f"[load] eod_return: {len(eod_by_sym_date)} (date, sym) pairs")

    # 2) Stream bot_events, pair entries with exits
    open_pos: dict = {}
    pairs = []
    with io.open(EVENTS_PATH, encoding="utf-8") as f:
        for ln in f:
            if '"event"' not in ln: continue
            try: e = json.loads(ln)
            except: continue
            ev = e.get("event", "")
            if ev not in ("entry", "exit"): continue
            ts = e.get("ts", "")
            try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
            except: continue
            if cut_dt and dt < cut_dt: continue
            sym = e.get("sym") or e.get("symbol") or ""
            if not sym: continue
            if ev == "entry":
                ep = float(e.get("price") or e.get("entry_price") or 0)
                if ep <= 0: continue
                open_pos[sym] = {
                    "entry_dt": dt,
                    "entry_price": ep,
                    "mode": e.get("mode") or e.get("signal_mode") or "?",
                    "tf": e.get("tf") or "?",
                    "daily_range": e.get("daily_range"),
                }
            else:
                ent = open_pos.pop(sym, None)
                if not ent: continue
                ex_p = float(e.get("exit_price") or e.get("price") or 0)
                if ex_p <= 0: continue
                pnl = (ex_p - ent["entry_price"]) / ent["entry_price"] * 100
                pairs.append({
                    "sym": sym,
                    "entry_dt": ent["entry_dt"],
                    "exit_dt": dt,
                    "entry_price": ent["entry_price"],
                    "exit_price": ex_p,
                    "pnl_pct": pnl,
                    "mode": ent["mode"],
                    "tf": ent["tf"],
                    "daily_range_entry": ent.get("daily_range"),
                    "daily_range_exit": e.get("daily_range"),
                    "reason": e.get("reason") or "",
                })

    span_days = (pairs[-1]["exit_dt"] - pairs[0]["entry_dt"]).total_seconds() / 86400 if pairs else 0
    print(f"[load] paired trades: {len(pairs)} over {span_days:.1f}d")

    # 3) Filter to H5-candidate exits AND profitable
    candidates = [p for p in pairs
                  if is_h5_candidate_reason(p["reason"]) and p["pnl_pct"] > 0]
    print(f"[h5] candidate exits (soft EMA/WEAK on profitable trade): {len(candidates)}")

    # 4) Compare schemes
    rows = []
    n_old_fires = n_new_fires = n_new_only = n_lost_fires = 0
    sum_pnl_old_continue = sum_pnl_new_continue = 0.0
    sum_eod_new_only = 0.0
    new_only_cases = []

    for p in candidates:
        mode = p["mode"]; tf = p["tf"]
        base = PER_MODE_BASE.get(mode, GLOBAL_BASE)
        dr = p["daily_range_exit"] or p["daily_range_entry"]
        try: dr_v = float(dr) if dr is not None else None
        except: dr_v = None

        # Current per-mode H5 (would have fired in production v2.20.0)
        old_fires = p["pnl_pct"] >= base

        # Vol-scaled H5
        if dr_v is not None and dr_v > 0:
            scaled = max(args.floor, vol_scaled_threshold(base, dr_v))
        else:
            scaled = base
        new_fires = p["pnl_pct"] >= scaled

        n_old_fires += old_fires
        n_new_fires += new_fires
        if new_fires and not old_fires:
            n_new_only += 1
            # Estimate continuation via eod_return_pct
            d = p["entry_dt"].strftime("%Y-%m-%d")
            eod = eod_by_sym_date.get((d, p["sym"]))
            if eod is not None:
                # Counterfactual: if H5 suppressed exit, position held to EOD
                # Realised would be eod_return_pct (close-of-day vs prev close).
                # Approximation: bot would catch portion of eod move from entry to close
                # vs the realized exit pnl. Net delta = (eod_pct - pnl_pct).
                delta = eod - p["pnl_pct"]
                sum_eod_new_only += delta
                new_only_cases.append({
                    **p, "scaled_threshold": scaled, "eod_pct": eod,
                    "delta_if_held": delta,
                })

        if old_fires and not new_fires:
            n_lost_fires += 1
            d = p["entry_dt"].strftime("%Y-%m-%d")
            eod = eod_by_sym_date.get((d, p["sym"]))
            if eod is not None:
                p["scaled_threshold_lost"] = scaled
                p["eod_pct_lost"] = eod
                p["delta_if_held_lost"] = eod - p["pnl_pct"]

    print(f"\n=== H5 scheme comparison ===")
    print(f"Eligible candidates (soft EMA/WEAK + profit > 0): {len(candidates)}")
    print(f"  Current per-mode H5 fires:  {n_old_fires}  ({100*n_old_fires/max(1,len(candidates)):.1f}%)")
    print(f"  Vol-scaled H5 fires:        {n_new_fires}  ({100*n_new_fires/max(1,len(candidates)):.1f}%)")
    print(f"  NEW-only fires (TON#2-class): {n_new_only}")
    print(f"  LOST fires (vol-scaled too strict on quiet days): {n_lost_fires}")

    if new_only_cases:
        cases_with_eod = [c for c in new_only_cases if c.get("eod_pct") is not None]
        avg_delta = sum(c["delta_if_held"] for c in cases_with_eod) / max(1, len(cases_with_eod))
        positive_delta = sum(1 for c in cases_with_eod if c["delta_if_held"] > 0)
        print(f"\n=== Counterfactual continuation (NEW-only fires) ===")
        print(f"Cases with EOD data: {len(cases_with_eod)} / {len(new_only_cases)}")
        print(f"Avg delta if held: {avg_delta:+.2f} pp")
        print(f"Positive delta (would have made more): {positive_delta} / {len(cases_with_eod)} "
              f"({100*positive_delta/max(1,len(cases_with_eod)):.0f}%)")

        print(f"\nTop-10 biggest wins (vol-scaled WOULD have caught these):")
        cases_with_eod.sort(key=lambda c: -c["delta_if_held"])
        print(f"  {'date':<11} {'sym':<10} {'mode/tf':<22} {'pnl_at_exit':>11} {'DR':>6} {'thr':>6} {'eod':>7} {'delta':>7}")
        for c in cases_with_eod[:10]:
            d = c["entry_dt"].strftime("%m-%d %H:%M")
            print(f"  {d:<11} {c['sym']:<10} {(c['mode']+'/'+c['tf']):<22} "
                  f"{c['pnl_pct']:>+9.2f}% {c['daily_range_exit'] or 0:>5.1f}% "
                  f"{c['scaled_threshold']:>5.2f}% {c['eod_pct']:>+6.2f}% {c['delta_if_held']:>+6.2f}pp")

        print(f"\nTop-5 biggest losses (vol-scaled fired, but actually dumped):")
        cases_with_eod.sort(key=lambda c: c["delta_if_held"])
        for c in cases_with_eod[:5]:
            d = c["entry_dt"].strftime("%m-%d %H:%M")
            print(f"  {d:<11} {c['sym']:<10} {(c['mode']+'/'+c['tf']):<22} "
                  f"{c['pnl_pct']:>+9.2f}% {c['daily_range_exit'] or 0:>5.1f}% "
                  f"{c['scaled_threshold']:>5.2f}% {c['eod_pct']:>+6.2f}% {c['delta_if_held']:>+6.2f}pp")

    # LOST analysis — vol-scaled stricter on quiet days, bot exits earlier
    lost_cases = [c for c in candidates if c.get("delta_if_held_lost") is not None]
    if lost_cases:
        avg_lost_delta = sum(c["delta_if_held_lost"] for c in lost_cases) / len(lost_cases)
        pos_lost = sum(1 for c in lost_cases if c["delta_if_held_lost"] > 0)
        print(f"\n=== LOST fires analysis (vol-scaled stricter on quiet days) ===")
        print(f"  n with eod data: {len(lost_cases)}")
        print(f"  avg foregone delta (would have made more if HELD): {avg_lost_delta:+.2f} pp")
        print(f"  positive delta (LOST = bot exits earlier and misses): {pos_lost} / {len(lost_cases)} "
              f"({100*pos_lost/max(1,len(lost_cases)):.0f}%)")

        # Net effect: gains from new − losses from lost
        if new_only_cases:
            cases_with_eod_g = [c for c in new_only_cases if c.get("eod_pct") is not None]
            sum_new_gain = sum(c["delta_if_held"] for c in cases_with_eod_g)
            sum_lost_foregone = sum(c["delta_if_held_lost"] for c in lost_cases)
            print(f"\n=== Net effect (aggregate pp across 108 d) ===")
            print(f"  + Gain from NEW-only fires    : {sum_new_gain:+.1f} pp aggregate")
            print(f"  − Foregone from LOST fires    : {sum_lost_foregone:+.1f} pp aggregate")
            print(f"  = Net delta                   : {sum_new_gain - sum_lost_foregone:+.1f} pp aggregate")

    # Per-mode breakdown
    print(f"\n=== Per-mode breakdown ===")
    print(f"  {'mode/tf':<22} {'cand':>5} {'old_fires':>10} {'new_fires':>10} {'new_only':>9}")
    by_mode = defaultdict(lambda: {"cand":0,"old":0,"new":0,"new_only":0})
    for p in candidates:
        k = f"{p['mode']}/{p['tf']}"
        base = PER_MODE_BASE.get(p["mode"], GLOBAL_BASE)
        dr = p["daily_range_exit"] or p["daily_range_entry"]
        try: dr_v = float(dr) if dr is not None else None
        except: dr_v = None
        old_f = p["pnl_pct"] >= base
        scaled = max(args.floor, vol_scaled_threshold(base, dr_v) if dr_v else base)
        new_f = p["pnl_pct"] >= scaled
        by_mode[k]["cand"] += 1
        by_mode[k]["old"] += old_f
        by_mode[k]["new"] += new_f
        if new_f and not old_f:
            by_mode[k]["new_only"] += 1

    for k, v in sorted(by_mode.items(), key=lambda x: -x[1]["cand"]):
        if v["cand"] < 3: continue
        print(f"  {k:<22} {v['cand']:>5d} {v['old']:>10d} {v['new']:>10d} {v['new_only']:>9d}")


if __name__ == "__main__":
    main()
