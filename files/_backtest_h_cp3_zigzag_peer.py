"""H-CP3 backtest: ZigZag-gated WEAK exit suppression with peer-breadth.

Hypothesis (combined from our roadmap + gpt_crypto_bot P3):
  When bot exits via WEAK RSI-div on a profitable position, two signals
  could refine the suppress decision:
  1. ZigZag still active uptrend in last 20 bars (structural)
  2. Cluster peers are still positive (peer-breadth)

  If BOTH agree, suppress WEAK exit. Use ATR-trail for reversal detection
  instead of pattern-based heuristic.

Window: MAX = 108d bot_events.
Counterfactual: same EOD-return proxy as H-CP1.

Methodology:
  - Filter to WEAK exit events on positions profitable AND mode ∈
    (alignment, trend, strong_trend, retest) (slow trend modes — affected
    most by premature WEAK).
  - For each, check if AT EXIT TIME there was an active ZigZag uptrend
    (using cached klines for last 20 × 15m bars). Requires klines coverage.
  - Compute delta if held to EOD.
  - Also stratify by ml_proba at entry (if available — proxy for cluster
    activity since cluster map not pre-computed per-day).
"""
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(ROOT / "files"))
from zigzag_labeler import detect_uptrends  # noqa: E402

EVENTS_PATH = ROOT / "files" / "bot_events.jsonl"
TG_DATASET_PATH = ROOT / "files" / "top_gainer_dataset.jsonl"
HISTORY_DIR = ROOT / "history"

SLOW_TREND_MODES = {"alignment", "trend", "strong_trend", "retest"}
WEAK_MARKER = "⚠️"


def is_weak_exit(reason: str) -> bool:
    if not reason: return False
    r = reason
    return r.startswith(WEAK_MARKER) or "weak" in r.lower()


def load_klines_bars(sym: str, tf: str = "15m") -> list[dict]:
    """Load cached klines as list[dict]."""
    p = HISTORY_DIR / f"{sym}_{tf}.csv"
    if not p.exists():
        return []
    bars = []
    with io.open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                bars.append({
                    "ts": datetime.fromisoformat(row["ts"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            except Exception:
                continue
    return bars


def zigzag_active_at(sym: str, exit_dt: datetime, lookback: int = 20) -> tuple:
    """At exit time, look back N bars and check ZigZag.

    Returns (has_active_uptrend, info_dict).
    has_active_uptrend = True iff last bar's index >= last detected
    trend's end_idx - 2 (i.e. trend is current or ended within 2 bars).
    """
    bars = load_klines_bars(sym, "15m")
    if not bars:
        return None, {"reason": "no_klines"}

    # Filter to bars ≤ exit_dt + 1 min
    threshold = exit_dt + timedelta(minutes=1)
    bars = [b for b in bars if b["ts"] <= threshold]
    if len(bars) < 10:
        return None, {"reason": "too_few_bars_before_exit"}

    # Look only at last `lookback` bars
    bars = bars[-lookback:]

    trends = detect_uptrends(bars, swing_pct=2.0,
                              max_drawdown_pct=1.5,
                              min_duration_bars=3)
    if not trends:
        return False, {"reason": "no_recent_uptrend", "n_bars": len(bars)}

    last_trend = trends[-1]
    is_active = last_trend.end_idx >= len(bars) - 2
    return is_active, {
        "n_bars": len(bars),
        "last_trend_gain": round(last_trend.gain_pct, 2),
        "last_trend_end_idx": last_trend.end_idx,
        "bars_since_peak": len(bars) - 1 - last_trend.end_idx,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    cut_dt = None
    if args.days > 0:
        cut_dt = datetime.now(timezone.utc) - timedelta(days=args.days)

    # 1) Load eod_return_pct (already in %)
    eod_by: dict = {}
    with io.open(TG_DATASET_PATH, encoding="utf-8") as f:
        for ln in f:
            try: e = json.loads(ln)
            except: continue
            ts_ms = e.get("ts");
            if not ts_ms: continue
            dt = datetime.fromtimestamp(int(ts_ms)/1000, tz=timezone.utc)
            sym = e.get("symbol")
            if not sym: continue
            eod = e.get("eod_return_pct")
            if eod is None: continue
            try: eod = float(eod)
            except: continue
            eod_by[(dt.strftime("%Y-%m-%d"), sym)] = eod

    # 2) Pair entries with exits, filter to slow-trend mode + WEAK exit + profit
    open_pos = {}; pairs = []
    with io.open(EVENTS_PATH, encoding="utf-8") as f:
        for ln in f:
            if '"event"' not in ln: continue
            try: e = json.loads(ln)
            except: continue
            ev = e.get("event","")
            if ev not in ("entry","exit"): continue
            ts = e.get("ts","")
            try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
            except: continue
            if cut_dt and dt < cut_dt: continue
            sym = e.get("sym") or e.get("symbol") or ""
            if not sym: continue
            if ev == "entry":
                ep = float(e.get("price") or e.get("entry_price") or 0)
                if ep <= 0: continue
                open_pos[sym] = {
                    "entry_dt": dt, "entry_price": ep,
                    "mode": e.get("mode") or "?", "tf": e.get("tf") or "?",
                }
            else:
                ent = open_pos.pop(sym, None)
                if not ent: continue
                ex_p = float(e.get("exit_price") or e.get("price") or 0)
                if ex_p <= 0: continue
                pnl = (ex_p - ent["entry_price"]) / ent["entry_price"] * 100
                reason = e.get("reason") or ""
                if not is_weak_exit(reason): continue
                if pnl <= 0: continue
                if ent["mode"] not in SLOW_TREND_MODES: continue
                pairs.append({
                    "sym": sym, "entry_dt": ent["entry_dt"], "exit_dt": dt,
                    "entry_price": ent["entry_price"], "exit_price": ex_p,
                    "pnl_pct": pnl, "mode": ent["mode"], "tf": ent["tf"],
                    "reason": reason,
                })

    print(f"[load] WEAK exits on slow-trend modes with profit: {len(pairs)}")

    # 3) For each, check ZigZag activity at exit time
    print(f"[zigzag] checking ZigZag activity at each exit (slow on first run)…")
    suppressable = []
    not_suppressable = []
    no_data = []
    for i, p in enumerate(pairs, 1):
        if i % 50 == 0:
            print(f"  [{i}/{len(pairs)}] processed")
        active, info = zigzag_active_at(p["sym"], p["exit_dt"], lookback=20)
        d = p["entry_dt"].strftime("%Y-%m-%d")
        eod = eod_by.get((d, p["sym"]))
        p["zigzag_active"] = active
        p["zigzag_info"] = info
        p["eod_pct"] = eod
        if active is None:
            no_data.append(p)
        elif active:
            if eod is not None:
                p["delta_if_held"] = eod - p["pnl_pct"]
            suppressable.append(p)
        else:
            not_suppressable.append(p)

    print(f"\n=== ZigZag-gated WEAK suppression candidates ===")
    print(f"  Total slow-trend WEAK + profit exits: {len(pairs)}")
    print(f"  No klines data:        {len(no_data)}")
    print(f"  WOULD suppress (ZigZag active): {len(suppressable)}")
    print(f"  Keep exit (no ZigZag uptrend): {len(not_suppressable)}")

    # 4) Counterfactual for suppressed
    with_eod = [p for p in suppressable if p.get("delta_if_held") is not None]
    if with_eod:
        avg_delta = sum(p["delta_if_held"] for p in with_eod) / len(with_eod)
        positive = sum(1 for p in with_eod if p["delta_if_held"] > 0)
        sum_delta = sum(p["delta_if_held"] for p in with_eod)
        print(f"\n=== Counterfactual (WOULD-suppress, with EOD data) ===")
        print(f"  Cases: {len(with_eod)}")
        print(f"  Avg delta if held: {avg_delta:+.2f} pp")
        print(f"  Positive delta: {positive} / {len(with_eod)} "
              f"({100*positive/max(1,len(with_eod)):.0f}%)")
        print(f"  Aggregate gain: {sum_delta:+.1f} pp")

        print(f"\n=== Top-10 wins (ZigZag-gated would catch these) ===")
        with_eod.sort(key=lambda p: -p["delta_if_held"])
        print(f"  {'date':<11} {'sym':<10} {'mode/tf':<22} "
              f"{'pnl_exit':>9} {'EOD':>7} {'delta':>7} ZZ_gain")
        for p in with_eod[:10]:
            d = p["exit_dt"].strftime("%m-%d %H:%M")
            zz_g = p["zigzag_info"].get("last_trend_gain", 0)
            print(f"  {d:<11} {p['sym']:<10} {(p['mode']+'/'+p['tf']):<22} "
                  f"{p['pnl_pct']:>+7.2f}% {p['eod_pct']:>+6.2f}% "
                  f"{p['delta_if_held']:>+6.2f}pp  {zz_g}%")

        print(f"\n=== Worst losses (ZigZag misfired — would-have-suppressed but dump) ===")
        with_eod.sort(key=lambda p: p["delta_if_held"])
        for p in with_eod[:5]:
            d = p["exit_dt"].strftime("%m-%d %H:%M")
            zz_g = p["zigzag_info"].get("last_trend_gain", 0)
            print(f"  {d:<11} {p['sym']:<10} {(p['mode']+'/'+p['tf']):<22} "
                  f"{p['pnl_pct']:>+7.2f}% {p['eod_pct']:>+6.2f}% "
                  f"{p['delta_if_held']:>+6.2f}pp  {zz_g}%")

    # Per-mode breakdown
    print(f"\n=== Per-mode breakdown (ZigZag-suppressible WEAK exits) ===")
    by_mode = defaultdict(lambda: {"weak":0,"zz_active":0,"sum_delta":0,"n_delta":0})
    for p in pairs:
        k = f"{p['mode']}/{p['tf']}"
        by_mode[k]["weak"] += 1
        if p.get("zigzag_active"):
            by_mode[k]["zz_active"] += 1
            d = p.get("delta_if_held")
            if d is not None:
                by_mode[k]["sum_delta"] += d
                by_mode[k]["n_delta"] += 1
    print(f"  {'mode/tf':<22} {'weak':>5} {'zz_act':>7} {'avg_delta':>10}")
    for k, v in sorted(by_mode.items(), key=lambda x: -x[1]["weak"]):
        avg = v["sum_delta"] / max(1, v["n_delta"]) if v["n_delta"] else 0
        print(f"  {k:<22} {v['weak']:>5d} {v['zz_active']:>7d} {avg:>+9.2f}pp")


if __name__ == "__main__":
    main()
