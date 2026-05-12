"""Backfill `label_sustained_uptrend` to top_gainer_dataset.jsonl.

Streams the original dataset, groups by (date, symbol), runs
zigzag_labeler.detect_uptrends on cached intraday klines, applies
the daily label to all snapshots, writes to a parallel v2 file for
side-by-side comparison.

Spec: docs/specs/features/sustained-uptrend-label-spec.md

Usage:
  pyembed/python.exe files/_backfill_sustained_uptrend.py
  pyembed/python.exe files/_backfill_sustained_uptrend.py --days 60
  pyembed/python.exe files/_backfill_sustained_uptrend.py --in-place
"""
from __future__ import annotations
import argparse, csv, io, json, sys
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

DATASET = ROOT / "files" / "top_gainer_dataset.jsonl"
OUT_V2 = ROOT / "files" / "top_gainer_dataset_v2.jsonl"
HISTORY_DIR = ROOT / "history"


def load_klines_for_day(sym: str, date_str: str, tf: str = "15m") -> list[dict]:
    """Load cached klines (history/<sym>_<tf>.csv), filter to UTC date."""
    path = HISTORY_DIR / f"{sym}_{tf}.csv"
    if not path.exists():
        return []
    bars = []
    target_date = date_str  # 'YYYY-MM-DD'
    with io.open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts_str = row.get("ts", "")
            if not ts_str.startswith(target_date):
                continue
            try:
                bars.append({
                    "ts": datetime.fromisoformat(ts_str),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            except Exception:
                continue
    return bars


def compute_label(bars: list[dict], swing_pct: float, drawdown_pct: float,
                  min_duration: int, min_eod_gain_pct: float) -> tuple[int, dict]:
    """Returns (label, debug_info)."""
    if len(bars) < 4:
        return 0, {"reason": "too_few_bars", "n_bars": len(bars)}
    trends = detect_uptrends(bars,
                             swing_pct=swing_pct,
                             max_drawdown_pct=drawdown_pct,
                             min_duration_bars=min_duration)
    if not trends:
        return 0, {"reason": "no_uptrends", "n_bars": len(bars)}
    best = max(trends, key=lambda t: t.gain_pct)
    if best.gain_pct < min_eod_gain_pct:
        return 0, {"reason": "gain_too_small",
                   "best_gain_pct": round(best.gain_pct, 2)}
    last_close = bars[-1]["close"]
    if last_close < best.start_price * 1.02:
        return 0, {"reason": "dumped_by_eod",
                   "best_gain_pct": round(best.gain_pct, 2),
                   "eod_vs_start_pct": round((last_close/best.start_price-1)*100, 2)}
    return 1, {"reason": "sustained",
               "best_gain_pct": round(best.gain_pct, 2),
               "duration_bars": int(best.end_idx - best.start_idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="Only process rows within last N days (skips older)")
    ap.add_argument("--swing-pct", type=float, default=4.0)
    ap.add_argument("--drawdown-pct", type=float, default=2.0)
    ap.add_argument("--min-duration", type=int, default=4)
    ap.add_argument("--min-eod-gain-pct", type=float, default=5.0)
    ap.add_argument("--in-place", action="store_true",
                    help="Write back to top_gainer_dataset.jsonl (DANGER: backup recommended)")
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="Process at most N rows (debug)")
    args = ap.parse_args()

    cut_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
    print(f"[backfill] dataset: {DATASET}")
    print(f"[backfill] window: last {args.days}d (>= {cut_dt.date()})")
    print(f"[backfill] thresholds: swing={args.swing_pct}% drawdown={args.drawdown_pct}% "
          f"min_duration={args.min_duration} min_eod_gain={args.min_eod_gain_pct}%")

    # Pass 1: collect (date, sym) pairs needing label
    needed: dict[tuple[str, str], int] = {}
    n_total = 0
    n_in_window = 0
    with io.open(DATASET, encoding="utf-8") as f:
        for ln in f:
            n_total += 1
            try:
                e = json.loads(ln)
            except Exception:
                continue
            ts_ms = e.get("ts")
            if not ts_ms:
                continue
            dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
            if dt < cut_dt:
                continue
            sym = e.get("symbol")
            if not sym:
                continue
            d = dt.strftime("%Y-%m-%d")
            needed[(d, sym)] = needed.get((d, sym), 0) + 1
            n_in_window += 1
            if args.limit_rows and n_total >= args.limit_rows:
                break

    print(f"[backfill] scanned {n_total} rows total, {n_in_window} in window, "
          f"{len(needed)} unique (date,sym) pairs")

    # Pass 2: compute label per (date, sym)
    labels: dict[tuple[str, str], int] = {}
    debug_info: dict[tuple[str, str], dict] = {}
    n_labeled = n_no_klines = 0
    for i, ((d, sym), _) in enumerate(needed.items(), 1):
        bars = load_klines_for_day(sym, d, tf="15m")
        if not bars:
            n_no_klines += 1
            labels[(d, sym)] = 0
            debug_info[(d, sym)] = {"reason": "no_klines"}
            continue
        lbl, info = compute_label(bars,
                                  swing_pct=args.swing_pct,
                                  drawdown_pct=args.drawdown_pct,
                                  min_duration=args.min_duration,
                                  min_eod_gain_pct=args.min_eod_gain_pct)
        labels[(d, sym)] = lbl
        debug_info[(d, sym)] = info
        n_labeled += 1
        if i % 200 == 0:
            n_pos = sum(1 for v in labels.values() if v == 1)
            print(f"  [{i}/{len(needed)}] labeled, {n_pos} positives so far")

    n_pos = sum(1 for v in labels.values() if v == 1)
    print(f"[backfill] labeling complete: {n_labeled} labeled, "
          f"{n_no_klines} no_klines, {n_pos} positives "
          f"({100*n_pos/max(1,n_labeled):.1f}%)")

    # Cross-tabulation: vs label_top20
    top20_set = set()
    with io.open(DATASET, encoding="utf-8") as f:
        for ln in f:
            try: e = json.loads(ln)
            except: continue
            ts_ms = e.get("ts");
            if not ts_ms: continue
            dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
            if dt < cut_dt: continue
            if e.get("label_top20") == 1:
                sym = e.get("symbol")
                d = dt.strftime("%Y-%m-%d")
                top20_set.add((d, sym))

    labeled_set = set(k for k, v in labels.items() if v == 1)
    both = top20_set & labeled_set
    only_top20 = top20_set - labeled_set
    only_sustained = labeled_set - top20_set
    print(f"\n[crosstab] (date, sym) pairs:")
    print(f"  label_top20=1 AND label_sustained=1: {len(both)} (both)")
    print(f"  label_top20=1 AND label_sustained=0: {len(only_top20)} ← pump+dump filtered out")
    print(f"  label_top20=0 AND label_sustained=1: {len(only_sustained)} ← sustained but not top-20")
    print(f"  label_top20 total: {len(top20_set)}")
    print(f"  label_sustained total: {len(labeled_set)}")
    if top20_set:
        print(f"  Pump-and-dump rate (top20=1 → sustained=0): "
              f"{100*len(only_top20)/len(top20_set):.1f}%")

    # Sample dropped pump-and-dumps
    if only_top20:
        print(f"\n[sample] First 10 pump+dumps (top20=1, sustained=0):")
        for k in list(only_top20)[:10]:
            d, sym = k
            info = debug_info.get(k, {})
            print(f"  {d}  {sym:<14}  {info}")

    # Pass 3: write v2 file
    out_path = DATASET if args.in_place else OUT_V2
    print(f"\n[backfill] writing {out_path} ...")
    n_written = 0
    with io.open(DATASET, encoding="utf-8") as src, \
         io.open(out_path, "w", encoding="utf-8") as dst:
        for ln in src:
            try:
                e = json.loads(ln)
            except Exception:
                dst.write(ln); n_written += 1; continue
            ts_ms = e.get("ts")
            if not ts_ms:
                dst.write(ln); n_written += 1; continue
            dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
            sym = e.get("symbol")
            if dt >= cut_dt and sym:
                d = dt.strftime("%Y-%m-%d")
                lbl = labels.get((d, sym))
                if lbl is not None:
                    e["label_sustained_uptrend"] = int(lbl)
            dst.write(json.dumps(e) + "\n"); n_written += 1
    print(f"[backfill] wrote {n_written} rows to {out_path}")


if __name__ == "__main__":
    main()
