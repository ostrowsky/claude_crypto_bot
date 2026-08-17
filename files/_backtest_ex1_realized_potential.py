"""EX1: realized-to-potential capture metric.

For each paired (entry, exit) on top-20 winners (last 30 d):
  realized_pct  = (exit_price - entry_price) / entry_price * 100

Two modes for `potential`:
  --use-zigzag (Phase D, preferred): potential = matched UpTrend.gain_pct
                from ZigZag labeler run on history/<sym>_15m.csv.
                Fall back to proxy if cache missing.
  default (legacy proxy): potential = max(eod_return_pct, tg_return_4h,
                tg_return_since_open) from top_gainer_dataset snapshots.

  EX1 = clamp(realized_pct / potential_pct, -0.5, 1.5)  if potential_pct > 0

Spec: docs/specs/features/ex1-realized-potential-spec.md (Phase D wiring).
"""
from __future__ import annotations
import argparse, csv, io, json, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# Phase D: optional ZigZag-based potential
_HISTORY_DIR = ROOT / "history"
try:
    sys.path.insert(0, str(ROOT / "files"))
    from zigzag_labeler import detect_uptrends, UpTrend
    _ZIGZAG_AVAILABLE = True
except Exception:
    _ZIGZAG_AVAILABLE = False


def _load_klines_csv(sym: str, tf: str = "15m") -> list[dict]:
    """Load history/<sym>_<tf>.csv produced by _run_signal_evaluator wrapper."""
    path = _HISTORY_DIR / f"{sym}_{tf}.csv"
    if not path.exists():
        return []
    bars = []
    with io.open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                bars.append({
                    "ts": datetime.fromisoformat(row["ts"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low":  float(row["low"]),
                    "close":float(row["close"]),
                    "volume": float(row["volume"]),
                })
            except Exception:
                continue
    return bars


def _zigzag_potential_for_trade(sym: str, entry_dt: datetime, exit_dt: datetime,
                                tf: str = "15m") -> tuple[float | None, dict]:
    """`(gain_pct, diag)` for the UpTrend covering this trade interval.

    Returns a REASON, not just None. A missing kline file, a symbol with no
    detected uptrends, and a trade that falls between uptrends are three
    different facts, and all three used to arrive downstream as "proxy" — which
    is how a coverage gap got attributed to missing data when not one row was
    actually missing a file.

    `nearest_gap_min` is how far the closest uptrend sits from the trade
    interval, in minutes: 0 means overlapping, a large number means the trade
    sat in a stretch the labeler sees no trend in. It is **None** when no
    uptrend exists at all — zero there would read as "adjacent", which is the
    opposite conclusion.
    """
    diag = {"why": "matched", "n_trends": 0, "nearest_gap_min": None}
    bars = _load_klines_csv(sym, tf)
    if not bars:
        diag["why"] = "no_klines"
        return None, diag
    trends = detect_uptrends(bars, symbol=sym,
                             swing_pct=4.0, max_drawdown_pct=2.0,
                             min_duration_bars=4)
    diag["n_trends"] = len(trends)
    if not trends:
        diag["why"] = "no_uptrends"
        return None, diag

    best = None
    nearest = None
    for t in trends:
        gap = max(0.0, (t.start_ts - exit_dt).total_seconds(),
                  (entry_dt - t.end_ts).total_seconds()) / 60.0
        if nearest is None or gap < nearest:
            nearest = gap
        if t.end_ts < entry_dt or t.start_ts > exit_dt:
            continue
        if best is None or abs((t.start_ts - entry_dt).total_seconds()) <                            abs((best.start_ts - entry_dt).total_seconds()):
            best = t
    diag["nearest_gap_min"] = round(nearest, 1) if nearest is not None else None
    if best is None:
        diag["why"] = "no_overlap"
        return None, diag
    return best.gain_pct, diag


# Phase D CLI flag — argparse only kicks in when run directly (not when imported)
if __name__ == "__main__":
    _argp = argparse.ArgumentParser()
    _argp.add_argument("--use-zigzag", action="store_true",
                       help="Phase D: use ZigZag-detected uptrend as `potential` "
                            "(falls back to proxy if klines cache missing).")
    _args, _ = _argp.parse_known_args()
    USE_ZIGZAG = _args.use_zigzag and _ZIGZAG_AVAILABLE
    if _args.use_zigzag and not _ZIGZAG_AVAILABLE:
        print("[ex1] WARNING: --use-zigzag requested but zigzag_labeler "
              "unavailable; falling back to proxy.", file=sys.stderr)
else:
    USE_ZIGZAG = False


def _normalize_pct(v):
    """eod_return_pct in dataset is sometimes decimal, sometimes %.
    Heuristic: if abs > 5 -> already in %; else multiply by 100.
    """
    if v is None: return None
    try: v = float(v)
    except: return None
    return v if abs(v) > 5 else v * 100


# 1) Top-20 + features (potential proxies) per (date, sym)
top20 = set()
potential_proxy = {}  # (d, sym) -> max(eod, tg_4h, tg_since_open)
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
        if e.get("label_top20") == 1:
            top20.add((d, sym))
        # Aggregate potential per (date, sym): take MAX across all snapshots
        feat = e.get("features") or {}
        candidates = []
        eod = _normalize_pct(e.get("eod_return_pct"))
        if eod is not None: candidates.append(eod)
        for k in ("tg_return_4h", "tg_return_since_open", "tg_return_1h"):
            v = feat.get(k)
            if v is not None:
                v_pct = _normalize_pct(v)
                if v_pct is not None: candidates.append(v_pct)
        if candidates:
            cur_max = potential_proxy.get((d, sym))
            new_max = max(candidates)
            potential_proxy[(d, sym)] = max(cur_max, new_max) if cur_max is not None else new_max

# 2) Pair entries with exits
open_t = {}; pairs = []
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event","")
        if ev not in ("entry","exit"): continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        if ev == "entry":
            open_t[sym] = {
                "d": dt.strftime("%Y-%m-%d"),
                "price": float(e.get("price") or e.get("entry_price") or 0),
                "mode": e.get("mode","?"),
                "tf": e.get("tf","?"),
                "entry_dt": dt,
            }
        else:
            ent = open_t.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent["price"] <= 0 or ex_p <= 0: continue
            pnl_pct = (ex_p - ent["price"]) / ent["price"] * 100
            reason = (e.get("reason") or "")
            # Classify exit reason coarsely
            r_lower = reason.lower()
            if "atr" in r_lower or "трейл" in r_lower or "trail" in r_lower:
                exit_class = "atr_trail"
            elif "max_hold" in r_lower or "время" in r_lower or "лимит" in r_lower:
                exit_class = "time_max_hold"
            elif "ema20" in r_lower or "ema 20" in r_lower:
                exit_class = "ema20_weakness"
            elif "rsi" in r_lower:
                exit_class = "rsi"
            elif "macd" in r_lower:
                exit_class = "macd"
            else:
                exit_class = "other"
            # Phase D: try ZigZag potential first if cache available, else proxy
            potential = None
            potential_source = "proxy"
            zz_diag = {"why": "not_attempted", "n_trends": 0,
                       "nearest_gap_min": None}
            if USE_ZIGZAG:
                zz, zz_diag = _zigzag_potential_for_trade(
                    sym, ent["entry_dt"], dt, tf=ent["tf"])
                if zz is not None:
                    potential = zz
                    potential_source = "zigzag"
            if potential is None:
                potential = potential_proxy.get((ent["d"], sym))
            pairs.append({
                "d": ent["d"], "sym": sym, "mode": ent["mode"], "tf": ent["tf"],
                "pnl": pnl_pct, "is_top20": (ent["d"], sym) in top20,
                "potential": potential,
                "potential_source": potential_source,
                "exit_class": exit_class,
                "exit_reason": reason[:60],
                # The trade's own interval, so a rejection can be examined
                # per trade instead of counted in aggregate.
                "entry_ts": ent["entry_dt"].isoformat(),
                "exit_ts": dt.isoformat(),
                "hold_hours": round((dt - ent["entry_dt"]).total_seconds() / 3600.0, 2),
                "zz_why": zz_diag["why"],
                "zz_n_trends": zz_diag["n_trends"],
                "zz_nearest_gap_min": zz_diag["nearest_gap_min"],
            })

print(f"=== EX1: realized-to-potential capture (30 d) ===\n")
print(f"Total paired trades: {len(pairs)}")
print(f"  on top-20 winners: {sum(1 for r in pairs if r['is_top20'])}")
print(f"  with potential data: {sum(1 for r in pairs if r['potential'] is not None)}\n")


def compute_ex1(rows):
    out = []
    for r in rows:
        p = r["potential"]
        if p is None or p <= 0: continue
        ex1 = r["pnl"] / p
        ex1 = max(-0.5, min(1.5, ex1))
        out.append(ex1)
    return out


def stats(values, label):
    if not values:
        print(f"  {label:<32} n=0"); return None
    vs = sorted(values); n = len(vs)
    median = vs[n//2]; mean = sum(vs)/n
    p25 = vs[n//4]; p75 = vs[3*n//4]
    pos = sum(1 for x in vs if x >= 0.5)
    print(f"  {label:<32} n={n:>4}  median={median:+.3f}  mean={mean:+.3f}  "
          f"p25={p25:+.3f}  p75={p75:+.3f}  ex1>=0.5: {100*pos/n:.0f}%")
    return {"n": n, "median": median, "mean": mean, "p25": p25, "p75": p75,
            "share_ex1_ge_05": 100*pos/n}


# ── Overall ──────────────────────────────────────────────────────────
print("Overall (all paired, with potential data):")
top20_pairs = [r for r in pairs if r["is_top20"] and r["potential"] is not None]
non_pairs = [r for r in pairs if not r["is_top20"] and r["potential"] is not None]
s_top20 = stats(compute_ex1(top20_pairs), "top-20 only")
s_non   = stats(compute_ex1(non_pairs),   "non-winners")

# ── Per mode/tf ──────────────────────────────────────────────────────
print(f"\nPer mode/tf (top-20 only):")
groups = defaultdict(list)
for r in top20_pairs:
    groups[f"{r['mode']}/{r['tf']}"].append(r)
for k, rows in sorted(groups.items(), key=lambda x: -len(x[1])):
    if len(rows) < 3: continue
    stats(compute_ex1(rows), k)

# ── Per exit-class ───────────────────────────────────────────────────
print(f"\nPer exit-class (top-20 only):")
exit_groups = defaultdict(list)
for r in top20_pairs:
    exit_groups[r["exit_class"]].append(r)
for k, rows in sorted(exit_groups.items(), key=lambda x: -len(x[1])):
    if len(rows) < 2: continue
    stats(compute_ex1(rows), k)

# ── Worst examples (lowest EX1 on top-20 — biggest left-on-table) ──
print(f"\nWorst 10 cases (top-20, lowest EX1 — most money left on table):")
worst = []
for r in top20_pairs:
    p = r["potential"]
    if p is None or p <= 0: continue
    ex1 = r["pnl"] / p
    ex1 = max(-0.5, min(1.5, ex1))
    worst.append((ex1, r))
worst.sort(key=lambda x: x[0])
print(f"  {'date':<11} {'sym':<10} {'mode/tf':<22} {'pnl':>7} {'pot':>7} {'ex1':>6}  exit_reason")
for ex1, r in worst[:10]:
    print(f"  {r['d']:<11} {r['sym']:<10} {r['mode']+'/'+r['tf']:<22} "
          f"{r['pnl']:>+6.2f}% {r['potential']:>+6.1f}% {ex1:>+5.2f}  {r['exit_reason']}")

# Which measure produced these numbers. The proxy takes the day's intraday
# high as the potential, so every ratio comes out smaller: on the same 30 days
# the proxy reported share_ex1_ge_05 = 0.0% where the canonical zigzag measure
# reported 11.1%. Two materially different numbers under one metric name is
# what TH-02 exists to prevent, and downstream they were indistinguishable.
# Counted over the rows that actually enter the metric (those with potential
# data), not over `pairs` as a whole. The first version read a stray `rows`
# that happened to exist at module scope, so it reported 1 and 1 out of 403 —
# Python raised nothing and the "mixed" verdict was built on noise.
_scored = top20_pairs + non_pairs
_n_zigzag = sum(1 for r in _scored if r.get("potential_source") == "zigzag")
_n_proxy = sum(1 for r in _scored if r.get("potential_source") == "proxy")
metric = {
    "metric": "EX1_realized_potential",
    # Canonical only when every row was measured against a matched uptrend. A
    # trade with no match falls back to the proxy rather than being dropped, so
    # a mixed population is possible and is reported rather than averaged away.
    "potential_source": ("zigzag" if USE_ZIGZAG and _n_proxy == 0
                         else "mixed" if USE_ZIGZAG else "proxy"),
    "n_zigzag": _n_zigzag,
    "n_proxy": _n_proxy,
    # Coverage of the canonical measure, and the top-20 subset it actually
    # covers — a "canonical" number averaged over 36% zigzag and 64% proxy
    # rows is two definitions wearing one name.
    "zigzag_coverage": round(_n_zigzag / max(1, len(_scored)), 4),
    # Why the canonical measure could not be used, per trade rather than in
    # aggregate. Counting "not zigzag" told us the size of the gap and nothing
    # about its cause, and the cause was guessed wrong once already.
    "match_failure": {
        w: sum(1 for r in _scored if r.get("zz_why") == w)
        for w in ("matched", "no_klines", "no_uptrends", "no_overlap")
    },
    "top20_match_failure": {
        w: sum(1 for r in top20_pairs if r.get("zz_why") == w)
        for w in ("matched", "no_klines", "no_uptrends", "no_overlap")
    },
    "top20_zigzag_n": sum(1 for r in top20_pairs
                          if r.get("potential_source") == "zigzag"),
    "top20": s_top20,
    "non_winners": s_non,
}
print("\nMETRIC_JSON:" + json.dumps(metric))
