"""2A: dynamic max_hold. For trades that exited via max_hold (time-based),
check if EOD price was significantly higher than exit price (= we left
money on the table by exiting too early).

Approach:
  - Identify exits with reason ~ "время N баров" or close-to-MAX_HOLD bars.
  - For each such exit on a top-20 winner: compare exit vs eod_return.
  - If eod_return > pnl_at_exit + 1% → opportunity to extend hold.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc); CUT = NOW - timedelta(days=30)

# eod_return + top20
eod_ret = {}; top20 = set()
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts");
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
        eod_ret[(d, sym)] = e.get("eod_return_pct")
        if e.get("label_top20") == 1: top20.add((d, sym))

# Pair entries with exits
entries = {}; pairs = []
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
        d = dt.strftime("%Y-%m-%d")
        if ev == "entry":
            entries[sym] = (dt, d, float(e.get("price") or e.get("entry_price") or 0),
                            e.get("mode","?"))
        else:
            ent = entries.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent[2] <= 0 or ex_p <= 0: continue
            pnl = (ex_p - ent[2]) / ent[2] * 100
            bars = int(e.get("bars_held") or 0)
            reason = (e.get("reason") or "")
            pairs.append({"d": ent[1], "sym": sym, "mode": ent[3],
                          "pnl": pnl, "bars": bars, "reason": reason,
                          "is_top20": (ent[1], sym) in top20,
                          "eod": eod_ret.get((ent[1], sym))})

# Identify exit-too-early candidates
def is_time_exit(r):
    return ("время" in r["reason"]) or ("max_hold" in r["reason"].lower()) or r["bars"] >= 20

print("=== 2A: dynamic max_hold validation ===\n")
print(f"Total paired trades 30d: {len(pairs)}")

time_exits = [r for r in pairs if is_time_exit(r)]
ema_exits = [r for r in pairs if "EMA20" in r["reason"]]
trail_exits = [r for r in pairs if "трейл" in r["reason"] or "trail" in r["reason"].lower()]
print(f"  exit by time/max_hold:    {len(time_exits)}")
print(f"  exit by EMA20 weakness:   {len(ema_exits)}")
print(f"  exit by ATR trail:        {len(trail_exits)}")

# For each time-exit on top-20 winner: compare pnl vs eod_return
left_money_count = 0
not_top20_count = 0
total_money_left = 0.0
samples = []
for r in time_exits:
    if r["eod"] is None: continue
    eod = float(r["eod"])
    if abs(eod) <= 5: eod *= 100  # normalise
    if r["is_top20"] and eod > r["pnl"] + 1.0:
        money_left = eod - r["pnl"]
        total_money_left += money_left
        left_money_count += 1
        if len(samples) < 8:
            samples.append({"d": r["d"], "sym": r["sym"], "mode": r["mode"],
                            "bars": r["bars"], "pnl": r["pnl"], "eod": eod,
                            "left": money_left})

print(f"\nTime-exits on top-20 winners with money_left ≥ 1% of move: {left_money_count}")
print(f"  total money left on table: {total_money_left:+.1f}% (sum of % per trade)")
if samples:
    print(f"\nSamples:")
    for s in samples:
        print(f"  {s['d']}  {s['sym']:<10} {s['mode']:<14} bars={s['bars']:>2}  "
              f"pnl={s['pnl']:+.2f}%  eod={s['eod']:+.1f}%  left={s['left']:+.1f}%")

# Same for EMA exits (path 2B)
ema_left = 0; ema_money = 0.0
for r in ema_exits:
    if r["eod"] is None: continue
    eod = float(r["eod"])
    if abs(eod) <= 5: eod *= 100
    if r["is_top20"] and eod > r["pnl"] + 1.0:
        ema_left += 1
        ema_money += (eod - r["pnl"])

print(f"\n[2B by-product] EMA20-weakness exits on top-20 winners w/ money_left: {ema_left}")
print(f"  total money left: {ema_money:+.1f}%")

# Net: assume capturing 50% of "left money" via dynamic max_hold
expected_capture_lift = (total_money_left * 0.5) / max(1, len(time_exits)) / 100  # in capture-ratio units
print(f"\nEstimated capture-ratio uplift from 2A: +{expected_capture_lift:.3f}")

metric = {
    "metric": "2A_dynamic_max_hold",
    "n_paired": len(pairs),
    "n_time_exits": len(time_exits),
    "n_ema_exits": len(ema_exits),
    "n_trail_exits": len(trail_exits),
    "time_exit_money_left_count": left_money_count,
    "time_exit_money_left_total_pct": total_money_left,
    "ema_exit_money_left_count": ema_left,
    "ema_exit_money_left_total_pct": ema_money,
}
print("\nMETRIC_JSON:" + json.dumps(metric))
