"""Does the "weak-extension overbought" finding transfer to the bot's REAL entries?

The overbought study (_backtest_overbought_continuation.py) found that among all
RSI>=80 episodes, weak extension over MA25 marks near-certain reversals (7.4%
continuation vs 14% base) and suggested using it as an ENTRY filter. This checks
that on the bot's actual take entries instead of on the open market.

Method: every take in critic_dataset with entry features; split the elevated-RSI
population by extension over EMA50 (the closest available analogue of the study's
MA25 extension) and compare forward ret_5, win rate and realized exit pnl.

Read-only.  pyembed\python.exe files\_backtest_weak_extension_entry_filter.py
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent

rows = []
for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"take"' not in ln:
        continue
    try:
        e = json.loads(ln)
    except Exception:
        continue
    if (e.get("decision", {}) or {}).get("action") != "take":
        continue
    f = e.get("f", {}) or {}
    lab = e.get("labels", {}) or {}

    def num(d, k):
        try:
            return float(d[k])
        except (KeyError, TypeError, ValueError):
            return None

    rsi, e50 = num(f, "rsi"), num(f, "close_vs_ema50")
    if rsi is None or e50 is None:
        continue
    rows.append({"rsi": rsi, "e50": e50, "r5": num(lab, "ret_5"),
                 "pnl": num(lab, "trade_exit_pnl")})


def stat(sel, name):
    r5 = [x["r5"] for x in sel if x["r5"] is not None]
    pnl = [x["pnl"] for x in sel if x["pnl"] is not None]
    if len(r5) < 20:
        print(f"{name:<46}{len(sel):>5}   too few resolved")
        return
    print(f"{name:<46}{len(sel):>5}{np.mean(r5):>+9.3f}"
          f"{100*np.mean([v > 0 for v in r5]):>7.0f}%"
          f"{(np.mean(pnl) if pnl else float('nan')):>+9.3f}{len(pnl):>6}")


ext = np.array([x["e50"] for x in rows])
rsi = np.array([x["rsi"] for x in rows])
print("=" * 82)
print(f"Weak-extension entry filter on the bot's own entries  ·  {len(rows)} takes")
print("=" * 82)
print(f"extension over EMA50: median {np.median(ext):.2f}%  p90 {np.quantile(ext,0.9):.2f}%  "
      f"p99 {np.quantile(ext,0.99):.2f}%")
print(f"entries inside the study's effective zone (>8% over the mean): "
      f"{(ext>8).sum()} ({100*(ext>8).mean():.1f}%), of them RSI>=80: {((ext>8)&(rsi>=80)).sum()}")
print(f"entries at RSI>=80 at all: {(rsi>=80).sum()} ({100*(rsi>=80).mean():.1f}%)")
print()
print(f"{'bucket':<46}{'n':>5}{'avg_r5%':>9}{'win':>7}{'avg_pnl%':>9}{'n_pnl':>6}")
stat(rows, "ALL entries (baseline)")
for lo in (70, 75):
    hot = [x for x in rows if x["rsi"] >= lo]
    if len(hot) < 60:
        continue
    e = np.array([x["e50"] for x in hot])
    q1, q2 = np.quantile(e, [1/3, 2/3])
    print(f"--- RSI >= {lo} (n={len(hot)}), terciles of extension at {q1:.2f}% / {q2:.2f}% ---")
    stat([x for x in hot if x["e50"] <= q1], f"  RSI>={lo} WEAK extension (would be blocked)")
    stat([x for x in hot if q1 < x["e50"] <= q2], f"  RSI>={lo} mid")
    stat([x for x in hot if x["e50"] > q2], f"  RSI>={lo} STRONG extension")

# VERDICT (2026-08-07, 4742 takes):
# DOES NOT TRANSFER — do not implement.
#
#   ALL entries (baseline)              4742  r5 -0.022  win 44%  pnl -0.075
#   RSI>=70 WEAK extension (blocked)     445  r5 -0.006  win 44%  pnl -0.098
#   RSI>=70 STRONG extension             445  r5 -0.219  win 39%  pnl +0.054
#
# The bucket the filter would remove performs AT baseline (r5 -0.006 vs -0.022),
# so blocking it costs ~445 neutral entries and buys nothing. If anything the
# gradient runs the other way inside the bot's population: strongly extended
# entries have the worst forward returns (though the trail turns them into the
# best realized pnl, +0.054).
#
# Why the study's effect vanishes here: the bot never operates where the effect
# lived. Its entries sit at a median 1.8% over EMA50 (p90 4.0%), while the
# overbought study's signal was in the top decile of extension, ~8%+ over MA25.
# Only 64 of 4742 entries (1.3%) reach that zone, and exactly ONE of them is at
# RSI>=80 — the existing gates (daily_range, trend_quality, price_edge) already
# cut the extended-overheated population before it can be entered.
#
# General lesson: a market-wide episode study describes a population the gated
# bot does not sample. Always re-test such findings on the bot's own entries
# before turning them into a gate.
