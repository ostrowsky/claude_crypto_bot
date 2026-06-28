"""Step 5.3 (decisive): is a fast_reversal GUARD coverage-safe? §4a forbids
enabling the guard unless recall@top20 holds. Sweeps the proba threshold on the
OOS critic holdout (TAKE candidates only — where a guard would act) and reports
the tradeoff: fast-flips cut (benefit) vs watchlist top-20 takes blocked (NS
coverage loss).

Finding (2026-06): the model (holdout AUC 0.636) cannot separate fast-flips from
top-20 winners (shared volatility/overheat features), so EVERY threshold pays
disproportionate coverage — 0.55 cuts 52% of flips but loses 22% of top-20 takes.
=> NOT coverage-safe; do NOT enable the guard, keep FAST_REVERSAL_LEARNING_ENABLED
off. The reward-in-bandit path is separately null (swamped by universal samples).

Read-only.  pyembed\python.exe files\_backtest_fast_reversal_guard.py
"""
from __future__ import annotations
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config
config.FAST_REVERSAL_LEARNING_ENABLED = True   # enable model load only
import offline_rl as orl

model, _ = orl._load_fast_reversal_model()
if model is None:
    print("fast_reversal model missing — run _train_fast_reversal_catboost.py first")
    sys.exit(1)

critic = [r for r in orl._load_critic_dataset(max_records=25000) if r.get("ts_signal")]
critic.sort(key=lambda r: r["ts_signal"])
days = sorted({r["ts_signal"][:10] for r in critic})
cut = days[int(len(days)*0.70)]
takes = [r for r in critic if r["ts_signal"][:10] >= cut
         and r.get("decision", {}).get("action") == "take"]


def proba(r):
    f = r.get("f", {}) or {}
    return orl._fr_predict(model, f, r.get("tf", "15m"), r.get("signal_type", "trend"))


rows = []
for r in takes:
    lab = r.get("labels", {}) or {}
    rows.append((proba(r), 1 if lab.get("label_fast_reversal") == 1 else 0,
                 1 if (lab.get("ret_10", 0.0) or 0.0) >= 3.0 else 0))
nT = len(rows); nflip = sum(x[1] for x in rows); ntop = sum(x[2] for x in rows)
print(f"OOS holdout split {cut} | takes={nT}  fast-flips={nflip} "
      f"({100*nflip/max(1,nT):.0f}%)  top20={ntop}")
print(f"{'thr':>6}{'blocked':>9}{'flips_cut':>11}{'top20_lost':>12}"
      f"{'flip_recall':>13}{'top20_cost':>12}")
for T in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    blk = [x for x in rows if x[0] > T]
    fc = sum(x[1] for x in blk); tl = sum(x[2] for x in blk)
    print(f"{T:>6.2f}{len(blk):>9}{fc:>11}{tl:>12}"
          f"{100*fc/max(1,nflip):>12.0f}%{100*tl/max(1,ntop):>11.0f}%")
print("\nACCEPTANCE (§4a): enable guard only if a threshold cuts flips at ~0 top20")
print("cost. Here every threshold pays disproportionate coverage -> NOT safe.")
