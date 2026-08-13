"""How many rockets CAN be predicted in real time, and how early?

The report says "модель находит 100% ракет, AUC 0.99" while live capture is 7 of
100, and reads that gap as "downstream filters lose them". Measured directly, the
entry bandit fires ENTER on 100% of top-20 rows and on 73.3% of everything else —
lift 1.36. Recall@20 is vacuous at that operating point: it is bought by alerting
on three quarters of the watchlist, so there is no accurate predictor for filters
to "lose". Removing gates would give ~100% recall at ~2% precision (~77 alerts a
day), which earlier work already rejected.

So the honest question is not "which filter to relax" but "what is the ceiling of
a real ranker". This script measures it:

  - one row per (day, symbol): the EARLIEST snapshot of that day, so the decision
    is made as early as the data allows (no intraday hindsight),
  - temporal split (train on older days, evaluate on later ones),
  - CatBoost ranks every watchlist coin per day,
  - recall@N = share of that day's actual top-20 captured if the bot alerted on
    the N highest-ranked coins.

recall@N against the alert budget N is exactly the trade the operator cares
about: N is alerts per day, recall@N is rockets caught.

Read-only.  pyembed\python.exe files\_backtest_rocket_ranking_ceiling.py
"""
from __future__ import annotations
import io, json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, Pool

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))

# earliest snapshot per (day, symbol)
rows: dict[tuple, dict] = {}
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln:
        continue
    try:
        e = json.loads(ln)
    except Exception:
        continue
    ts, sym = e.get("ts"), e.get("symbol")
    if not ts or sym not in WL:
        continue
    d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    key = (d, sym)
    if key not in rows or ts < rows[key]["ts"]:
        rows[key] = {"ts": ts, "d": d, "sym": sym,
                     "f": e.get("features") or {},
                     "y": 1 if e.get("label_top20") == 1 else 0}

recs = sorted(rows.values(), key=lambda r: r["ts"])
if len(recs) < 500:
    print(f"only {len(recs)} rows — not enough"); sys.exit(0)

feat_names = sorted({k for r in recs for k in r["f"].keys()
                     if isinstance(r["f"].get(k), (int, float))})
X = np.array([[float(r["f"].get(k) or 0.0) for k in feat_names] for r in recs])
y = np.array([r["y"] for r in recs])
days = [r["d"] for r in recs]

uniq_days = sorted(set(days))
cut_day = uniq_days[int(len(uniq_days) * 0.70)]
tr = [i for i, d in enumerate(days) if d < cut_day]
ho = [i for i, d in enumerate(days) if d >= cut_day]
print(f"rows={len(recs)}  features={len(feat_names)}  "
      f"train={len(tr)} days<{cut_day}  holdout={len(ho)}")
print(f"base rate (top-20 share of all coin-days): {100*y.mean():.1f}%")

m = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                       loss_function="Logloss", verbose=False, random_seed=42,
                       auto_class_weights="Balanced")
m.fit(Pool(X[tr], y[tr]))
p = m.predict_proba(Pool(X[ho]))[:, 1]


def auc(t, s):
    t = np.asarray(t); s = np.asarray(s)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = (cs - cnt + cs + 1) / 2.0
    r = avg[inv]; np_, nn = int(t.sum()), len(t) - int(t.sum())
    return float("nan") if np_ == 0 or nn == 0 else (r[t == 1].sum() - np_*(np_+1)/2) / (np_*nn)


print(f"holdout AUC: {auc(y[ho], p):.3f}")

by_day: dict[str, list] = defaultdict(list)
for idx, gi in enumerate(ho):
    by_day[days[gi]].append((p[idx], y[gi]))

print()
print(f"{'алертов/день (N)':<20}{'поймано ракет':>16}{'точность':>11}")
total_top = sum(sum(1 for _, yy in v if yy) for v in by_day.values())
for N in (3, 5, 10, 15, 20, 30):
    caught = hits = shots = 0
    for d, v in by_day.items():
        v.sort(key=lambda t: -t[0])
        top_n = v[:N]
        caught += sum(1 for _, yy in top_n if yy)
        hits += sum(1 for _, yy in top_n if yy)
        shots += len(top_n)
    rec = 100*caught/max(1, total_top)
    prec = 100*hits/max(1, shots)
    print(f"{N:<20}{rec:>15.0f}%{prec:>10.0f}%")
print(f"\nвсего ракет в holdout: {total_top} за {len(by_day)} дней "
      f"(~{total_top/max(1,len(by_day)):.1f}/день)")
print("\nrecall@N — сколько ракет поймали бы, отправляя N самых вероятных монет в")
print("день. Это потолок предсказания на данных, которые есть СЕЙЧАС; всё, что")
print("ниже него в бою, теряется на фильтрах и выходах.")
