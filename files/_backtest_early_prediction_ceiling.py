"""Can rockets be predicted EARLY, before the move — and how many?

Follows CLAUDE.md §0a. The previous attempt scored AUC 0.99 / recall@5 98% and
was not reportable: it used the earliest available snapshot of each day (often
06:00 UTC) with `tg_return_since_open` among the inputs, while the label is "the
day's return landed in the top-20". That model confirms a move already under way
(rule 2), which is the opposite of the product goal.

This one is built so the question is honest:

  * decision point = the 00 UTC snapshot only. The day has just started, so
    `tg_return_since_open` is ~0 for everyone and cannot carry the answer.
  * two feature sets, reported side by side (rule 2):
      ALL        — everything available at that hour
      NO-RETURN  — same-day/recent return features dropped outright
                   (return_since_open, return_1h/4h, vs_btc_1h/4h,
                    daily_range_pct, range_position)
  * temporal split by day, train on the older days, evaluate on the later ones
    (rule 3).
  * results as recall@N vs an alert budget of N coins per day, always next to
    the base rate and precision (rule 1) — N alerts/day is the cost, recall@N is
    the rockets caught.

Read-only.  pyembed\python.exe files\_backtest_early_prediction_ceiling.py
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

DECISION_HOUR = 0          # UTC — the day has just started
RETURN_FEATURES = {        # anything that already measures today's move
    "tg_return_since_open", "tg_return_1h", "tg_return_4h",
    "tg_vs_btc_1h", "tg_vs_btc_4h", "tg_daily_range_pct", "tg_range_position",
}

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
    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
    if dt.hour != DECISION_HOUR:
        continue
    key = (dt.strftime("%Y-%m-%d"), sym)
    if key not in rows or ts < rows[key]["ts"]:
        rows[key] = {"ts": ts, "d": key[0], "sym": sym,
                     "f": e.get("features") or {},
                     "y": 1 if e.get("label_top20") == 1 else 0}

recs = sorted(rows.values(), key=lambda r: r["ts"])
if len(recs) < 500:
    print(f"only {len(recs)} rows at {DECISION_HOUR:02d} UTC — not enough"); sys.exit(0)

names_all = sorted({k for r in recs for k in r["f"].keys()
                    if isinstance(r["f"].get(k), (int, float))})
names_nr = [k for k in names_all if k not in RETURN_FEATURES]
y = np.array([r["y"] for r in recs])
days = [r["d"] for r in recs]
uniq = sorted(set(days))
cut = uniq[int(len(uniq) * 0.70)]
tr = [i for i, d in enumerate(days) if d < cut]
ho = [i for i, d in enumerate(days) if d >= cut]

# sanity: is the day really untouched at the decision point? (rule 2 evidence)
rso = [r["f"].get("tg_return_since_open") for r in recs
       if isinstance(r["f"].get("tg_return_since_open"), (int, float))]
print("=" * 74)
print(f"Ранний прогноз ракет · снимок {DECISION_HOUR:02d}:xx UTC · "
      f"{len(recs)} строк, {len(uniq)} дней")
print("=" * 74)
if rso:
    print(f"проверка утечки: |return_since_open| в этот час — медиана "
          f"{np.median(np.abs(rso)):.2f}%, p90 {np.percentile(np.abs(rso), 90):.2f}% "
          f"(день ещё не начался — признак не может нести ответ)")
print(f"train: дни < {cut} ({len(tr)} строк) · holdout: {len(ho)} строк")
base = y[ho].mean()
print(f"базовая ставка: {100*base:.1f}% монето-дней оказываются в top-20")


def auc(t, s):
    t = np.asarray(t); s = np.asarray(s)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = (cs - cnt + cs + 1) / 2.0
    r = avg[inv]; npos, nneg = int(t.sum()), len(t) - int(t.sum())
    return float("nan") if npos == 0 or nneg == 0 else (r[t == 1].sum() - npos*(npos+1)/2) / (npos*nneg)


def run(names: list[str], title: str):
    X = np.array([[float(r["f"].get(k) or 0.0) for k in names] for r in recs])
    m = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                           loss_function="Logloss", verbose=False, random_seed=42,
                           auto_class_weights="Balanced")
    m.fit(Pool(X[tr], y[tr]))
    p = m.predict_proba(Pool(X[ho]))[:, 1]
    by_day = defaultdict(list)
    for idx, gi in enumerate(ho):
        by_day[days[gi]].append((p[idx], y[gi]))
    total_top = sum(sum(1 for _, yy in v if yy) for v in by_day.values())
    print(f"\n{title}  ·  признаков {len(names)}  ·  holdout AUC {auc(y[ho], p):.3f}")
    print(f"  {'алертов/день':<14}{'поймано ракет':>15}{'точность':>11}{'лифт':>8}")
    for N in (3, 5, 10, 20):
        caught = shots = 0
        for _d, v in by_day.items():
            v.sort(key=lambda t: -t[0])
            caught += sum(1 for _, yy in v[:N] if yy)
            shots += len(v[:N])
        rec = caught/max(1, total_top)
        prec = caught/max(1, shots)
        print(f"  {N:<14}{100*rec:>14.0f}%{100*prec:>10.0f}%{prec/max(base,1e-9):>8.2f}x")
    return total_top, len(by_day)


tt, nd = run(names_all, "ВСЕ признаки часа 00 UTC")
run(names_nr, "БЕЗ признаков дневной доходности")
print(f"\nв holdout {tt} ракет за {nd} дней (~{tt/max(1,nd):.1f}/день)")
print("\nлифт = во сколько раз точность выше базовой ставки; 1.0 = модель не")
print("даёт ничего. Это и есть реальный потолок раннего предсказания на текущих")
print("данных — всё, что бот теряет ниже него, теряется уже на фильтрах и выходах.")
