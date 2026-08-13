"""Can a rocket be predicted at the START of the day? Built from klines only.

Why not from top_gainer_dataset: its label cannot answer this (see the verdict of
_backtest_early_prediction_ceiling.py). At the 00/12/18 UTC snapshots a labelled
rocket already shows the whole move in `tg_return_since_open` (+13.98% vs an
eod_return of +13.94%) — features and label are the same number. And the 06 UTC
snapshot uses a different meaning of the same field: ~12.9 "rockets" per day at a
median +2.55%, against ~1.5/day at +13.7% elsewhere. 84% of all positives come
from that loose 06:00 labelling, which is what teaches the bandit to fire ENTER
at almost everything.

So this test builds its own ground truth from 1h klines:

  decision point  T = 00:00 UTC of each day
  features        computed ONLY from bars strictly before T (prior 24h/72h/7d
                  returns, volatility, volume, RSI, MA distances, BTC context)
  label           the coin gains >= ROCKET_PCT between T and the day's end
  split           by time — train on earlier days, evaluate on later ones
  reported        recall@N against an alert budget of N coins/day, always with
                  the base rate and lift (CLAUDE.md §0a rule 1)

If lift is ~1, rockets are not predictable at 00:00 from price history alone, and
no amount of gate tuning downstream will change that.

RESULT (2026-08-13, 98 symbols x 191 days, holdout = 58 later days, 120 rockets):

    base rate 2.18% of coin-days gain >= +10%      holdout AUC 0.714

      alerts/day   rockets caught   precision   lift
         3              18%            12.1%    5.54x
         5              22%             9.3%    4.28x
        10              37%             7.6%    3.48x
        20              49%             5.1%    2.34x

So the edge at day start is REAL but bounded: ~3.5x better than random, and even
spending 20 alerts a day (a fifth of the universe) finds under half the rockets.
Predicting 100% of rockets at 00:00 UTC from price history is not available at
any alert budget — the information is not in the pre-day bars. Earliness and
recall trade against each other, and the intraday path (detect the move once it
starts) is where the remaining rockets live, not a better day-start model.

Read-only.  pyembed\python.exe files\_backtest_early_prediction_klines.py
Needs the long cache:  pyembed\python.exe files\_fetch_hourly_history.py 200
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
# Prefer the long cache (200d, contiguous). The short one holds 1000 bars/symbol
# (~42 days) and 11 gappy series whose stray timestamps wreck a time split —
# a 70/30 cut on it put 271 rows in train against 3100 in holdout.
_long = ROOT/"files"/"_hourly_ohlcv_long.json"
_src = _long if _long.exists() else ROOT/"files"/"_hourly_ohlcv.json"
H = json.load(io.open(_src, encoding="utf-8"))
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
ROCKET_PCT = 10.0          # what counts as a rocket, in one day


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    k = 2.0/(n+1); au = np.empty_like(up); ad = np.empty_like(dn)
    au[0], ad[0] = up[0], dn[0]
    for i in range(1, len(up)):
        au[i] = up[i]*k + au[i-1]*(1-k); ad[i] = dn[i]*k + ad[i-1]*(1-k)
    return 100 - 100/(1 + au/np.maximum(ad, 1e-12))


# BTC context per hour
btc = H.get("BTCUSDT")
btc_ret = {}
if btc:
    bt = [r[0] for r in btc]; bc = [r[4] for r in btc]
    for i in range(24, len(bc)):
        btc_ret[int(bt[i])] = (bc[i]/bc[i-24]-1)*100

rows = []
for sym, k in H.items():
    if sym not in WL or len(k) < 250:
        continue
    t = np.array([r[0] for r in k]); hi = np.array([r[2] for r in k])
    c = np.array([r[4] for r in k]); v = np.array([r[5] for r in k])
    r14 = rsi(c)
    dts = [datetime.fromtimestamp(x/1000, tz=timezone.utc) for x in t]
    for i in range(200, len(c) - 24):
        if dts[i].hour != 0:                       # decision at 00:00 UTC
            continue
        # label: the day AHEAD (T .. T+24h)
        fwd_hi = float(np.max(hi[i+1:i+25]))
        fwd = (fwd_hi/c[i] - 1) * 100
        # features: strictly BEFORE T
        w = c[:i+1]
        if len(w) < 200 or w[-1] <= 0:
            continue
        f = {
            "ret_24h": (w[-1]/w[-25]-1)*100,
            "ret_72h": (w[-1]/w[-73]-1)*100,
            "ret_7d": (w[-1]/w[-169]-1)*100,
            "vol_24h": float(np.std(np.diff(np.log(w[-25:]))))*100,
            "vol_7d": float(np.std(np.diff(np.log(w[-169:]))))*100,
            "volx_24h": float(np.mean(v[i-23:i+1]) / max(np.mean(v[i-167:i-23]), 1e-9)),
            "rsi": float(r14[i]),
            "dist_ma25": (w[-1]/float(np.mean(w[-25:]))-1)*100,
            "dist_ma99": (w[-1]/float(np.mean(w[-99:]))-1)*100,
            "range_7d": (float(np.max(hi[i-167:i+1]))/w[-1]-1)*100,
            "btc_ret_24h": btc_ret.get(int(t[i]), 0.0),
        }
        rows.append({"d": dts[i].strftime("%Y-%m-%d"), "sym": sym,
                     "f": f, "y": 1 if fwd >= ROCKET_PCT else 0})

if len(rows) < 500:
    print(f"only {len(rows)} rows — not enough"); sys.exit(0)

# keep only days the whole universe covers, else the split mixes eras
per_day = defaultdict(int)
for r in rows:
    per_day[r["d"]] += 1
full = max(per_day.values())
rows = [r for r in rows if per_day[r["d"]] >= full * 0.8]

names = sorted(rows[0]["f"].keys())
X = np.array([[r["f"][k] for k in names] for r in rows])
y = np.array([r["y"] for r in rows])
days = [r["d"] for r in rows]
uniq = sorted(set(days)); cut = uniq[int(len(uniq)*0.70)]
tr = [i for i, d in enumerate(days) if d < cut]
ho = [i for i, d in enumerate(days) if d >= cut]

print("=" * 74)
print(f"Предсказание ракеты (>= +{ROCKET_PCT:.0f}% за сутки) в 00:00 UTC · "
      f"только свечи, признаки строго ДО момента решения")
print("=" * 74)
print(f"строк {len(rows)} · дней {len(uniq)} · train < {cut} ({len(tr)}) · holdout {len(ho)}")
base = y[ho].mean()
print(f"базовая ставка: {100*base:.2f}% монето-дней дают >= +{ROCKET_PCT:.0f}%")

m = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05,
                       loss_function="Logloss", verbose=False, random_seed=42,
                       auto_class_weights="Balanced")
m.fit(Pool(X[tr], y[tr]))
p = m.predict_proba(Pool(X[ho]))[:, 1]


def auc(t_, s):
    t_ = np.asarray(t_); s = np.asarray(s)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = (cs - cnt + cs + 1)/2.0
    r = avg[inv]; npos, nneg = int(t_.sum()), len(t_)-int(t_.sum())
    return float("nan") if npos == 0 or nneg == 0 else (r[t_ == 1].sum()-npos*(npos+1)/2)/(npos*nneg)


print(f"holdout AUC: {auc(y[ho], p):.3f}")
by_day = defaultdict(list)
for idx, gi in enumerate(ho):
    by_day[days[gi]].append((p[idx], y[gi]))
total = int(y[ho].sum())
print(f"\n  {'алертов/день':<14}{'поймано ракет':>15}{'точность':>11}{'лифт':>8}")
for N in (3, 5, 10, 20):
    caught = shots = 0
    for _d, vv in by_day.items():
        vv.sort(key=lambda x: -x[0])
        caught += sum(1 for _, yy in vv[:N] if yy); shots += len(vv[:N])
    rec = caught/max(1, total); prec = caught/max(1, shots)
    print(f"  {N:<14}{100*rec:>14.0f}%{100*prec:>10.1f}%{prec/max(base,1e-9):>8.2f}x")
print(f"\nв holdout {total} ракет за {len(by_day)} дней (~{total/max(1,len(by_day)):.1f}/день)")
print("\nлифт 1.0 = модель не даёт ничего сверх случайного выбора.")
