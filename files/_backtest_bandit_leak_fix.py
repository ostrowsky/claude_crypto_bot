"""Does fixing the bandit's training label change what it actually admits?

The leak, measured: `train_entry_bandit(use_earliest_snapshot=True)` takes the
earliest snapshot of each calendar day, which is the 00 UTC record for 41% of
samples — and that record is the EOD resolution of a day already over. 34.8% of
the chosen rows have |eod_return - return_since_open| < 0.5%: nothing left to
decide. Among the rows rewarded +1.0 for ENTER, the median move still ahead is
+1.75%, and only 41% still have +3% ahead. The bandit is paid for entering moves
that already happened.

The honest target for an ENTER decision is the move that remains AFTER the
snapshot:

    forward_pct = ((1 + eod/100) / (1 + since_open/100) - 1) * 100

This trains two bandits on the same contexts and the same temporal split —

    OLD  arm/reward from `label_top20`   (the leaky label)
    NEW  arm/reward from forward_pct >= BANDIT_FORWARD_MIN_PCT, with rows whose
         day is already decided dropped outright

— and scores every variant on the SAME honest holdout target, reporting the
ENTER rate next to recall and lift (§0a rule 1). A policy that fires ENTER at
everything reaches high recall with lift ~1; that is the failure this measures.

VERDICT (2026-08-13, 309 days, split by time, holdout 9765 rows; target = the
coin still has >= +3% ahead of the snapshot, base rate 2.0%):

    training label                 ENTER   caught  precision   lift
    old label_top20                80.3%     52%      1.3%    0.65x
    rank top-20, no floor          98.0%    100%      2.1%    1.02x
    rank top-20 + floor +3%        43.5%     99%      4.6%    2.28x
    rank top-10 + floor +3%        24.3%     99%      8.3%    4.07x   <- shipped
    floor +3%, no rank             80.0%    100%      2.5%    1.25x

Two things to not re-litigate. The old label is not merely weak, it is BELOW
random (0.65x) — it teaches the bandit to prefer coins whose move is over. And
a pure rank label is worthless here (1.02x): ranking mints exactly N winners a
day whatever the market does, so the floor is the part that carries meaning.

Read-only.  pyembed\python.exe files\_backtest_bandit_leak_fix.py
"""
from __future__ import annotations
import io, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contextual_bandit import LinUCBBandit, extract_context, N_FEATURES  # noqa: E402

FWD_MIN = 3.0        # % still ahead for an ENTER to have been worth it
DECIDED_EPS = 0.5    # |eod - since_open| below this: the day is already over
TRAIN_FRAC = 0.70


def forward_pct(rec: dict) -> float | None:
    f = rec.get("features") or {}
    so, eo = f.get("tg_return_since_open"), rec.get("eod_return_pct")
    if not isinstance(so, (int, float)) or not isinstance(eo, (int, float)):
        return None
    den = 1.0 + so/100.0
    if den <= 0.01:
        return None
    return ((1.0 + eo/100.0)/den - 1.0) * 100.0


def ctx(rec: dict) -> np.ndarray:
    f = rec.get("features") or {}
    state = {
        "slope_pct": f.get("tg_ema20_slope", 0.0),
        "adx": f.get("tg_adx", 20.0),
        "rsi": f.get("tg_rsi", 50.0),
        "vol_x": f.get("tg_volume_ratio_1h", 1.0),
        "ml_proba": 0.5,
        "daily_range": f.get("tg_daily_range_pct", 3.0),
        "macd_hist": f.get("tg_ema20_slope", 0.0),
    }
    btc = f.get("tg_btc_return_4h", 0.0)
    bull = btc > 0.3
    return extract_context(state, mode="trend", tf="15m", is_bull_day=bull,
                           market_regime="bull" if bull else "neutral",
                           btc_vs_ema50=btc)


WL = set(json.load(io.open(Path(__file__).parent/"watchlist.json", encoding="utf-8")))
by_day: dict[str, dict[str, dict]] = defaultdict(dict)
for ln in io.open(Path(__file__).parent/"top_gainer_dataset.jsonl",
                  encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln:
        continue
    try:
        e = json.loads(ln)
    except Exception:
        continue
    ts, sym = e.get("ts"), e.get("symbol")
    if not ts or sym not in WL or not e.get("features"):
        continue
    d = datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d")
    cur = by_day[d].get(sym)
    if cur is None or ts < cur["ts"]:
        by_day[d][sym] = e

rows = []
for d in sorted(by_day):
    for sym, rec in by_day[d].items():
        fw = forward_pct(rec)
        if fw is None:
            continue
        f = rec.get("features") or {}
        so, eo = f.get("tg_return_since_open"), rec.get("eod_return_pct")
        rows.append({"d": d, "x": ctx(rec), "fwd": fw,
                     "old_top": 1 if rec.get("label_top20") == 1 else 0,
                     "decided": abs(eo - so) < DECIDED_EPS})

days = sorted({r["d"] for r in rows})
cut = days[int(len(days)*TRAIN_FRAC)]
tr = [r for r in rows if r["d"] < cut]
ho = [r for r in rows if r["d"] >= cut]

print("=" * 74)
print("Утечка в обучении бандита: старая метка против движения ВПЕРЁД")
print("=" * 74)
print(f"строк {len(rows)} · дней {len(days)} · train < {cut} ({len(tr)}) · holdout {len(ho)}")
n_dec = sum(1 for r in rows if r["decided"])
print(f"день уже закрыт на момент снимка: {n_dec} строк ({100*n_dec/len(rows):.1f}%) — "
      f"новая разметка их выбрасывает")

# честная цель: у монеты остаётся >= FWD_MIN хода после снимка
truth = np.array([1 if r["fwd"] >= FWD_MIN else 0 for r in ho])
base = truth.mean()
print(f"\nчестная цель на holdout: {100*base:.1f}% строк ещё дают +{FWD_MIN:.0f}% "
      f"после снимка ({int(truth.sum())} из {len(ho)})")


def train(samples):
    b = LinUCBBandit(n_arms=2, n_features=N_FEATURES, alpha=2.0)
    b.batch_update(samples)
    return b


def build(rows_, kind: str, top_n: int = 0, floor: float | None = None):
    """Mark positives, then emit both arms per row with the unchanged rewards."""
    if kind == "old":
        winners = {id(r) for r in rows_ if r["old_top"]}
        pool = rows_
    else:
        pool = [r for r in rows_ if not r["decided"]]
        by_d = defaultdict(list)
        for r in pool:
            by_d[r["d"]].append(r)
        winners = set()
        for _d, rs in by_d.items():
            rs.sort(key=lambda r: -r["fwd"])
            for r in (rs[:top_n] if top_n else rs):
                if floor is None or r["fwd"] >= floor:
                    winners.add(id(r))
    s = []
    for r in pool:
        if id(r) in winners:
            s.append((r["x"], 1, 1.0)); s.append((r["x"], 0, -0.8))
        else:
            s.append((r["x"], 0, 0.10)); s.append((r["x"], 1, -0.12))
    return s, len(winners)


def run(title: str, kind: str, top_n: int = 0, floor: float | None = None):
    samples, npos = build(tr, kind, top_n, floor)
    b = train(samples)
    arms = np.array([b.select_arm(r["x"])[0] for r in ho])
    fire = arms.mean()
    hit = int(((arms == 1) & (truth == 1)).sum())
    rec = hit/max(1, int(truth.sum()))
    prec = hit/max(1, int((arms == 1).sum()))
    print(f"  {title:<32}{100*fire:>7.1f}%{100*rec:>9.0f}%{100*prec:>10.1f}%"
          f"{prec/max(base,1e-9):>8.2f}x{npos:>10}")
    return prec/max(base, 1e-9)


print(f"\n  {'разметка обучения':<32}{'ENTER':>8}{'поймано':>9}{'точность':>10}"
      f"{'лифт':>9}{'позитивов':>10}")
lift_old = run("СТАРОЕ label_top20", "old")
run("ранг top-20, без пола", "new", 20, None)
run("ранг top-20 + пол +3%", "new", 20, FWD_MIN)
lift_new = run("ранг top-10 + пол +3%  <-", "new", 10, FWD_MIN)
run("пол +3% без ранга", "new", 0, FWD_MIN)

print("\n" + "-" * 74)
print(f"лифт {lift_old:.2f}x -> {lift_new:.2f}x на одной и той же цели.")
print("Правка стоит того, только если лифт вырос. Если бандит просто стал реже")
print("входить при том же лифте — это не обучение, а сдвиг порога.")
