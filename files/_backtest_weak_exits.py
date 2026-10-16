"""Weakness exits vs holding on the trail -- offline replay, nothing deployed.

WHY (2026-10-05, prod-tools plan, exit side; rocket-segment-spec.md audit)
Capture is the North Star's weak factor (0.22-0.25). On 255 bought rockets the
bot kept a median +1.3% of a 14.1% day; after exits on WEAK RSI divergence, RSI
overbought and similar signals the coin rose another +5.6..+8.2% the same day.
H5 (trail-only above break-even) deliberately does NOT suppress WEAK / RSI / MACD
exits. This replay asks whether holding on the trail instead would capture more.

DESIGN, FIXED BEFORE THE RUN
  trades   every real 15m trade with an exit since 2026-03-01 (exit_validator's
           grid and trade pairing), exit reason kept.
  weakness exits = WEAK RSI divergence, RSI overbought, WEAK quality recheck
           (price_edge / MACD), WEAK EMA fan compressed, WEAK volume exhaustion.
  current  exit at the close of the weakness bar (grid close, same basis as the variant).
  variant  no weakness exit: the calibrated P-1 plain trail (trade's trail_k, the
           mode's floor) ratcheted from the entry bar, continued up to 96 bars after
           the weakness bar; if the ratcheted stop is already above the close at
           that bar, the trade exits there (no change). Trail crossings before the
           weakness bar are ignored: the live trade did not exit there.
  primary  ALL weakness categories together -> one verdict with exit_validator.decide
           (rocket lower95 > 0, all lower95 >= -0.10 pp, >= 60% months >= 0).
  per category: descriptive only (five categories; picking the best after looking
           would be selection on the test).
  Trades with any other exit reason are unchanged (difference 0).
Run from files/:  ../pyembed/python.exe _backtest_weak_exits.py
"""
from __future__ import annotations

import collections
import io
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config                         # noqa: E402
import exit_validator as EV           # noqa: E402

AFTER_BARS = 96
OUT = HERE.parent / ".runtime" / "weak_exits_backtest.json"


def category(reason: str):
    r = (reason or "").lower()
    if "rsi" in r and ("дивергенц" in r or "diverg" in r):
        return "rsi_divergence"
    if "rsi перекуплен" in r or "overbought" in r:
        return "rsi_overbought"
    if "quality recheck" in r and "price_edge" in r:
        return "quality_price_edge"
    if "quality recheck" in r and "macd" in r:
        return "quality_macd"
    if "веер" in r:
        return "ema_fan"
    if "истощени" in r:
        return "volume_exhaustion"
    return None


def load_trades_with_reason(grid, since):
    ent, exs = collections.defaultdict(list), collections.defaultdict(list)
    with io.open(EV.EVENTS, "rb") as fh:
        for raw in fh:
            if b'"entry"' not in raw and b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") not in ("entry", "exit") or e.get("tf") != "15m" \
                    or str(e.get("ts", "")) < since or e.get("sym") not in grid.col:
                continue
            d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
            d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            (ent if e["event"] == "entry" else exs)[e["sym"]].append((d, e))
    out = []
    bar = lambda t: t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0)  # noqa: E731
    for sym, es in ent.items():
        xs = sorted(exs.get(sym, []), key=lambda x: x[0])
        for d, e in sorted(es, key=lambda x: x[0]):
            x = next((x for x in xs if x[0] > d), None)
            if x is None or not isinstance(x[1].get("pnl_pct"), (int, float)) or not e.get("price"):
                continue
            ie, ix = grid.idx(bar(d)) - 1, grid.idx(bar(x[0])) - 1
            if ie < 100 or ix <= ie or ix >= grid.N:
                continue
            k = e.get("trail_k")
            out.append({"sym": sym, "j": grid.col[sym], "ie": ie, "ix": ix, "ep": float(e["price"]),
                        "k": float(k) if isinstance(k, (int, float)) else 2.0, "mode": e.get("mode") or e.get("signal_mode"),
                        "day": d.strftime("%Y-%m-%d"), "reason": str(x[1].get("reason") or ""), "real": float(x[1]["pnl_pct"])})
    return out


def hold_on_trail(grid, t, floor):
    """P-1 plain trail ratcheted from the entry bar; exits only after the weakness bar ix."""
    j, ie, ix, k, ep = t["j"], t["ie"], t["ix"], t["k"], t["ep"]
    C, A = grid.C, grid.ATR
    c0 = C[ie, j]
    if not math.isfinite(c0):
        return None
    stop = c0 - max(k * A[ie, j] if math.isfinite(A[ie, j]) else 0.0, floor * c0)
    last = min(grid.N - 1, ix + AFTER_BARS)
    for q in range(ie + 1, last + 1):
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(k * A[q, j], floor * cq))
        if q >= ix and cq < stop:
            return (cq / ep - 1) * 100
    q = last
    while q > ix and not math.isfinite(C[q, j]):
        q -= 1
    return (C[q, j] / ep - 1) * 100 if math.isfinite(C[q, j]) else None


def at_bar(grid, t):
    c = grid.C[t["ix"], t["j"]]
    return (c / t["ep"] - 1) * 100 if math.isfinite(c) else None


def stats(rows, rockets, winners, seed=29):
    rnd = random.Random(seed)
    d_all = [y - x for _, x, y in rows]
    d_rk = [y - x for t, x, y in rows if (t["day"], t["sym"]) in rockets]
    d_wn = [y - x for t, x, y in rows if (t["day"], t["sym"]) in winners]
    changed = sum(1 for v in d_all if abs(v) > 1e-9)
    bym = collections.defaultdict(list)
    for t, x, y in rows:
        if (t["day"], t["sym"]) in rockets:
            bym[t["day"][:7]].append(y - x)
    months = {m: {"mean": round(sum(v) / len(v), 3), "n": len(v)} for m, v in sorted(bym.items())}
    scored = [m for m, v in months.items() if v["n"] >= EV.MONTH_MIN]
    agree = (sum(1 for m in scored if months[m]["mean"] >= 0) / len(scored)) if scored else 0.0
    all_s, rk_s, wn_s = EV.paired(d_all, rnd), EV.paired(d_rk, rnd), EV.paired(d_wn, rnd)
    return {"trades": len(rows), "changed_trades": changed, "all_trades_diff_pp": all_s, "rocket_day_diff_pp": rk_s,
            "winner_day_diff_pp": wn_s, "rocket_months": months, "month_agreement": round(agree, 3),
            "months_scored": len(scored), "verdict": EV.decide(all_s, rk_s, agree, changed)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    grid = EV.load_grid(int(getattr(config, "ATR_PERIOD", 14)))
    trades = load_trades_with_reason(grid, EV.SINCE_DEFAULT)
    rockets, winners = EV.day_labels()
    cats = collections.Counter(category(t["reason"]) for t in trades)
    print(f"trades since {EV.SINCE_DEFAULT}: {len(trades)}; weakness exits by category: {dict(cats)}", flush=True)
    base, var = {}, {}
    for i, t in enumerate(trades):
        c = category(t["reason"])
        if c is None:
            continue
        x, y = at_bar(grid, t), hold_on_trail(grid, t, EV.floor_of(t["mode"], config))
        if x is not None and y is not None:
            base[i], var[i] = x, y
    report = {"since": EV.SINCE_DEFAULT, "after_bars": AFTER_BARS, "categories": dict(cats), "per_category": {}}
    # primary: all weakness categories; unchanged trades contribute 0
    rows = [(t, base[i], var[i]) if i in base else (t, 0.0, 0.0) for i, t in enumerate(trades)]
    report["ALL_WEAKNESS"] = stats(rows, rockets, winners)
    for c in sorted(k for k in cats if k):
        sub = [(t, base[i], var[i]) for i, t in enumerate(trades) if i in base and category(t["reason"]) == c]
        if not sub:
            continue
        r = stats(sub, rockets, winners)
        r["mean_exit_at_weakness"] = round(sum(x for _, x, _ in sub) / len(sub), 3)
        r["mean_hold_on_trail"] = round(sum(y for _, _, y in sub) / len(sub), 3)
        report["per_category"][c] = r
    a = report["ALL_WEAKNESS"]
    print(f"\n[ALL WEAKNESS, primary] verdict={a['verdict']} trades={a['trades']} changed={a['changed_trades']}\n"
          f"  rocket-days {a['rocket_day_diff_pp']}\n  all {a['all_trades_diff_pp']}\n  winners {a['winner_day_diff_pp']}\n"
          f"  months agree {a['month_agreement']:.0%} of {a['months_scored']}: {a['rocket_months']}")
    print("\nper category (descriptive only): n, mean exit-at-signal -> mean hold-on-trail, paired diff on all / rocket-days")
    for c, r in report["per_category"].items():
        print(f"  {c:20s} n={r['trades']:4d}  {r['mean_exit_at_weakness']:+.2f} -> {r['mean_hold_on_trail']:+.2f}  "
              f"all {r['all_trades_diff_pp']}  rocket {r['rocket_day_diff_pp']}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
