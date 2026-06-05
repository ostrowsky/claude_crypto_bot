"""Backtest regime curtailment with auto-revive for impulse_speed.

Decided after 4 analyses proved no learnable entry-time signal for the mode
(OOS AUC ~0.50): the only evidence-based lever is mode-level regime gating.

Policy: walk days in order. On each day, look at impulse_speed's trailing
WINDOW-day realized pnl from PRIOR days only (causal). If that trailing mean is
< CURTAIL_THRESHOLD, the mode is CURTAILED that day (its entries are dropped);
otherwise it is ACTIVE. This auto-revives when the recent regime turns
profitable again — no permanent disable.

Tradeoff measured (per impulse_speed entry, full critic_dataset history):
  - realized pnl saved   (sum of pnl on curtailed days = losses we avoid)
  - big movers lost       (ret_10>=WINNER_RET caught on curtailed days)
  - net realized pnl baseline vs curtailed

Sweeps WINDOW and threshold so we deploy a robust setting, not a fit one.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_impulse_speed_curtail.py
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from datetime import date, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET = "critic_dataset.jsonl"
WINNER_RET = 5.0


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    rows = []
    for ln in open(DATASET, encoding="utf-8", errors="replace"):
        if "impulse_speed" not in ln:
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("signal_type") != "impulse_speed":
            continue
        dec = e.get("decision", {}) or {}
        if str(dec.get("action", "")) != "take":
            continue
        lab = e.get("labels", {}) or {}
        pnl = _f(lab.get("trade_exit_pnl"))
        if pnl is None:
            continue
        d = str(e.get("ts_signal", ""))[:10]
        if not d:
            continue
        rows.append({"day": d, "pnl": pnl, "ret10": _f(lab.get("ret_10")) or 0.0})
    return rows


def daypnl(rows):
    agg = defaultdict(lambda: [0.0, 0])
    for r in rows:
        agg[r["day"]][0] += r["pnl"]; agg[r["day"]][1] += 1
    return agg


def simulate(rows, window, thr):
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    days = sorted(by_day)
    dmean = {d: (sum(x["pnl"] for x in by_day[d]) / len(by_day[d])) for d in days}

    active_pnl = active_big = curtailed_pnl = curtailed_big = 0.0
    n_active = n_curtailed = 0
    for d in days:
        dt = date.fromisoformat(d)
        prior = [dmean[p] for p in days
                 if date.fromisoformat(p) < dt
                 and date.fromisoformat(p) >= dt - timedelta(days=window)]
        active = True
        if len(prior) >= 3:                       # need some history to judge regime
            active = (sum(prior) / len(prior)) >= thr
        for r in by_day[d]:
            big = r["ret10"] >= WINNER_RET
            if active:
                active_pnl += r["pnl"]; n_active += 1; active_big += big
            else:
                curtailed_pnl += r["pnl"]; n_curtailed += 1; curtailed_big += big
    return {
        "n_active": n_active, "n_curtailed": n_curtailed,
        "active_pnl": active_pnl, "curtailed_pnl": curtailed_pnl,
        "active_big": active_big, "curtailed_big": curtailed_big,
    }


def main():
    rows = load()
    if not rows:
        print("no data"); return
    days = sorted({r["day"] for r in rows})
    base_sum = sum(r["pnl"] for r in rows)
    base_big = sum(1 for r in rows if r["ret10"] >= WINNER_RET)
    print("=" * 74)
    print("impulse_speed regime curtailment (auto-revive) backtest")
    print("=" * 74)
    print(f"n={len(rows)}  span {days[0]}..{days[-1]}  "
          f"baseline: realized_pnl_sum={base_sum:+.1f}  "
          f"avg/trade={base_sum/len(rows):+.3f}  big_movers={base_big}")
    print(f"\n{'window':>7}{'thr':>6}{'active_n':>9}{'curtail_n':>10}"
          f"{'kept_pnl':>10}{'avoided_pnl':>12}{'big_lost':>9}{'kept_avg':>9}")
    print("-" * 72)
    for window in (10, 14, 21):
        for thr in (0.0, -0.1, -0.2):
            s = simulate(rows, window, thr)
            kept_avg = s["active_pnl"]/s["n_active"] if s["n_active"] else float("nan")
            print(f"{window:>7}{thr:>6.1f}{s['n_active']:>9}{s['n_curtailed']:>10}"
                  f"{s['active_pnl']:>+10.1f}{s['curtailed_pnl']:>+12.1f}"
                  f"{int(s['curtailed_big']):>9}{kept_avg:>+9.3f}")
    print("\nRead: avoided_pnl negative = losses we removed (good). big_lost = big")
    print("movers (ret_10>=5%) that fell on curtailed days (the cost). Want avoided")
    print("strongly negative, kept_avg positive, big_lost small. Pick robust cell.")


if __name__ == "__main__":
    main()
