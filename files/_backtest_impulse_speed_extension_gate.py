"""Validate an EXTENSION gate for impulse_speed over the full critic_dataset
history (temporal split), with an over-block guard.

Finding (2026-06-05): impulse_speed loses when the entry is already extended
above its EMAs (causal real-time proxy for the hindsight 'lateness'). Candidate
gate: BLOCK impulse_speed if close_vs_ema50 > A OR ema50_vs_ema200 > B.

We must avoid the classic trap (CLAUDE.md s7): a gate that also blocks the big
winners. No top20 label in critic_dataset, so we proxy a 'big mover' by
ret_10 >= WINNER_RET and report how many of those the gate KEEPS (recall).

Targets measured (causal, no exit-policy noise): ret_5, ret_10, and the real
realized trade_exit_pnl.

Temporal split: earliest (1-TEST_FRAC) train, latest TEST_FRAC test. Thresholds
are fixed from the discriminator search (not fit on test). We also sweep them on
TRAIN only and re-report the train-chosen gate on TEST.

ASCII-only. Run from repo root:
    pyembed\python.exe files\_backtest_impulse_speed_extension_gate.py
"""
from __future__ import annotations
import json, sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET = "critic_dataset.jsonl"
TEST_FRAC = 0.30
WINNER_RET = 5.0          # ret_10 >= 5% = a 'big mover' we must not block
A_EMA50 = 3.0             # block if close_vs_ema50 > A
B_5020 = 2.45             # block if ema50_vs_ema200 > B


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
        r5 = _f(lab.get("ret_5"))
        if r5 is None:
            continue
        f = e.get("f", {}) or {}
        rows.append({
            "day": str(e.get("ts_signal", ""))[:10],
            "ret5": r5,
            "ret10": _f(lab.get("ret_10")),
            "pnl": _f(lab.get("trade_exit_pnl")),
            "cv_ema50": _f(f.get("close_vs_ema50")),
            "ema50_200": _f(f.get("ema50_vs_ema200")),
        })
    rows.sort(key=lambda r: r["day"])
    return rows


def blocked(r, a, b):
    cv = r["cv_ema50"]; ee = r["ema50_200"]
    return (cv is not None and cv > a) or (ee is not None and ee > b)


def stat(rs, key):
    vals = [r[key] for r in rs if r.get(key) is not None]
    n = len(vals)
    if not n:
        return None
    avg = sum(vals) / n
    win = sum(1 for v in vals if v > 0) / n * 100
    return {"n": n, "avg": avg, "win": win}


def report(rows, a, b, title):
    keep = [r for r in rows if not blocked(r, a, b)]
    cut = [r for r in rows if blocked(r, a, b)]
    winners = [r for r in rows if (r["ret10"] or 0) >= WINNER_RET]
    kept_winners = [r for r in winners if not blocked(r, a, b)]
    print(f"\n-- {title} (gate: close_vs_ema50>{a} OR ema50_vs_ema200>{b}) --")
    print(f"  total={len(rows)}  keep={len(keep)}  block={len(cut)} "
          f"({len(cut)/len(rows)*100:.0f}% blocked)")
    for key, lab in (("ret5", "ret_5"), ("pnl", "realized_pnl")):
        a0 = stat(rows, key); k = stat(keep, key); c = stat(cut, key)
        if a0 and k and c:
            print(f"  {lab:<12} ALL avg={a0['avg']:+.3f} win={a0['win']:.0f}% | "
                  f"KEEP avg={k['avg']:+.3f} win={k['win']:.0f}% (n={k['n']}) | "
                  f"BLOCK avg={c['avg']:+.3f} win={c['win']:.0f}% (n={c['n']})")
    if winners:
        rec = len(kept_winners) / len(winners) * 100
        print(f"  big movers (ret_10>={WINNER_RET}%): {len(winners)}  "
              f"KEPT by gate: {len(kept_winners)} -> recall {rec:.0f}%")


def main():
    rows = load()
    if len(rows) < 50:
        print("not enough data:", len(rows)); return
    days = sorted({r["day"] for r in rows})
    n_test = max(1, int(round(len(days) * TEST_FRAC)))
    test_days = set(days[-n_test:])
    train = [r for r in rows if r["day"] not in test_days]
    test = [r for r in rows if r["day"] in test_days]
    print("=" * 74)
    print("impulse_speed EXTENSION gate — full-history temporal validation")
    print("=" * 74)
    print(f"n={len(rows)}  span {days[0]}..{days[-1]}  "
          f"train={len(train)} ({days[0]}..{days[-n_test-1]})  "
          f"test={len(test)} ({days[-n_test]}..{days[-1]})")

    # 1) fixed thresholds on full / train / test
    report(rows, A_EMA50, B_5020, "FULL period")
    report(train, A_EMA50, B_5020, "TRAIN")
    report(test, A_EMA50, B_5020, "TEST (out-of-sample)")

    # 2) sweep thresholds on TRAIN, pick best by kept ret_5 with >=50% kept and
    #    winner-recall >= 80%, then re-report that gate on TEST
    print("\n-- threshold sweep on TRAIN (pick: max keep avg_ret5 s.t. "
          "keep>=50% and big-mover recall>=80%) --")
    winners_tr = [r for r in train if (r["ret10"] or 0) >= WINNER_RET]
    best = None
    for a in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        for b in (1.5, 2.0, 2.45, 3.0, 4.0):
            keep = [r for r in train if not blocked(r, a, b)]
            if len(keep) < len(train) * 0.5:
                continue
            kw = [r for r in winners_tr if not blocked(r, a, b)]
            rec = (len(kw) / len(winners_tr) * 100) if winners_tr else 100
            if rec < 80:
                continue
            ks = stat(keep, "ret5")
            if ks is None:
                continue
            if best is None or ks["avg"] > best[0]:
                best = (ks["avg"], a, b, len(keep), rec)
    if best:
        _, a, b, kn, rec = best
        print(f"  best train gate: close_vs_ema50>{a} OR ema50_vs_ema200>{b}  "
              f"(train keep n={kn}, keep_avg_ret5={best[0]:+.3f}, recall={rec:.0f}%)")
        report(test, a, b, "TEST with TRAIN-chosen gate")
    else:
        print("  no gate met the constraints on train.")


if __name__ == "__main__":
    main()
