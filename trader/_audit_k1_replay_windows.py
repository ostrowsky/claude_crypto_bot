"""K1 check (docs/audit/review-verification-2026-11-05.md): E1 replay of X4 on the same days as the live E7 shadow
and the demo account, so the comparison is between equal windows (TH-04). Read-only.

  (from trader/) ../pyembed/python.exe -u _audit_k1_replay_windows.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# (name, day_from, day_to exclusive): the H3 test half, the E7 shadow readout window, the demo executor window
WINDOWS = (("h3_test", "2026-07-01", "2026-10-11"), ("e7_window", "2026-10-13", "2026-11-05"),
           ("demo_window", "2026-10-20", "2026-11-05"))


def x4_policy():
    import trading_policy as TP
    return replace(TP.CHAMPION, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def run_windows(run, decompose, windows=WINDOWS) -> list:
    """run(day_from, day_to) -> replay result with 'agg' and 'trades'; one summary row per window."""
    out = []
    for name, lo, hi in windows:
        r = run(lo, hi)
        g, dc = r["agg"], decompose(r["trades"])
        out.append({"name": name, "from": lo, "to": hi, "n_days": g.get("n_days"), "mean_pct": g.get("mean_pct"),
                    "lo95": g.get("lo95"), "hi95": g.get("hi95"), "trades": dc.get("n"),
                    "net_per_trade": dc.get("net_mean_pct"), "gross_per_trade": dc.get("gross_mean_pct")})
    return out


def main() -> int:
    import execution_replay as ER
    import fill_model as FM
    trades, _ = ER.load_events()
    grid = ER._grid()
    ER.attach_grid(trades, grid)
    pol, fm, floor = x4_policy(), FM.model("taker", 1), ER._floor()
    rows = run_windows(lambda lo, hi: ER.run(pol, fm, grid, trades, floor_of=floor, day_from=lo, day_to=hi),
                       ER.decompose)
    for w in rows:
        print(f"{w['name']} {w['from']}..{w['to']}: days {w['n_days']} R {w['mean_pct']} [{w['lo95']}, {w['hi95']}] "
              f"trades {w['trades']} net/trade {w['net_per_trade']} gross/trade {w['gross_per_trade']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
