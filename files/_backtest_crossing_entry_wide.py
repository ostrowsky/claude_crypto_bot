"""Follow-up to _backtest_crossing_entry.py (a SECOND look at the same data --
reported as such): the rank<=3 crossing entry exited in leader mode from the
entry bar (wide trail max(k*ATR, 8%), exit on losing the day's top-10 after 1h,
7-day cap), instead of the plain trail that switches only later.

Why this single variant: rank<=3 at the crossing had precision 0.202 (2x the
bot's own 15m entries) and a rocket share of 0.125, but lost -0.373%/trade on the
plain trail; the leader exit is the one exit that has shown a gain on rockets
(+1.18 pp, leader-exit-x6-spec.md). Same pre-registered verdict as the parent:
precision >= bot, pnl lower 95% >= bot mean - 0.10 pp, reach >= 10% late/none.
"""
import sys
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
import exit_validator as EV  # noqa: E402
import _backtest_crossing_entry as X  # noqa: E402


def wide_now(g, j, ie, k, floor, P):
    return EV.wide_from(g, {"j": j, "ep": g.C[ie, j], "k": k}, ie, P), True


X.sim_exit = wide_now
X.KS = (3, 5)
if __name__ == "__main__":
    X.main()
