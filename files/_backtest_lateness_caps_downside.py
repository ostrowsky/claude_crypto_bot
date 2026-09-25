"""The peak is a proxy (TH-11): add the downside for the same rows.

For every row of lateness_rows.jsonl: trough_4h (lowest low of the next 16 bars),
close_4h (close 16 bars later), and a simple exit-realistic outcome -- the ATR
trail from _backtest_weak_exit_above_breakeven.replay, entered at the row's
close, capped at 96 bars, trail_k 2.0 with the entry mode's min-buffer floor.
Compared within the same daily_range bucket so volatility is held roughly equal.
"""
import collections
import io
import json
import sys
from pathlib import Path

FILES = Path("D:/Projects/claude_crypto_bot/files")
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _backtest_trend_start_detector as TD  # noqa: E402
import _backtest_weak_exit_above_breakeven as W  # noqa: E402
import config as cfg  # noqa: E402
from datetime import datetime  # noqa: E402

SP = FILES.parent / ".runtime" / "backtests"
rows = [json.loads(l) for l in io.open(SP / "lateness_rows.jsonl", encoding="utf-8")]
by_sym = collections.defaultdict(list)
for r in rows:
    by_sym[r["sym"]].append(r)

out = []
for k, (sym, rs) in enumerate(sorted(by_sym.items()), 1):
    bars = TD.bars_15m(sym)
    idx = {b[0]: i for i, b in enumerate(bars)}
    for r in rs:
        i = idx.get(datetime.fromisoformat(r["ts"]))
        if i is None or i + 16 >= len(bars):
            continue
        px = bars[i][4]
        fut = bars[i + 1: i + 17]
        trough = (min(b[3] for b in fut) / px - 1) * 100
        close4 = (fut[-1][4] / px - 1) * 100
        mode = r["rule"] if r["rule"] in ("trend", "impulse", "alignment", "impulse_speed") else "trend"
        trail = W.replay(bars, i, i, px, 2.0, mode, cfg, 96)
        out.append(dict(r, trough_4h=trough, close_4h=close4, trail=trail))
    if k % 25 == 0:
        print("  %d/%d symbols" % (k, len(by_sym)), flush=True)


def s(v):
    v = sorted(x for x in v if x is not None)
    n = len(v)
    return (sum(v) / n, v[n // 2], n) if n else (float("nan"), float("nan"), 0)


print("\n%-9s %7s %10s %10s %10s %12s %12s %10s" % ("band", "n", "peak4h", "trough4h", "close4h",
                                                      "trail mean", "trail med", "trail>0"))
for b in ("FIRE", "BULLBAND", "LATE"):
    v = [r for r in out if r["band"] == b]
    pk, tr, cl, ta = s([r["peak_4h"] for r in v]), s([r["trough_4h"] for r in v]), s([r["close_4h"] for r in v]), s([r["trail"] for r in v])
    pos = sum(1 for r in v if r["trail"] is not None and r["trail"] > 0) / max(1, sum(1 for r in v if r["trail"] is not None))
    print("%-9s %7d %9.2f%% %9.2f%% %9.2f%% %11.2f%% %11.2f%% %9.0f%%" % (b, len(v), pk[0], tr[0], cl[0], ta[0], ta[1], 100 * pos))

print("\nSame daily_range bucket -- trail outcome mean / median [n]  (the like-for-like view)")
print("%-10s %28s %28s %28s" % ("dr bucket", "FIRE", "BULLBAND", "LATE"))
for lo, hi in ((7, 10), (10, 15), (15, 25), (25, 1e9)):
    cells = []
    for b in ("FIRE", "BULLBAND", "LATE"):
        v = [r["trail"] for r in out if r["band"] == b and r["dr"] is not None and lo <= r["dr"] < hi and r["trail"] is not None]
        m, md, n = s(v)
        cells.append(("%+5.2f%% / %+5.2f%% [%5d]" % (m, md, n)) if n >= 30 else "--")
    print("%-10s %28s %28s %28s" % ("%g-%s" % (lo, "" if hi > 1e8 else "%g" % hi), *cells))

print("\nclose_4h within bucket (mean [n]):")
for lo, hi in ((7, 10), (10, 15), (15, 25), (25, 1e9)):
    cells = []
    for b in ("FIRE", "LATE"):
        v = [r["close_4h"] for r in out if r["band"] == b and r["dr"] is not None and lo <= r["dr"] < hi]
        m, md, n = s(v)
        cells.append(("%+5.2f%% [%5d]" % (m, n)) if n >= 30 else "--")
    print("  %-10s FIRE %s   LATE %s" % ("%g-%s" % (lo, "" if hi > 1e8 else "%g" % hi), *cells))

qnt = [r for r in out if r["sym"] == "QNTUSDT" and r["day"] in ("2026-09-24", "2026-09-25")]
print("\nQNT 09-24/25 rows:", [(r["ts"][5:16], r["band"], r["rule"], r["dr"], round(r["peak_4h"], 1), None if r["trail"] is None else round(r["trail"], 1)) for r in qnt])
json.dump(out, io.open(SP / "lateness_rows_enriched.json", "w", encoding="utf-8"))
