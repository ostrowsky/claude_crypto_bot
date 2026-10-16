"""Does the portfolio hold the day's biggest movers? Recorded, not argued.

WHY

On 2026-08-21 both bots ran a full portfolio while four watchlist coins sat in
the day's top-50 by MOVE -- ENA +47.1% (4th), BCH +34.8% (15th), CRV +23.1%,
ZRO +21.6% -- and **neither held a single one of them**. Both were instead loaded
with coins ranked 89th to 601st out of 696, entered with a median of 1.85% (gpt)
and 2.91% (this bot) still to run before the day's high.

That is the North Star restated as a daily fact: the target is the day's largest
MOVE, and the question is whether the portfolio contains it. One day of that is an
anecdote. Recorded every few hours for months it becomes the answer to whether
the pipeline systematically buys the middle of moderate moves instead of the
start of large ones.

WHAT IS RECORDED

    per bot   open positions with entry price and timestamp, current price,
              unrealised P&L, the coin's day MOVE, its rank in the whole futures
              universe, and how much of that move was still ahead at entry
    per day   the watchlist coins that reached the top-N by MOVE, and which of
              them each bot held -- caught and missed, by name

Both bots are read from their own positions.json, so nothing is typed by hand and
the comparison cannot drift from what actually ran.

WHAT IT IS NOT

    Not a scoreboard for picking a winner between the two. On the day this was
    written the P&L difference was 0.68 percentage points across 8-10 positions,
    which is noise. The column that carries information is `caught` against
    `available`: how many of the day's real movers the portfolio actually held.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORE = HERE / "portfolio_vs_target.jsonl"

BOTS = {
    "claude": HERE / "positions.json",
    "gpt": Path("D:/Projects/gpt_crypto_bot/files/positions.json"),
}

FAPI = "https://fapi.binance.com/fapi/v1/ticker/24hr"


def tickers():
    with urllib.request.urlopen(FAPI, timeout=30) as r:
        return json.loads(r.read().decode())


def universe(raw):
    """Every USDT perp ranked by the day's MOVE — high against open.

    Move, not close: a coin that ran +40% and gave it back was still a move the
    bot existed to catch, and ranking by close would hide exactly those.
    """
    rows = []
    for t in raw:
        s = t.get("symbol", "")
        if not s.endswith("USDT"):
            continue
        try:
            op, hi, la = float(t["openPrice"]), float(t["highPrice"]), float(t["lastPrice"])
        except (KeyError, TypeError, ValueError):
            continue
        if op <= 0:
            continue
        rows.append({"sym": s, "move": (hi / op - 1) * 100, "high": hi, "last": la})
    rows.sort(key=lambda r: -r["move"])
    return rows


def positions(path):
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def cmd_snapshot(args):
    wl = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
    raw = tickers()
    uni = universe(raw)
    by = {r["sym"]: r for r in uni}
    rank = {r["sym"]: i + 1 for i, r in enumerate(uni)}
    topn = [r for r in uni[:args.top] if r["sym"] in wl]
    top_syms = {r["sym"] for r in topn}
    now = datetime.now(timezone.utc)

    rec = {"ts": now.isoformat(), "universe_size": len(uni), "top_n": args.top,
           "available": [{"sym": r["sym"], "rank": rank[r["sym"]],
                          "move": round(r["move"], 2)} for r in topn],
           "bots": {}}

    print("watchlist coins in the day's top-%d by MOVE: %d" % (args.top, len(topn)))
    for r in topn:
        print("   %-11s rank %3d   move %+.1f%%" % (r["sym"].replace("USDT", ""),
                                                    rank[r["sym"]], r["move"]))
    print()

    for name, path in BOTS.items():
        pos = positions(path)
        if not pos:
            print("%-8s no positions file at %s" % (name, path))
            continue
        held = []
        for sym, p in pos.items():
            e = p.get("entry_price")
            if not isinstance(e, (int, float)) or e <= 0:
                continue
            t = by.get(sym)
            held.append({
                "sym": sym,
                "entry": float(e),
                "entry_ts": p.get("entry_ts"),
                "last": t["last"] if t else None,
                "pnl_pct": ((t["last"] / e - 1) * 100) if t else None,
                "day_move": round(t["move"], 2) if t else None,
                "rank": rank.get(sym),
                # the North Star sub-metric: how much of the day's move was
                # still ahead when this position was opened
                "ahead_at_entry": ((t["high"] / e - 1) * 100) if t else None,
            })
        caught = sorted(top_syms & {h["sym"] for h in held})
        missed = sorted(top_syms - {h["sym"] for h in held})
        rec["bots"][name] = {"n": len(held), "held": held,
                             "caught": caught, "missed": missed}

        pnl = [h["pnl_pct"] for h in held if h["pnl_pct"] is not None]
        ah = [h["ahead_at_entry"] for h in held if h["ahead_at_entry"] is not None]
        rk = [h["rank"] for h in held if h["rank"]]
        q = lambda v, p: (sorted(v)[int(p * (len(v) - 1))] if v else float("nan"))
        print("%-8s %d positions | caught %d of %d | P&L med %+.2f%% | "
              "ahead at entry med %+.2f%% | rank med %s"
              % (name, len(held), len(caught), len(topn),
                 q(pnl, .5), q(ah, .5),
                 int(q(rk, .5)) if rk else "-"))
        if caught:
            print("         caught: %s" % ", ".join(s.replace("USDT", "") for s in caught))

    with io.open(STORE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = sum(1 for _ in io.open(STORE, encoding="utf-8"))
    print()
    print("appended to %s (store now holds %d snapshots)" % (STORE.name, n))


def cmd_report(args):
    if not STORE.exists():
        print("no store yet")
        return
    recs = []
    with io.open(STORE, encoding="utf-8") as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    days = sorted({r["ts"][:10] for r in recs})
    print("snapshots: %d over %d day(s): %s .. %s"
          % (len(recs), len(days), days[0], days[-1]))

    agg = defaultdict(lambda: {"caught": 0, "avail": 0, "pnl": [], "ahead": [], "n": 0})
    for r in recs:
        avail = len(r.get("available") or [])
        for name, b in (r.get("bots") or {}).items():
            a = agg[name]
            a["caught"] += len(b.get("caught") or [])
            a["avail"] += avail
            a["n"] += 1
            for h in b.get("held") or []:
                if h.get("pnl_pct") is not None:
                    a["pnl"].append(h["pnl_pct"])
                if h.get("ahead_at_entry") is not None:
                    a["ahead"].append(h["ahead_at_entry"])

    q = lambda v, p: (sorted(v)[int(p * (len(v) - 1))] if v else float("nan"))
    print()
    print("%-10s%9s%14s%14s%16s" % (
        "bot", "snaps", "caught/avail", "P&L med", "ahead at entry"))
    print("-" * 66)
    for name, a in sorted(agg.items()):
        share = (100.0 * a["caught"] / a["avail"]) if a["avail"] else float("nan")
        print("%-10s%9d%9d/%-4d%13.2f%%%15.2f%%" % (
            name, a["n"], a["caught"], a["avail"], q(a["pnl"], .5), q(a["ahead"], .5)))

    print()
    print("READ THIS")
    print("  caught/avail is the column that matters: of the watchlist coins that")
    print("  actually reached the day's top-%d by move, how many were held. P&L over"
          % (recs[-1].get("top_n", 50)))
    print("  a handful of positions is noise -- on the first day recorded, the two")
    print("  bots differed by 0.68pp across 8-10 positions and caught 0 of 4 each.")
    if len(days) < 20:
        print("  %d day(s) recorded. Not enough to call anything systematic."
              % len(days))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--top", type=int, default=50)
    s.set_defaults(fn=cmd_snapshot)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
