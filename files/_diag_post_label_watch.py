"""The 24-hour read on the peak-label switch, and trend_quality on the new flow.

Two questions, both fixed in writing before the watch began
(docs/specs/features/peak-training-label-spec.md):

  HALF 1  does ml_zone admit 30-60% of candidates?
  HALF 2  do the coins it admits move MORE than the ones it rejects?

Half 2 is the one that matters and could not be read at hour two: it needs
forward bars that did not exist yet. Now they do.

And the follow-on question the switch created. trend_quality used to sit behind
an inert gate -- ml_zone admitted 100% on the old scale, so trend_quality was
judging the raw candidate flow. It now judges only what a working ml_zone passed,
which is a different and better population. Its old verdict (rejects 1.20% vs
pool 1.14% -- filtering nothing) was measured on the old population and does not
transfer, so it is re-measured here.

METHOD

Decisions are read from bot_events.jsonl since the restart. A candidate blocked
with reason_code=ml_zone was REJECTED by it; one blocked by any later gate, or
entered, PASSED it -- that is the split, taken from the pipeline order rather
than from ml_proba, so it reflects what the bot actually did.

Forward moves come from freshly fetched klines, not the cache: the cache is
refreshed at 06:00 and would not cover this afternoon's decisions, and a
half-covered window would read as "these candidates went nowhere".

Deduplicated by symbol-hour-gate: a gate that re-fires every poll would
otherwise weight its own opinion by how often the loop ran.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVENTS = HERE / "bot_events.jsonl"

_KL: dict = {}


def klines(sym: str, interval: str = "1h", limit: int = 200):
    key = "%s|%s" % (sym, interval)
    if key in _KL:
        return _KL[key]
    url = ("https://fapi.binance.com/fapi/v1/klines?"
           + urllib.parse.urlencode(dict(symbol=sym, interval=interval, limit=limit)))
    out = []
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            for b in json.loads(r.read().decode()):
                out.append((datetime.fromtimestamp(int(b[0]) / 1000, tz=timezone.utc),
                            float(b[2]), float(b[4])))   # ts, high, close
    except Exception:
        out = []
    _KL[key] = out
    time.sleep(0.05)
    return out


def forward_peak(sym, when, px, bars_ahead):
    ks = klines(sym)
    if not ks:
        return None
    fut = [k for k in ks if k[0] > when][:bars_ahead]
    # Require the full window: a partly-resolved candidate would report a small
    # peak and read as a bad one, which biases against the most recent decisions.
    if len(fut) < bars_ahead:
        return None
    return (max(k[1] for k in fut) / px - 1.0) * 100.0


def load(since: str, tail_mb: int = 60):
    sz = os.path.getsize(EVENTS)
    off = max(0, sz - tail_mb * 1_000_000)
    out = []
    with io.open(EVENTS, "rb") as fh:
        fh.seek(off)
        if off:
            fh.readline()
        for raw in fh:
            if b'"blocked"' not in raw and b'"entry"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            ev = e.get("event")
            if ev not in ("blocked", "entry"):
                continue
            ts = str(e.get("ts") or "")
            if ts < since:
                continue
            sym, px = e.get("sym"), e.get("price")
            if not sym or not isinstance(px, (int, float)) or px <= 0:
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            rc = str(e.get("reason_code") or "")
            if ev == "blocked" and (not rc or rc == "None"):
                rc = str(e.get("reason") or "?").split(":")[0].strip()[:24]
            out.append({
                "sym": sym, "hour": d.replace(minute=0, second=0, microsecond=0),
                "px": float(px), "ev": ev, "rc": rc if ev == "blocked" else "ENTRY",
                "ml": e.get("ml_proba"),
            })
    return out


def stat(name, vals, base=None):
    if not vals:
        print("%-26s%8s" % (name, "n=0"))
        return
    v = sorted(vals)
    med = v[len(v) // 2]
    p75 = v[int(0.75 * (len(v) - 1))]
    s3 = 100.0 * sum(1 for x in v if x > 3) / len(v)
    s5 = 100.0 * sum(1 for x in v if x > 5) / len(v)
    flag = ""
    if base is not None and med > base:
        flag = "  <- above the compared set"
    print("%-26s%8d%11.2f%%%11.2f%%%9.0f%%%9.0f%%%s"
          % (name, len(v), med, p75, s3, s5, flag))
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-07T19:31",
                    help="UTC; the restart that put the peak label live")
    ap.add_argument("--bars", type=int, default=5,
                    help="forward bars; 5 matches the label horizon")
    ap.add_argument("--floor", type=float, default=0.15)
    args = ap.parse_args()

    rows = load(args.since)
    print("decisions since %s UTC: %d" % (args.since, len(rows)))
    if not rows:
        print("nothing to read yet")
        return

    seen = {}
    for r in rows:
        seen.setdefault((r["sym"], r["hour"], r["rc"]), r)
    rows = list(seen.values())
    print("deduplicated by symbol-hour-gate: %d" % len(rows))
    hours = sorted(set(r["hour"] for r in rows))
    print("window: %s .. %s  (%d hours)"
          % (hours[0].strftime("%m-%d %H:%M"), hours[-1].strftime("%m-%d %H:%M"),
             len(hours)))

    by_gate = collections.Counter(r["rc"] for r in rows)
    print()
    print("what stopped each candidate:")
    for k, v in by_gate.most_common(10):
        print("   %-26s %5d  (%.0f%%)" % (k[:25], v, 100.0 * v / len(rows)))

    # ---- HALF 1: admit rate ------------------------------------------------
    rejected_ml = [r for r in rows if r["rc"] == "ml_zone"]
    passed_ml = [r for r in rows if r["rc"] != "ml_zone"]
    admit = len(passed_ml) / len(rows)
    print()
    print("=" * 88)
    print("HALF 1 — does ml_zone admit 30-60%%?")
    print("=" * 88)
    print("passed ml_zone %d, rejected %d  ->  admit rate %.1f%%"
          % (len(passed_ml), len(rejected_ml), 100 * admit))
    verdict1 = "PASS" if 0.30 <= admit <= 0.60 else (
        "ROLL BACK (blackout re-forming)" if admit < 0.10 else
        "ROLL BACK (gate inert again)" if admit > 0.90 else
        "outside the keep band but not a rollback trigger")
    print("verdict: %s" % verdict1)
    mls = [r["ml"] for r in rows if isinstance(r["ml"], (int, float))]
    if mls:
        mls_sorted = sorted(mls)
        print("ml_proba live: median %.4f  p90 %.4f  max %.4f  (floor %.2f)"
              % (mls_sorted[len(mls_sorted) // 2],
                 mls_sorted[int(0.9 * (len(mls_sorted) - 1))],
                 mls_sorted[-1], args.floor))

    # ---- HALF 2: do the admitted move more? --------------------------------
    print()
    print("=" * 88)
    print("HALF 2 — do the coins it ADMITS move more than the ones it REJECTS?")
    print("=" * 88)
    print("forward peak over the next %d bars, from freshly fetched klines" % args.bars)
    print()
    print("%-26s%8s%12s%12s%10s%10s" % ("set", "n", "median", "p75", ">3%", ">5%"))
    print("-" * 88)
    fwd = {}
    for name, group in (("REJECTED by ml_zone", rejected_ml),
                        ("ADMITTED by ml_zone", passed_ml)):
        vals = []
        for r in group:
            f = forward_peak(r["sym"], r["hour"], r["px"], args.bars)
            if f is not None:
                vals.append(f)
        fwd[name] = vals
    med_rej = stat("REJECTED by ml_zone", fwd["REJECTED by ml_zone"])
    med_adm = stat("ADMITTED by ml_zone", fwd["ADMITTED by ml_zone"], base=med_rej)
    print("-" * 88)
    if med_rej is not None and med_adm is not None:
        ok2 = med_adm > med_rej
        print("verdict: %s  (admitted %.2f%% vs rejected %.2f%%)"
              % ("PASS" if ok2 else "FAIL — the gate is selecting the wrong way",
                 med_adm, med_rej))
    else:
        print("verdict: not enough resolved rows to judge")

    # ---- trend_quality on the new population -------------------------------
    print()
    print("=" * 88)
    print("trend_quality, re-measured on what a WORKING ml_zone now passes")
    print("=" * 88)
    tq_rej = [r for r in passed_ml if r["rc"] == "trend_quality"]
    tq_rest = [r for r in passed_ml if r["rc"] != "trend_quality"]
    print("of the %d that passed ml_zone: %d stopped by trend_quality, %d went further"
          % (len(passed_ml), len(tq_rej), len(tq_rest)))
    print()
    print("%-26s%8s%12s%12s%10s%10s" % ("set", "n", "median", "p75", ">3%", ">5%"))
    print("-" * 88)
    v_rej, v_rest = [], []
    for group, sink in ((tq_rej, v_rej), (tq_rest, v_rest)):
        for r in group:
            f = forward_peak(r["sym"], r["hour"], r["px"], args.bars)
            if f is not None:
                sink.append(f)
    m_rest = stat("passed trend_quality", v_rest)
    m_rej = stat("REJECTED by trend_quality", v_rej, base=m_rest)
    print("-" * 88)
    # A median alone decided this line in the first version and over-claimed:
    # on 2026-09-08 the rejects had the higher MEDIAN (1.34 vs 1.18) while the
    # passed set had more BIG movers (>3%: 26 vs 21, >5%: 13 vs 10). Under a goal
    # of catching the largest moves the tail is what counts, so a verdict that
    # reads only the middle of the distribution points the wrong way.
    def _share(v, t):
        return 100.0 * sum(1 for x in v if x > t) / len(v) if v else float("nan")
    if v_rej and v_rest:
        med_worse = m_rej > m_rest
        tail_worse = _share(v_rej, 3) > _share(v_rest, 3) and             _share(v_rej, 5) > _share(v_rest, 5)
        if med_worse and tail_worse:
            print("trend_quality is removing the BETTER half -- the section-7 failure")
        elif not med_worse and not tail_worse:
            print("trend_quality's rejects are the weaker half -- it is doing its job")
        else:
            print("MIXED: median says %s, big-mover tail says %s -- one window and"
                  % ("rejects are better" if med_worse else "rejects are worse",
                     "rejects are better" if tail_worse else "rejects are worse"))
            print("       n=%d/%d is too small to call; re-read on more data."
                  % (len(v_rej), len(v_rest)))

    print()
    print("READ THIS")
    print("  Half 2 is the half that matters: a gate can hit any admit rate and")
    print("  still be selecting the wrong candidates. The admit rate only says the")
    print("  floor sits where the remap predicted.")
    print("  22 hours is one window, not a trend -- and a market that moved one way")
    print("  all day would flatter or damn any gate equally.")


if __name__ == "__main__":
    main()
