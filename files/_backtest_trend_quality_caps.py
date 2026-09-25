"""trend_quality's price_edge / daily_range / RSI caps: do they remove good entries?

WHY THIS EXISTS

After the QNT day (2026-09-24, +29%), every later impulse_speed candidate was
blocked by trend_quality on `price edge 4.6-6.8% > 3.20%` and `daily_range
34% > 10%`. The strategy-level lateness caps were refuted the same day
(_backtest_lateness_caps.py); these are a separate set, inside
monitor._trend_entry_quality_guard_reason, applied to 15m trend candidates
(impulse_speed falls back to trend while it is curtailed). The forecast/alt
path of the same guard was refuted on 2026-09-18
(_backtest_trend_quality_zero_forecast.py); this covers the other three.

METHOD

The guard's checks run in order and the first that fails is logged:
    price_edge > 3.2% (bull 4.0)  ->  daily_range > 10% (bull 14)  ->
    RSI > 72 (bull 76)  ->  forecast < 0.25 unless vol>=1.2 & ADX>=24 & slope>=0.35
Relaxing one cap only admits a row if every LATER check also passes, so each
logged rejection is re-judged by applying the CURRENT thresholds and a RELAXED
set to the same logged inputs (not by what the gate decided on the day -- the
thresholds moved over the period). Inputs come from the event fields, or from
the reason text on older rows that lack them. The forecast is only logged on
`weak 15m trend` rows; elsewhere it is taken as 0.0 -- the value 78% of known
rows carry -- which makes a relaxed row pass only via the alt path (stated;
conservative towards the cap).

Band for a cap step = rows blocked under the current thresholds that pass under
the relaxed one. Compared, by the SAME outcome engine, against trend/15m
entries (the matched control -- only entries log their mode; they also passed
every later gate, which biases against relaxing):
    peak_4h, trough_4h   next 16 x 15m bars
    trailed              ATR trail from the row's price, trail_k 2.0, 96-bar cap
                         (_backtest_weak_exit_above_breakeven.replay)
Deduplicated to one row per (symbol, UTC hour, group). By month, and the share
of rows on immutable top-20 winner-days (lift vs control).

VERDICT 2026-09-25: REFUTED as a lever -- the three caps stay. Do not re-test
without new evidence.

2026-03-10 .. 2026-09-25: 6 240 trend_quality rejections, re-judged under
today's thresholds (weak 3 564, RSI 1 286, price_edge 381, daily_range 378,
unreplayable 615); control 653 trend/15m entries (trailed mean -0.16%, median
-0.42%, peak>=5% 6.6%, 7.6% on winner-days).

  price_edge   of 381 rows it stops, lifting it frees only 17: 232 are then
               stopped by daily_range and 113 by RSI. Nothing to gain.
  RSI          72 -> 80 frees 683 rows: trailed -0.12% / +0.09%, winner-day
               lift 1.0x / 0.8x. No edge.
  daily_range  10 -> 20 frees 115: mean +0.49%, difference vs entries +0.62%,
               95% CI [+0.01, +1.38] -- but median -0.46%, and without its 3 best
               trades (+11.4, +20.8, +21.2) the mean is +0.02%. It is one of 12
               grid points tested, so one borderline interval is expected by
               chance. Lifted entirely: +0.55%, CI [-0.01, +1.21].
  all three    1 109 rows freed: trailed -0.02% vs -0.16%, difference +0.14%,
               CI [-0.05, +0.34], 4 of 5 months >= entries.

QNT 2026-09-24/25, which raised it: price_edge 4.6-6.8% AND daily_range ~34%;
relaxing price_edge frees nothing while daily_range blocks, and 34% is past every
band with a usable n.
"""
from __future__ import annotations

import collections
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _backtest_weak_exit_above_breakeven as W  # noqa: E402
import immutable_labels as IL  # noqa: E402

CUR = dict(pe=cfg.TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT, pe_b=cfg.TREND_15M_QUALITY_PRICE_EDGE_MAX_BULL_DAY_PCT,
           dr=cfg.TREND_15M_QUALITY_DAILY_RANGE_MAX, dr_b=cfg.TREND_15M_QUALITY_DAILY_RANGE_MAX_BULL_DAY,
           rsi=cfg.TREND_15M_QUALITY_RSI_MAX, rsi_b=cfg.TREND_15M_QUALITY_RSI_MAX_BULL_DAY,
           fc=cfg.TREND_15M_QUALITY_FORECAST_MIN, av=cfg.TREND_15M_QUALITY_ALT_VOL_MIN,
           aa=cfg.TREND_15M_QUALITY_ALT_ADX_MIN, asl=cfg.TREND_15M_QUALITY_ALT_SLOPE_MIN)
GRID = {"pe": [4.0, 5.0, 6.0, 8.0], "dr": [14.0, 20.0, 30.0, 1e9], "rsi": [76.0, 80.0, 85.0, 1e9]}

RE_PE = re.compile(r"price edge ([\d.]+)%")
RE_DR = re.compile(r"daily_range ([\d.]+)%")
RE_RSI = re.compile(r"guard: RSI ([\d.]+)")
RE_WEAK = re.compile(r"forecast (-?[\d.]+) < [\d.]+, vol ([\d.]+), ADX ([\d.]+), slope (-?[\d.]+)")


def num(e, k):
    v = e.get(k)
    return float(v) if isinstance(v, (int, float)) else None


def inputs(e):
    """Guard inputs from fields, falling back to the reason text."""
    r = str(e.get("reason") or "")
    x = {"pe": num(e, "price_edge_ema20_pct"), "dr": num(e, "daily_range"), "rsi": num(e, "rsi"),
         "vol": num(e, "vol_x"), "adx": num(e, "adx"), "slope": num(e, "slope_pct"),
         "bull": bool(e.get("is_bull_day")), "fc": None}
    if x["pe"] is None and num(e, "ema20") and num(e, "price"):
        x["pe"] = (e["price"] / e["ema20"] - 1) * 100
    m = RE_PE.search(r)
    if m and x["pe"] is None:
        x["pe"] = float(m.group(1))
    m = RE_DR.search(r)
    if m and x["dr"] is None:
        x["dr"] = float(m.group(1))
    m = RE_RSI.search(r)
    if m and x["rsi"] is None:
        x["rsi"] = float(m.group(1))
    m = RE_WEAK.search(r)
    if m:
        x["fc"], x["vol"], x["adx"], x["slope"] = map(float, m.groups())
    return x


def guard(x, t):
    """First failing check, or None. None inputs -> 'unknown' (not replayable)."""
    b = x["bull"]
    for key, lim in (("pe", t["pe_b" if b else "pe"]), ("dr", t["dr_b" if b else "dr"]),
                     ("rsi", t["rsi_b" if b else "rsi"])):
        if x[key] is None:
            return "unknown"
        if x[key] > lim:
            return key
    fc = 0.0 if x["fc"] is None else x["fc"]
    if fc >= t["fc"]:
        return None
    if None in (x["vol"], x["adx"], x["slope"]):
        return "unknown"
    if x["vol"] >= t["av"] and x["adx"] >= t["aa"] and x["slope"] >= t["asl"]:
        return None
    return "weak"


def relaxed(key, v):
    t = dict(CUR)
    t[key] = v
    t[key + "_b"] = max(v, CUR[key + "_b"])
    return t


def load():
    rows, seen = [], set()
    with io.open(FILES / "bot_events.jsonl", "rb") as fh:
        for raw in fh:
            if b'"15m"' not in raw:
                continue
            is_tq = b"trend quality guard" in raw
            if not is_tq and b'"entry"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("tf") != "15m":
                continue
            ev = e.get("event")
            if ev == "entry":
                if (e.get("signal_mode") or e.get("mode")) != "trend":
                    continue
                grp = "ENTRY"
            elif ev == "blocked" and is_tq:
                grp = "TQ"
            else:
                continue
            px = e.get("price")
            if not e.get("sym") or not isinstance(px, (int, float)) or px <= 0:
                continue
            d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            k = (e["sym"], d.strftime("%Y-%m-%d %H"), grp)
            if k in seen:
                continue
            seen.add(k)
            rows.append({"sym": e["sym"], "dt": d, "px": float(px), "grp": grp, "x": inputs(e)})
    return rows


_B = {}


def outcome(r):
    if r["sym"] not in _B:
        b = TD.bars_15m(r["sym"])
        _B[r["sym"]] = (b, {x[0]: i for i, x in enumerate(b)})
    bars, idx = _B[r["sym"]]
    i = idx.get(r["dt"].replace(minute=(r["dt"].minute // 15) * 15, second=0, microsecond=0))
    if i is None or i + 16 >= len(bars):
        return None
    fut = bars[i + 1:i + 17]
    px = r["px"]
    return {"peak": (max(b[2] for b in fut) / px - 1) * 100, "trough": (min(b[3] for b in fut) / px - 1) * 100,
            "trail": W.replay(bars, i, i, px, 2.0, "trend", cfg, 96)}


def summ(rs):
    o = [r["o"] for r in rs if r.get("o") and r["o"]["trail"] is not None]
    n = len(o)
    if not n:
        return None
    t = sorted(x["trail"] for x in o)
    return {"n": n, "peak": sum(x["peak"] for x in o) / n, "trough": sum(x["trough"] for x in o) / n,
            "trail": sum(t) / n, "trail_med": t[n // 2], "win": sum(1 for x in t if x > 0) / n,
            "big": sum(1 for x in o if x["peak"] >= 5) / n}


def line(name, s, ref=None):
    if not s:
        print("  %-28s n<1" % name)
        return
    print("  %-28s n=%5d  peak4h %5.2f%%  trough %6.2f%%  trailed mean %+6.2f%%  med %+6.2f%%  "
          "trail>0 %3.0f%%  peak>=5%% %4.1f%%" % (name, s["n"], s["peak"], s["trough"], s["trail"],
                                              s["trail_med"], 100 * s["win"], 100 * s["big"]))


def main():
    rows = load()
    tq = [r for r in rows if r["grp"] == "TQ"]
    ent = [r for r in rows if r["grp"] == "ENTRY"]
    for r in tq:
        r["now"] = guard(r["x"], CUR)
    print("rows: TQ rejections %d, trend/15m entries %d  (%s .. %s)" % (
        len(tq), len(ent), min(r["dt"] for r in rows).date(), max(r["dt"] for r in rows).date()))
    print("re-judged under TODAY's thresholds:", dict(collections.Counter(r["now"] for r in tq).most_common()))
    print("  (None = would pass today -- rejected under an older threshold; excluded)")
    for r in tq + ent:
        r["o"] = outcome(r)
    ctrl = summ(ent)
    print("\nCONTROL")
    line("trend/15m entries", ctrl)
    blocked_now = [r for r in tq if r["now"] in ("pe", "dr", "rsi", "weak")]

    wl = json.load(io.open(FILES / "watchlist.json", encoding="utf-8"))
    wl = wl if isinstance(wl, list) else wl.get("symbols", wl)
    win, _ = IL.winners_by_day(top_n=20, watchlist=set(wl), rank_before_filter=True)

    def lift(rs):
        cd = {(r["dt"].strftime("%Y-%m-%d"), r["sym"]) for r in rs}
        return len(cd & win) / max(1, len(cd)), len(cd)

    base_w, _n = lift(ent)
    print("  entries on immutable winner-days: %.1f%% of %d coin-days" % (100 * base_w, _n))

    for key, name in (("pe", "price_edge"), ("dr", "daily_range"), ("rsi", "RSI")):
        print("\n=== %s cap: today %.1f (bull %.1f) ===" % (name, CUR[key], CUR[key + "_b"]))
        prev = set()
        own = [r for r in blocked_now if r["now"] == key]
        print("  rejections whose FIRST failing check is %s today: %d" % (name, len(own)))
        for v in GRID[key]:
            t = relaxed(key, v)
            passing = [r for r in blocked_now if guard(r["x"], t) is None]
            band = [r for r in passing if id(r) not in prev]
            prev |= {id(r) for r in passing}
            s = summ(band)
            lab = "%s -> %s" % (CUR[key], "no cap" if v > 1e8 else v)
            line(lab + " (new rows)", s)
            if band:
                w, n = lift(band)
                print("  %-28s winner-day share %.1f%% (%.1fx control), coin-days %d" % ("", 100 * w, w / base_w if base_w else 0, n))
        # stopped anyway: rows the relaxed check would free but a later check blocks
        t = relaxed(key, 1e9)
        after = collections.Counter(guard(r["x"], t) for r in own)
        print("  of the %d, with %s lifted entirely: %s" % (len(own), name, dict(after.most_common())))

    print("\n=== by month: rows freed by lifting ALL THREE caps vs entries (trailed mean, n) ===")
    t_all = dict(CUR, pe=1e9, pe_b=1e9, dr=1e9, dr_b=1e9, rsi=1e9, rsi_b=1e9)
    freed = [r for r in blocked_now if guard(r["x"], t_all) is None]
    bm = collections.defaultdict(lambda: {"F": [], "E": []})
    for r in freed:
        if r.get("o") and r["o"]["trail"] is not None:
            bm[r["dt"].strftime("%Y-%m")]["F"].append(r["o"]["trail"])
    for r in ent:
        if r.get("o") and r["o"]["trail"] is not None:
            bm[r["dt"].strftime("%Y-%m")]["E"].append(r["o"]["trail"])
    better = scored = 0
    for m in sorted(bm):
        f, e = bm[m]["F"], bm[m]["E"]
        if len(f) >= 20 and len(e) >= 20:
            scored += 1
            better += sum(f) / len(f) >= sum(e) / len(e)
        print("  %s  freed %+6.2f%% [%4d]   entries %+6.2f%% [%4d]" % (
            m, sum(f) / len(f) if f else float("nan"), len(f), sum(e) / len(e) if e else float("nan"), len(e)))
    print("  months (n>=20 each) where freed >= entries: %d of %d" % (better, scored))
    line("ALL THREE lifted, freed rows", summ(freed))
    line("control: entries", ctrl)

    print("\nsame-bucket check, price_edge (trailed mean [n]):")
    for lo, hi in ((0, 2), (2, 3.2), (3.2, 5), (5, 8), (8, 1e9)):
        f = [r["o"]["trail"] for r in freed if r.get("o") and r["o"]["trail"] is not None and r["x"]["pe"] is not None and lo <= r["x"]["pe"] < hi]
        e = [r["o"]["trail"] for r in ent if r.get("o") and r["o"]["trail"] is not None and r["x"]["pe"] is not None and lo <= r["x"]["pe"] < hi]
        fmt = lambda v: ("%+6.2f%% [%4d]" % (sum(v) / len(v), len(v))) if len(v) >= 20 else "      --      "
        print("  pe %4s-%-4s  freed %s   entries %s" % (lo, "" if hi > 1e8 else hi, fmt(f), fmt(e)))


if __name__ == "__main__":
    main()
