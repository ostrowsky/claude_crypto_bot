"""Where are the day's winners lost? A funnel over immutable top-20 winner-days.

Population: watchlist INTERSECT global top-20 (later-EOD klines), days the bot
was fully up, with an intraday +2.5% crossing time T (label tiers).
Birth window = [UTC open, T]. For each winner-day:

  A  a live entry rule fires on 15m inside the birth window (strategy replay)
  B  the live bot logged a candidate for the coin inside the window
  C  ... and what stopped it
  D  first entry of the day, relative to T
  E  exits of the day: reason, and how much of the day's move was left

Rule-layer replay is 15m only; 1h-monitored coins are flagged separately.
"""
import bisect
import collections
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _compute_early_capture as E  # noqa: E402
import immutable_labels as IL  # noqa: E402
import label_store as LS  # noqa: E402

DAYS = 60
cut = E.NOW - timedelta(days=DAYS)
wl = E.load_watchlist()
full, partial, _ = E.load_uptime(cut)
win, eod = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = sorted(k for k in win if k[0] >= cut.strftime("%Y-%m-%d") and k[0] in full and k in dl)
print("winner-days (%d days, bot fully up, with crossing time): %d" % (DAYS, len(W)))

# replay fires (15m), signal available at bar close
fires = collections.defaultdict(list)
for l in io.open(FILES.parent / ".runtime/backtests/lateness_rows.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = datetime.fromisoformat(r["ts"]) + timedelta(minutes=15)
    fires[(r["day"], r["sym"])].append((t, r["band"], r["rule"], r["dr"]))

# bot events for the winner coins
want = {s for _, s in W}
evs = collections.defaultdict(list)
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if not any(s.encode() in raw for s in ("USDT",)):
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        sym = e.get("sym")
        if sym not in want or e.get("event") not in ("blocked", "entry", "exit", "cooldown_start"):
            continue
        ts = str(e.get("ts", ""))
        if ts < cut.strftime("%Y-%m-%d"):
            continue
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        e["_dt"] = d
        evs[sym].append(e)
for v in evs.values():
    v.sort(key=lambda e: e["_dt"])


def gate(e):
    rc = e.get("reason_code")
    if rc:
        return rc
    r = str(e.get("reason") or "")
    for k, name in (("bandit skip", "bandit"), ("портфель полон", "portfolio_full"), ("MTF", "mtf"),
                    ("continuation", "late_continuation"), ("группа", "group_cap")):
        if k in r:
            return name
    return re.sub(r"[\d.]+", "#", r)[:30]


bars1h = {}


def day_path(sym, day):
    if sym not in bars1h:
        bars1h[sym] = LS._read_1h_bars(sym)
    d0 = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    return [b for b in bars1h[sym] if d0 <= b[0] < d0 + 86_400_000]


stage = collections.Counter()
gate_first = collections.Counter()
gate_any = collections.Counter()
nolive_reason = collections.Counter()
entry_rel = []
entry_tf = collections.Counter()
exits = []
rule_first = collections.Counter()
t_after_open = []
for day, sym in W:
    open_dt, T = dl[(day, sym)]
    t_after_open.append((T - open_dt).total_seconds() / 3600)
    f = sorted(fires.get((day, sym), []))
    f_birth = [x for x in f if open_dt <= x[0] <= T]
    ev_day = [e for e in evs.get(sym, []) if e["_dt"].strftime("%Y-%m-%d") == day]
    ev_birth = [e for e in ev_day if e["_dt"] <= T and e.get("event") in ("blocked", "entry")]
    ent_birth = [e for e in ev_birth if e["event"] == "entry"]
    if f_birth:
        rule_first[f_birth[0][1]] += 1
    # open position carried in from before the day?
    prior = [e for e in evs.get(sym, []) if e["_dt"] < open_dt and e["event"] in ("entry", "exit")]
    holding = bool(prior) and prior[-1]["event"] == "entry"
    if ent_birth:
        stage["4 entered BEFORE the +2.5% crossing"] += 1
    elif holding:
        stage["0 already holding the coin at the open"] += 1
    elif ev_birth:
        stage["3 candidate(s) in the window, all blocked"] += 1
        gate_first[gate(ev_birth[0])] += 1
        for gname in {gate(e) for e in ev_birth}:
            gate_any[gname] += 1
    elif f_birth:
        stage["2 a 15m rule fired, but no live candidate"] += 1
        cd = [e for e in evs.get(sym, []) if e["event"] == "cooldown_start" and open_dt - timedelta(hours=5) <= e["_dt"] <= T]
        tfs = {e.get("tf") for e in ev_day}
        if cd:
            nolive_reason["cooldown after an exit"] += 1
        elif tfs == {"1h"}:
            nolive_reason["coin was monitored on 1h that day"] += 1
        elif f_birth[0][1] != "FIRE":
            nolive_reason["only with bull/lifted caps (not live)"] += 1
        else:
            nolive_reason["unexplained (not polled? pre-gate return?)"] += 1
    else:
        stage["1 no 15m rule fired in the window"] += 1
    ent_day = [e for e in ev_day if e["event"] == "entry"]
    if ent_day:
        e0 = ent_day[0]
        entry_rel.append((e0["_dt"] - T).total_seconds() / 3600)
        entry_tf[e0.get("tf")] += 1
    # exits of the day and the move left after each
    path = day_path(sym, day)
    for x in [e for e in ev_day if e["event"] == "exit" and isinstance(e.get("exit_price"), (int, float))]:
        after = [b for b in path if b[0] >= int(x["_dt"].timestamp() * 1000) - 3_600_000 + 1]
        hi_after = max((b[2] for b in after), default=None)
        left = (hi_after / x["exit_price"] - 1) * 100 if hi_after else None
        exits.append({"reason": re.sub(r"[\d.]+", "#", str(x.get("reason") or ""))[:55],
                      "pnl": x.get("pnl_pct"), "left": left, "mode": x.get("mode"), "tf": x.get("tf")})

n = len(W)
print("\nhours from UTC open to the +2.5%% crossing: median %.1f" % sorted(t_after_open)[len(t_after_open) // 2])
print("\n=== BIRTH WINDOW [open, +2.5%% crossing] -- %d winner-days ===" % n)
for k in sorted(stage):
    print("  %-48s %4d  (%4.1f%%)" % (k, stage[k], 100.0 * stage[k] / n))
print("\n  stage 3 -- the gate that stopped the FIRST candidate:")
for g, c in gate_first.most_common(12):
    print("      %-30s %4d" % (g, c))
print("  stage 3 -- gates present at all in the window (a day can count several):")
for g, c in gate_any.most_common(12):
    print("      %-30s %4d" % (g, c))
print("  stage 2 -- why no live candidate:")
for g, c in nolive_reason.most_common():
    print("      %-44s %4d" % (g, c))
print("  first replay rule band inside the window:", dict(rule_first))

er = sorted(entry_rel)
print("\n=== FIRST ENTRY OF THE DAY vs the crossing (hours; negative = before) ===")
print("  winner-days with an entry: %d of %d; p10 %+.1f  p25 %+.1f  median %+.1f  p75 %+.1f  p90 %+.1f" % (
    len(er), n, er[len(er) // 10], er[len(er) // 4], er[len(er) // 2], er[3 * len(er) // 4], er[9 * len(er) // 10]))
print("  first-entry timeframe:", dict(entry_tf))

print("\n=== EXITS ON WINNER-DAYS: how much of the day's move was left after the exit ===")
by = collections.defaultdict(list)
for x in exits:
    by[x["reason"]].append(x)
print("  %-56s %5s %8s %10s %9s" % ("exit reason (numbers masked)", "n", "pnl med", "left med", "left>=5%"))
for r, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:14]:
    p = sorted(x["pnl"] for x in v if isinstance(x["pnl"], (int, float)))
    L = sorted(x["left"] for x in v if x["left"] is not None)
    if not L:
        continue
    print("  %-56s %5d %7.2f%% %9.2f%% %8.0f%%" % (r, len(v), p[len(p) // 2] if p else float("nan"), L[len(L) // 2],
                                                100.0 * sum(1 for y in L if y >= 5) / len(L)))
json.dump({"stage": stage, "gate_first": gate_first, "gate_any": gate_any, "nolive": nolive_reason,
           "entry_rel": er, "n": n}, io.open(FILES.parent / ".runtime/backtests/audit_funnel.json", "w", encoding="utf-8"), default=str)
