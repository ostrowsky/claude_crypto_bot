"""Remaining checks of the 2026-09-25 algorithm audit, in one reproducible file.

  1. trend_quality blocks inside winner birth windows, by check
  2. first entry vs the +2.5% crossing, by timeframe
  3. logged daily_range on blocks, by timeframe (96-bar window = 24h on 15m, 4 days on 1h)
  4. candidate ranker: spread and AUC against the 4h forward peak
  5. static: threshold counts per indicator family, duplicate top-level defs,
     runtime overrides and their review dates

Numbers are quoted in docs/specs/features/algorithm-audit-0925-spec.md.
"""
import ast
import collections
import glob
import io
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _compute_early_capture as E  # noqa: E402
import config as C  # noqa: E402
import immutable_labels as IL  # noqa: E402
import label_store as LS  # noqa: E402
import pipeline_replay_validator as V  # noqa: E402

cut = E.NOW - timedelta(days=60)
wl = E.load_watchlist()
full, _, _ = E.load_uptime(cut)
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = {k for k in win if k[0] >= cut.strftime("%Y-%m-%d") and k[0] in full and k in dl}

tq = collections.Counter()
ent = collections.defaultdict(list)
dr_tf = collections.defaultdict(list)
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        ts = str(e.get("ts", ""))
        if ts < cut.strftime("%Y-%m-%d"):
            continue
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if e.get("event") == "blocked" and isinstance(e.get("daily_range"), (int, float)):
            dr_tf[e.get("tf")].append(e["daily_range"])
        key = (d.strftime("%Y-%m-%d"), e.get("sym"))
        if key not in W:
            continue
        if e.get("event") == "blocked" and e.get("reason_code") == "trend_quality" and d <= dl[key][1]:
            r = str(e.get("reason"))
            tq["price_edge" if "price edge" in r else "daily_range" if "daily_range" in r
               else "RSI" if "RSI" in r else "forecast/alt"] += 1
        if e.get("event") == "entry":
            ent[key].append((d, e.get("tf")))
print("1. trend_quality blocks in winner birth windows:", dict(tq))
rel = collections.defaultdict(list)
for k, v in ent.items():
    d, tf = sorted(v)[0]
    rel[tf].append((d - dl[k][1]).total_seconds() / 3600)
for tf, v in rel.items():
    v.sort()
    print("2. first entry vs crossing tf=%s: n=%d median %+.1f h, before crossing %.0f%%"
          % (tf, len(v), v[len(v) // 2], 100 * sum(x < 0 for x in v) / len(v)))
for tf, v in dr_tf.items():
    v.sort()
    print("3. logged daily_range on %s blocks: median %.1f%%, p75 %.1f%%, >10%% %.0f%%"
          % (tf, v[len(v) // 2], v[3 * len(v) // 4], 100 * sum(x > 10 for x in v) / len(v)))

rows, seen = [], set()
sz = os.path.getsize(FILES / "bot_events.jsonl")
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    fh.seek(max(0, sz - 90_000_000))
    fh.readline()
    for raw in fh:
        if b"ranker_quality_proba" not in raw:
            continue
        e = json.loads(raw.decode("utf-8", "replace"))
        if e.get("event") not in ("blocked", "entry") or str(e.get("ts")) < "2026-09-08":
            continue
        if not isinstance(e.get("ranker_quality_proba"), (int, float)):
            continue
        d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        k = (e["sym"], d.strftime("%m-%d %H"))
        if k in seen:
            continue
        seen.add(k)
        p = V.forward_peak(e["sym"], str(e.get("tf")), d, float(e["price"]))
        if p is not None:
            rows.append((e["ranker_quality_proba"], e.get("ranker_final_score"), e.get("ranker_top_gainer_prob"), p, d.strftime("%m-%d")))


def auc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    r = np.argsort(np.argsort(s)) + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


pk = np.array([r[3] for r in rows])
y = pk >= 3.0
q = np.array([r[0] for r in rows])
byday = collections.defaultdict(list)
for r in rows:
    byday[r[4]].append(r[0])
print("4. ranker: n=%d, quality_proba std %.4f, within-day std %.4f" % (
    len(rows), q.std(), np.median([np.std(v) for v in byday.values() if len(v) > 5])))
for name, j in (("quality_proba", 0), ("final_score", 1), ("top_gainer_prob", 2)):
    s = np.array([r[j] if isinstance(r[j], (int, float)) else np.nan for r in rows])
    m = ~np.isnan(s)
    print("   AUC %-15s vs 4h peak>=3%%: %.3f (base %.1f%%)" % (name, auc(y[m], s[m]), 100 * y[m].mean()))

fam = collections.Counter()
for k in dir(C):
    v = getattr(C, k)
    if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool):
        for g in ("RSI", "RANGE", "EDGE", "ADX", "VOL"):
            if g in k and any(t in k for t in ("MAX", "MIN", "HI", "LO")):
                fam[g] += 1
print("5. threshold keys per family:", dict(fam))
for f in sorted(glob.glob(str(FILES / "*.py"))):
    n = os.path.basename(f)
    if n.startswith(("test_", "_backtest", "_diag", "_tmp", "_audit")):
        continue
    try:
        tree = ast.parse(io.open(f, encoding="utf-8").read())
    except Exception:
        continue
    cnt = collections.Counter(x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)))
    dup = [k for k, v in cnt.items() if v > 1]
    if dup:
        print("   duplicate top-level defs:", n, dup)
ov = json.load(io.open(FILES.parent / ".runtime/release/runtime_overrides.json", encoding="utf-8"))
ov = ov.get("overrides", ov)
for k, v in ov.items():
    print("   override %-46s value=%s review_by=%s provenance=%s" % (k, v.get("value"), v.get("review_by"), v.get("provenance")))
