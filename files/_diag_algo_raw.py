"""Dump raw ALGO entry/exit events to see schema."""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

NOW = datetime.now(timezone.utc)
CUT = NOW - timedelta(days=14)

events = []
with io.open(ROOT / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if "ALGO" not in ln: continue
        try: e = json.loads(ln)
        except: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if sym != "ALGOUSDT": continue
        ts = e.get("ts","")
        try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        e["_dt"] = dt
        events.append(e)

events.sort(key=lambda e: e["_dt"])

print(f"=== Raw entry/exit events (full keys) ===\n")
for e in events:
    typ = e.get("event","")
    if typ not in ("entry","exit"): continue
    ts = e["_dt"].strftime("%m-%d %H:%M")
    keys = sorted(e.keys())
    print(f"\n{ts}  event={typ}  keys={keys}")
    for k in keys:
        if k.startswith("_"): continue
        v = e[k]
        if isinstance(v, (dict,list)):
            v = json.dumps(v)[:80]
        print(f"  {k:20s} = {v}")
