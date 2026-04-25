"""Inspect what features are stored on alignment entry events."""
from __future__ import annotations
import json, io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

samples = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") == "entry" and e.get("mode") == "alignment":
            samples.append(e)
            if len(samples) >= 5: break

for i, e in enumerate(samples):
    print(f"=== sample {i} ===")
    print(f"keys: {list(e.keys())}")
    feat = e.get("features") or {}
    print(f"feature keys: {list(feat.keys())}")
    print(f"features: {json.dumps(feat, indent=2, ensure_ascii=False)[:1500]}")
    print()
