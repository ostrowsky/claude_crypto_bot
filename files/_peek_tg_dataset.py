"""Peek at top_gainer_dataset.jsonl and critic_dataset.jsonl structure."""
import json, io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

print("=== top_gainer_dataset.jsonl: first 3 rows, all keys ===")
with io.open(FILES / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for i, ln in enumerate(f):
        if not ln.strip(): continue
        try: row = json.loads(ln)
        except: continue
        print(f"Row {i}: keys={sorted(row.keys())}")
        # show non-feature values
        for k in sorted(row.keys()):
            if k not in ("f", "features"):
                print(f"  {k}: {row[k]!r}")
        if i >= 2: break

print("\n=== critic_dataset.jsonl: first taken row with labels ===")
with io.open(FILES / "critic_dataset.jsonl", encoding="utf-8") as f:
    found = 0
    for ln in f:
        if not ln.strip(): continue
        try: row = json.loads(ln)
        except: continue
        dec = row.get("decision") or {}
        if dec.get("action") != "take": continue
        labels = row.get("labels") or {}
        print(f"Keys: {sorted(row.keys())}")
        print(f"labels keys: {sorted(labels.keys())}")
        print(f"labels: {labels}")
        print(f"decision keys: {sorted(dec.keys())}")
        found += 1
        if found >= 2: break
