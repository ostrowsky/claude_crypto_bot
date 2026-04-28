"""Peek at top_gainer_dataset.jsonl schema (1 sample row)."""
from __future__ import annotations
import json, io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

n_lines = 0
sample = None
keys_union = set()
label_keys = set()
with io.open(ROOT / "files" / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        n_lines += 1
        if n_lines > 5000: break
        try: e = json.loads(ln)
        except: continue
        if sample is None: sample = e
        keys_union.update(e.keys())
        for k in e.keys():
            if "label" in k.lower() or "top" in k.lower() or "rank" in k.lower():
                label_keys.add(k)

print(f"Lines scanned: {n_lines}")
print(f"All keys: {sorted(keys_union)}")
print(f"\nLabel-ish keys: {sorted(label_keys)}")
print(f"\n=== Sample row (first) ===")
if sample:
    for k in sorted(sample.keys()):
        v = sample[k]
        if isinstance(v, (dict, list)):
            v = json.dumps(v)[:80]
        print(f"  {k:30s} = {v}")
