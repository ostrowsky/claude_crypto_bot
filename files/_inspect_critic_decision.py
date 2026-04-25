"""Peek at critic_dataset 'decision' field — does it store ranker outputs?"""
import json, io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

seen_dec_keys = set()
samples_take = []
samples_take_1h = []
with io.open(FILES / "critic_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        dec = e.get("decision") or {}
        seen_dec_keys |= set(dec.keys())
        if dec.get("action") == "take" and len(samples_take) < 3:
            samples_take.append(e)
        if dec.get("action") == "take" and e.get("tf") == "1h" and e.get("signal_type") in ("impulse", "impulse_speed") and len(samples_take_1h) < 3:
            samples_take_1h.append(e)

print(f"decision keys seen: {sorted(seen_dec_keys)}\n")
print("=== sample 'take' decisions ===")
for s in samples_take:
    print(json.dumps(s.get("decision"), indent=2, ensure_ascii=False))
    print("---")
print("\n=== 1h impulse/impulse_speed 'take' samples ===")
for s in samples_take_1h:
    print(f"sym={s.get('sym')} sig={s.get('signal_type')} tf={s.get('tf')}")
    print(f"decision: {json.dumps(s.get('decision'), indent=2, ensure_ascii=False)}")
    print(f"labels: {json.dumps(s.get('labels'), indent=2, ensure_ascii=False)}")
    print("---")
