import json
from pathlib import Path

# Read offline_rl log tail
p = Path("offline_rl_log.jsonl")
if p.exists():
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    print(f"offline_rl_log: {len(lines)} entries")
    for ln in lines[-3:]:
        try:
            r = json.loads(ln)
            print(" keys:", list(r.keys()))
            print(" ", json.dumps(r, indent=2)[:2000])
        except Exception as e:
            print("  parse err:", e)

# Read optimizer state
op = Path("rl_optimizer_state.json")
if op.exists():
    d = json.loads(op.read_text())
    print(f"\nrl_optimizer_state: generation={d.get('generation')}, "
          f"best_reward={d.get('best_reward')}, sigma={d.get('sigma')}")
    h = d.get("history") or []
    print(f"  history entries: {len(h)}")
    for entry in h[-5:]:
        print(" ", entry)

# Memory stats
from rl_memory import load_experiences, memory_stats
exps = load_experiences()
print(f"\nrl_memory experiences: {len(exps)}")
if exps:
    m = memory_stats(exps)
    print(f"  wr={m.get('win_rate', 0)*100:.1f}% avg_pnl={m.get('avg_pnl', 0):+.2f}%")
