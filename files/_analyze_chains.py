"""Ad-hoc analysis of entry/exit chains and cluster concentration."""
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path


def parse(ts):
    try:
        return datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def main() -> None:
    events = []
    path = Path("bot_events.jsonl")
    if not path.exists():
        path = Path("..") / "bot_events.jsonl"
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ev = r.get("event")
            if ev in ("entry", "exit"):
                t = parse(r.get("ts", ""))
                if t:
                    events.append((t, r.get("sym"), ev, r.get("tf"),
                                   r.get("signal_type") or r.get("mode")))
    events.sort()
    print(f"entry/exit events: {len(events)}  "
          f"span={events[0][0].date()}..{events[-1][0].date()}")

    per_sym = defaultdict(list)
    for t, sym, ev, tf, mode in events:
        per_sym[sym].append((t, ev, tf, mode))

    print("\n=== Symbols with >=2 entries (chains) ===")
    rows = []
    for sym, seq in per_sym.items():
        ne = sum(1 for s in seq if s[1] == "entry")
        nx = sum(1 for s in seq if s[1] == "exit")
        if ne >= 2:
            rows.append((sym, ne, nx, seq))
    for sym, ne, nx, seq in sorted(rows, key=lambda r: -r[1])[:12]:
        print(f"  {sym}: {ne} entries, {nx} exits")
        for t, ev, tf, mode in seq:
            ts = t.strftime("%m-%d %H:%M")
            print(f"      {ts}  {ev:<5} {tf:<4} {mode}")

    print("\n=== Rapid re-entry after exit (< 60 min, same symbol) ===")
    flips = 0
    for sym, seq in per_sym.items():
        last_exit = None
        for t, ev, tf, mode in seq:
            if ev == "exit":
                last_exit = t
            elif ev == "entry" and last_exit is not None:
                dt_min = (t - last_exit).total_seconds() / 60
                if dt_min < 60:
                    print(f"  {sym}: re-entered {dt_min:.1f}m after exit  ({mode} {tf})")
                    flips += 1
                last_exit = None
    print(f"  total rapid flips: {flips}")

    print("\n=== Hold-time distribution ===")
    holds = []
    for sym, seq in per_sym.items():
        open_t = None
        for t, ev, tf, mode in seq:
            if ev == "entry" and open_t is None:
                open_t = t
            elif ev == "exit" and open_t is not None:
                holds.append((sym, (t - open_t).total_seconds() / 60, tf, mode))
                open_t = None
    holds.sort(key=lambda h: h[1])
    if holds:
        mins = [h[1] for h in holds]
        print(f"  n_closed={len(holds)}  min={mins[0]:.0f}m  "
              f"median={mins[len(mins)//2]:.0f}m  max={mins[-1]:.0f}m")
        print(f"  < 30m:  {sum(1 for m in mins if m < 30)}")
        print(f"  30-60m: {sum(1 for m in mins if 30 <= m < 60)}")
        print(f"  1-4h:   {sum(1 for m in mins if 60 <= m < 240)}")
        print(f"  > 4h:   {sum(1 for m in mins if m >= 240)}")
        print("  shortest holds:")
        for sym, m, tf, mode in holds[:10]:
            print(f"    {sym:<12} {m:>5.0f}m  {tf:<4} {mode}")

    # --- Current portfolio cluster concentration ---
    print("\n=== Current portfolio (positions.json) ===")
    pp = Path("positions.json")
    if not pp.exists():
        pp = Path("..") / "positions.json"
    if pp.exists():
        positions = json.loads(pp.read_text(encoding="utf-8"))
        if isinstance(positions, dict):
            # dict-of-sym layout
            syms = list(positions.keys())
            rows = [(s, positions[s]) for s in syms]
        else:
            rows = [(p.get("sym"), p) for p in positions]
        print(f"  open positions: {len(rows)}")
        by_mode = Counter()
        by_tf = Counter()
        for sym, p in rows:
            by_mode[p.get("mode") or p.get("signal_type") or "?"] += 1
            by_tf[p.get("tf") or "?"] += 1
        print(f"  by tf:   {dict(by_tf)}")
        print(f"  by mode: {dict(by_mode)}")
        # rough sector tagging (heuristic by ticker prefix)
        SECTORS = {
            "AI_data":    {"TAO", "RENDER", "FET", "AGIX", "GRT", "INJ", "WLD", "NEAR", "ROSE", "XAI"},
            "L1":         {"ETH", "SOL", "AVAX", "APT", "SUI", "SEI", "NEAR", "ADA", "DOT", "ATOM", "TIA"},
            "L2":         {"ARB", "OP", "MATIC", "POL", "METIS"},
            "meme":       {"DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "BOME"},
            "DeFi":       {"UNI", "AAVE", "LDO", "MKR", "SUSHI", "CRV", "DYDX", "GMX", "COMP", "SNX", "LQTY"},
            "bitcoin":    {"BTC", "ORDI", "SATS", "1000SATS"},
            "gaming":     {"ILV", "AXS", "SAND", "MANA", "GALA", "IMX", "APE"},
        }
        per_sector = defaultdict(list)
        for sym, _ in rows:
            if not sym: continue
            base = sym.replace("USDT", "").replace("USD", "")
            for sec, members in SECTORS.items():
                if base in members:
                    per_sector[sec].append(sym)
        print("  by sector (heuristic):")
        for sec, syms in sorted(per_sector.items(), key=lambda x: -len(x[1])):
            print(f"    {sec:<10} {len(syms):>2}  {syms}")
    else:
        print("  positions.json not found")


if __name__ == "__main__":
    main()
