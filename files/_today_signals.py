"""Quality audit of today's bot signals."""
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path


def parse(ts):
    try:
        return datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def main():
    today = datetime.now(timezone.utc).date()
    # Bot writes to files/bot_events.jsonl (cwd=files/)
    path = Path("files/bot_events.jsonl")
    if not path.exists():
        path = Path("bot_events.jsonl")

    entries = []   # list of dicts
    exits = []
    blocks = Counter()
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = parse(r.get("ts", ""))
            if not t or t.date() != today:
                continue
            ev = r.get("event")
            if ev == "entry":
                entries.append(r)
            elif ev == "exit":
                exits.append(r)
            elif ev == "blocked":
                blocks[r.get("signal_type", "?")] += 1

    print(f"=== Today ({today}) ===")
    print(f"Entries: {len(entries)}  Exits: {len(exits)}")
    print(f"Blocks: {sum(blocks.values())}  top={blocks.most_common(8)}")
    print()

    # Match entries to exits per symbol (FIFO)
    per_sym_open = defaultdict(list)
    for e in entries:
        per_sym_open[e.get("sym")].append(e)
    closed = []
    open_syms = defaultdict(list)
    # build chronological trades
    for e in entries:
        open_syms[e.get("sym")].append(e)
    for x in exits:
        sym = x.get("sym")
        if open_syms[sym]:
            entry = open_syms[sym].pop(0)
            closed.append((entry, x))

    still_open = []
    for sym, lst in open_syms.items():
        for e in lst:
            still_open.append(e)

    # Enrichment: load positions.json for currently-open live data
    pp = Path("positions.json")
    if not pp.exists():
        pp = Path("..") / "positions.json"
    live = {}
    if pp.exists():
        data = json.loads(pp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            live = data

    def fmt_entry(e):
        sym = e.get("sym"); tf = e.get("tf"); mode = e.get("signal_type", "?")
        t = parse(e.get("ts", ""))
        return (t, sym, tf, mode,
                e.get("price"), e.get("rsi"), e.get("adx"),
                e.get("slope_pct") or e.get("slope"), e.get("vol_x"))

    print(f"--- Closed trades today: {len(closed)} ---")
    print(f"{'time':<6} {'sym':<12} {'tf':<4} {'mode':<16} "
          f"{'rsi':>5} {'adx':>5} {'vol':>5} {'pnl%':>7} {'bars':>4} {'reason'}")
    pnls = []
    wins = 0
    for entry, x in sorted(closed, key=lambda p: parse(p[0].get("ts", "")) or datetime.min.replace(tzinfo=timezone.utc)):
        t = parse(entry.get("ts", "")).strftime("%H:%M")
        sym = entry.get("sym", "")
        tf = entry.get("tf", "")
        mode = entry.get("signal_type", "")
        rsi = entry.get("rsi") or 0
        adx = entry.get("adx") or 0
        vol = entry.get("vol_x") or 0
        pnl = x.get("pnl_pct") or x.get("change_pct") or x.get("ret_pct")
        if pnl is None:
            # compute from prices
            p0 = entry.get("price"); p1 = x.get("price")
            if p0 and p1:
                pnl = (p1 - p0) / p0 * 100
        bars = x.get("bars") or x.get("bars_in_pos") or "-"
        reason = (x.get("reason") or "")[:40]
        pnl_s = f"{pnl:+.2f}" if pnl is not None else "  ?  "
        if pnl is not None:
            pnls.append(pnl)
            if pnl > 0:
                wins += 1
        print(f"{t:<6} {sym:<12} {tf:<4} {mode:<16} {rsi:>5.1f} {adx:>5.1f} "
              f"{vol:>5.2f} {pnl_s:>7} {str(bars):>4} {reason}")

    if pnls:
        avg = sum(pnls) / len(pnls)
        wr = 100 * wins / len(pnls)
        print(f"\nSummary closed: n={len(pnls)} avg_pnl={avg:+.3f}%  "
              f"win%={wr:.1f}  worst={min(pnls):+.2f}%  best={max(pnls):+.2f}%")

    print(f"\n--- Still open from today's entries: {len(still_open)} ---")
    print(f"{'time':<6} {'sym':<12} {'tf':<4} {'mode':<16} "
          f"{'rsi':>5} {'adx':>5} {'vol':>5} {'live_pnl%':>9}")
    for e in still_open:
        t = parse(e.get("ts", "")).strftime("%H:%M")
        sym = e.get("sym", "")
        tf = e.get("tf", "")
        mode = e.get("signal_type", "")
        rsi = e.get("rsi") or 0
        adx = e.get("adx") or 0
        vol = e.get("vol_x") or 0
        p0 = e.get("price")
        live_pnl = "?"
        if sym in live and p0:
            entry_price = live[sym].get("entry_price", p0)
            last = live[sym].get("last_price") or live[sym].get("trail_stop") or entry_price
            # try updated field
            for k in ("current_price", "last_price", "mark_price"):
                if live[sym].get(k):
                    last = live[sym][k]
                    break
            try:
                live_pnl = f"{(float(last)-float(entry_price))/float(entry_price)*100:+.2f}"
            except Exception:
                pass
        print(f"{t:<6} {sym:<12} {tf:<4} {mode:<16} {rsi:>5.1f} {adx:>5.1f} "
              f"{vol:>5.2f} {live_pnl:>9}")

    # Quality flags: ADX<22 or vol_x<1.5 on impulse_speed, catch-up drift
    print("\n--- Quality red flags on today's entries ---")
    red = []
    for e in entries:
        rsi = e.get("rsi") or 0
        adx = e.get("adx") or 0
        vol = e.get("vol_x") or 0
        mode = e.get("signal_type", "")
        sym = e.get("sym", "")
        tf = e.get("tf", "")
        flags = []
        if mode == "impulse_speed" and adx < 22:
            flags.append(f"ADX={adx:.1f} weak impulse")
        if mode == "impulse_speed" and vol < 1.8:
            flags.append(f"vol_x={vol:.2f} weak vol for impulse")
        if mode == "alignment" and vol < 1.3:
            flags.append(f"vol_x={vol:.2f} weak vol for alignment")
        if mode in ("trend", "strong_trend") and adx < 20:
            flags.append(f"ADX={adx:.1f} weak trend")
        if rsi and rsi < 50 and mode in ("impulse_speed", "trend", "alignment"):
            flags.append(f"RSI={rsi:.1f} no momentum")
        reason = e.get("reason", "") or ""
        if "catch-up" in reason.lower() or "drift" in reason.lower():
            flags.append("catch-up fill")
        if flags:
            red.append((sym, tf, mode, flags))
    if red:
        for sym, tf, mode, flags in red:
            print(f"  {sym:<12} {tf:<4} {mode:<16} {' | '.join(flags)}")
        print(f"  total flagged: {len(red)} / {len(entries)} "
              f"({100*len(red)/max(len(entries),1):.0f}%)")
    else:
        print("  no flags.")


if __name__ == "__main__":
    main()
