"""
Diagnose today's block reasons — why is portfolio 1/10 despite active market?

Stream bot_events.jsonl for today's 'blocked' and 'entry' events, group by
reason_code and by symbol. For top-blocked symbols, fetch current price and
report what would have happened (would those entries be winning or losing?).
"""
from __future__ import annotations
import json, io, sys, urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent
TODAY = "2026-04-19"

blocked = []
entries = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        ts = e.get("ts", "")
        if not ts.startswith(TODAY): continue
        et = e.get("event")
        if et == "blocked": blocked.append(e)
        elif et == "entry": entries.append(e)

print(f"=== TODAY ({TODAY}) — activity ===")
print(f"Entries fired: {len(entries)}  |  Blocked events: {len(blocked)}")

print(f"\n=== Entries today ===")
for e in entries:
    print(f"  {e.get('ts')}  {e.get('sym'):12s}  {e.get('tf','?'):>3s}  {e.get('mode','?'):>14s}  @ {e.get('price')}")

print(f"\n=== Blocks by reason_code (top 15) ===")
by_reason = Counter(b.get("reason_code", b.get("signal_type", "?")) for b in blocked)
for code, n in by_reason.most_common(15):
    print(f"  {n:>4d}  {code}")

print(f"\n=== Blocks by (reason_code, tf, mode) (top 20) ===")
by_trio = Counter((b.get("reason_code") or b.get("signal_type") or "?",
                   b.get("tf") or "?", b.get("mode") or "?") for b in blocked)
for (code, tf, mode), n in by_trio.most_common(20):
    print(f"  {n:>4d}  {code:>25s}  {tf:>3s}  {mode}")

print(f"\n=== Symbols most-blocked today (top 20) ===")
by_sym = Counter(b.get("sym", "?") for b in blocked)
for sym, n in by_sym.most_common(20):
    # Show dominant reasons for this sym
    reasons = Counter(b.get("reason_code","?") for b in blocked if b.get("sym")==sym)
    top3 = ", ".join(f"{r}×{c}" for r, c in reasons.most_common(3))
    print(f"  {n:>4d}  {sym:12s}  reasons: {top3}")

# ── Fetch today's movers ────────────────────────────────────────
print("\n=== What the market did today (24h % top movers on Binance futures) ===")
def fetch_24h():
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)
    except Exception as e:
        print(f"  err: {e}"); return []

tickers = fetch_24h()
usdt = [t for t in tickers if t.get("symbol","").endswith("USDT")]
# Sort by priceChangePercent
usdt.sort(key=lambda t: float(t.get("priceChangePercent",0) or 0), reverse=True)
print("  TOP 20 gainers (24h):")
top20_syms = set()
for t in usdt[:20]:
    sym = t["symbol"]
    top20_syms.add(sym)
    pct = float(t["priceChangePercent"])
    vol = float(t.get("quoteVolume", 0))
    was_blocked = by_sym.get(sym, 0)
    was_entered = any(e.get("sym")==sym for e in entries)
    marker = "✓ ENTERED" if was_entered else (f"✗ BLOCKED ×{was_blocked}" if was_blocked else "— not scored")
    print(f"    {sym:14s}  {pct:+6.2f}%  vol=${vol/1e6:>5.1f}M  {marker}")

# ── Critical: of today's top-20 gainers, how many did bot block? ────
print(f"\n=== CRITICAL: bot vs top-20-gainers match ===")
entered_syms = {e.get("sym") for e in entries}
blocked_top20 = top20_syms & set(by_sym.keys())
entered_top20 = top20_syms & entered_syms
print(f"  Top-20 gainers that bot ENTERED:  {len(entered_top20)}  {sorted(entered_top20)}")
print(f"  Top-20 gainers that bot BLOCKED:  {len(blocked_top20)}  (see reasons below)")
print(f"  Top-20 gainers bot NEVER SAW:     {len(top20_syms) - len(blocked_top20) - len(entered_top20)}")

print(f"\n=== For each top-20 gainer that was blocked, dominant reason ===")
for sym in sorted(blocked_top20):
    sym_blocks = [b for b in blocked if b.get("sym")==sym]
    reasons = Counter(b.get("reason_code","?") for b in sym_blocks)
    first_ts = sym_blocks[0].get("ts","?") if sym_blocks else "?"
    last_ts  = sym_blocks[-1].get("ts","?") if sym_blocks else "?"
    top_r = reasons.most_common(3)
    rs = ", ".join(f"{r}×{c}" for r,c in top_r)
    print(f"  {sym:14s}  {len(sym_blocks)} blocks  {first_ts[:16]}..{last_ts[11:16]}  {rs}")
    # Show one sample block reason text
    sample = sym_blocks[-1]
    reason_text = sample.get("reason", "")
    if reason_text:
        print(f"       last reason: {reason_text[:160]}")
