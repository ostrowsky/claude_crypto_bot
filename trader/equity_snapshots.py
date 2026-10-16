"""Equity snapshots of the demo account -- input of DailyRevenue v1 (daily-revenue-metric-1010-spec.md).

E(t) = sum over all assets (free + locked) x mid price in USDT; USDT = 1; an asset
with no USDT pair is priced through BTC; an asset with no price goes to `unpriced`
(share of equity reported -- above 0.1% the day becomes UNKNOWN in the metric).
Append-only: trader/runtime/trading/equity_snapshots.jsonl. Phase A writes it with
trading off, so the series starts before the first trade.

  ../pyembed/python.exe equity_snapshots.py snapshot
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trader_config as TC  # noqa: E402
OUT = TC.RUNTIME / "trading" / "equity_snapshots.jsonl"
STABLE_ONE = frozenset({"USDT"})


def value(balances: list, mids: dict) -> dict:
    """balances: [{"asset", "free", "locked"}]; mids: {"BTCUSDT": mid, ...}."""
    btc = mids.get("BTCUSDT")
    eq, unpriced, assets = 0.0, {}, {}
    for b in balances:
        a = b["asset"]
        q = float(b.get("free") or 0) + float(b.get("locked") or 0)
        if q <= 0:
            continue
        if a in STABLE_ONE:
            px = 1.0
        elif mids.get(a + "USDT"):
            px = mids[a + "USDT"]
        elif btc and mids.get(a + "BTC"):
            px = mids[a + "BTC"] * btc
        else:
            px = None
        if px is None:
            unpriced[a] = q
            continue
        assets[a] = {"qty": q, "px": px, "usdt": q * px}
        eq += q * px
    return {"equity_usdt": round(eq, 6), "assets": len(assets), "unpriced": unpriced,
            "holdings": {a: {"qty": v["qty"], "px": v["px"]} for a, v in sorted(assets.items())},
            "unpriced_count": len(unpriced), "unpriced_share": None if not unpriced else "unknown (no price)",
            "positions": {a: round(v["usdt"], 4) for a, v in sorted(assets.items(), key=lambda kv: -kv[1]["usdt"])}}


def snapshot(client=None) -> dict:
    import exchange_client as X
    client = client or X.ExchangeClient()
    acct = client.account()
    mids = {}
    for r in client.book_tickers():
        bid, ask = float(r.get("bidPrice") or 0), float(r.get("askPrice") or 0)
        if bid > 0 and ask > 0:
            mids[r["symbol"]] = (bid + ask) / 2
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "demo_account",
           **value(acct.get("balances", []), mids)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import audit
    audit.set_process("equity")
    import faulthandler
    faulthandler.dump_traceback_later(60, exit=True)   # watchdog, see microstructure_recorder
    r = snapshot()
    print(json.dumps({k: r[k] for k in ("ts", "equity_usdt", "assets", "unpriced_count")}))
