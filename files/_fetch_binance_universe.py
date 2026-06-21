"""Fetch all actively-trading USDT spot pairs from Binance.

Writes universe to .runtime/binance_universe.json — used by extended
correlation-cluster analysis to find cluster-mates of watchlist coins
that we don't currently scan.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".runtime" / "binance_universe.json"


async def main():
    import aiohttp
    url = "https://api.binance.com/api/v3/exchangeInfo"
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(url) as r:
            r.raise_for_status()
            data = await r.json()

    symbols = data.get("symbols", [])
    usdt_active = []
    for sym in symbols:
        if sym.get("status") != "TRADING": continue
        if sym.get("quoteAsset") != "USDT": continue
        # Skip leveraged / stable / wrapped
        base = sym.get("baseAsset", "")
        if base.endswith("UP") or base.endswith("DOWN"): continue
        if base in ("USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USDT"): continue
        # Skip BULL/BEAR
        if base.endswith("BULL") or base.endswith("BEAR"): continue
        usdt_active.append(sym.get("symbol"))

    usdt_active.sort()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n": len(usdt_active),
        "symbols": usdt_active,
    }, indent=2), encoding="utf-8")
    print(f"[universe] {len(usdt_active)} active USDT pairs → {OUT}")
    print(f"[universe] sample: {usdt_active[:10]} … {usdt_active[-5:]}")


if __name__ == "__main__":
    asyncio.run(main())
