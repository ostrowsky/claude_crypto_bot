from __future__ import annotations

import asyncio
import json
from pathlib import Path

import data_collector


ROOT = Path(__file__).resolve().parent


async def _amain() -> int:
    btc_ctx = await data_collector._get_btc_context()
    stats = await data_collector._collect_once(btc_ctx)
    payload = {
        "collector": stats,
        "btc_context": btc_ctx,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
