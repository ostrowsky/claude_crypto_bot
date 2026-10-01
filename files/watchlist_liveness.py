"""Daily liveness check of the watchlist: is every pair still trading?

WHY (2026-10-01, watchlist-dead-1001-spec.md)

Nine watchlist pairs had been delisted from Binance spot for 45 to 533 days --
BAKEUSDT's last candle was 2025-09-17 -- and the bot kept polling them, using
slots of the 45-per-cycle rotation and evaluating year-old bars, while nothing
reported it. The 2026-08-19 cleanup removed three such pairs by hand; this
makes the next one visible the day it happens.

A pair is DEAD when the exchange no longer lists it as TRADING, or its newest
15m candle is older than DEAD_AFTER_HOURS. The check never edits the watchlist
(it is the operator's list, CLAUDE.md §14): it writes
.runtime/watchlist_liveness.json and the morning report shows a step. A network
failure is "unknown", never "dead" (fail open: no false alarm from an outage).
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / ".runtime" / "watchlist_liveness.json"
API = "https://api.binance.com"
DEAD_AFTER_HOURS = 24


def classify(status: str | None, last_open: datetime | None, now: datetime) -> str:
    """'live' | 'dead' | 'unknown' for one pair."""
    if status is None:
        return "unknown"
    if status != "TRADING":
        return "dead"
    if last_open is None:
        return "unknown"
    return "dead" if (now - last_open).total_seconds() / 3600 > DEAD_AFTER_HOURS else "live"


def _get(url: str, timeout: int = 20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check(symbols: list, now: datetime | None = None, get=_get) -> dict:
    now = now or datetime.now(timezone.utc)
    try:
        info = get(f"{API}/api/v3/exchangeInfo", timeout=30)
        status = {s["symbol"]: s["status"] for s in info.get("symbols", [])}
    except Exception as e:
        return {"checked_at": now.isoformat(timespec="seconds"), "available": False, "error": repr(e)[:200]}
    rows = []
    for sym in symbols:
        st = status.get(sym, "ABSENT")
        last = None
        try:
            q = urllib.parse.urlencode({"symbol": sym, "interval": "15m", "limit": 1})
            k = get(f"{API}/api/v3/klines?{q}")
            if k:
                last = datetime.fromtimestamp(int(k[-1][0]) / 1000, timezone.utc)
        except Exception:
            last = None
        verdict = classify(st if st != "ABSENT" else "ABSENT", last, now)
        rows.append({"symbol": sym, "status": st, "last_candle": last.isoformat() if last else None,
                     "verdict": verdict})
    dead = [r for r in rows if r["verdict"] == "dead"]
    return {"checked_at": now.isoformat(timespec="seconds"), "available": True, "n": len(rows),
            "dead": dead, "unknown": [r["symbol"] for r in rows if r["verdict"] == "unknown"]}


def load_latest(path: Path | None = None) -> dict:
    try:
        return json.loads((path or OUT).read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import config
    res = check(config.load_watchlist())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if not res.get("available"):
        print(f"[liveness] exchange unavailable: {res.get('error')}")
        return 0
    print(f"[liveness] {res['n']} pairs, dead {len(res['dead'])}: "
          + ", ".join(f"{d['symbol']} ({d['status']}, last {str(d['last_candle'])[:10]})" for d in res["dead"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
