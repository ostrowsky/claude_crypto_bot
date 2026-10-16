"""Trading universe of the demo account -- phase A1 (demo-connection-a-1010-spec.md).

Every pair the demo exchange trades against USDT, minus what cannot or should not
be traded, each exclusion with its reason. The liquidity filter is applied only
once >= 7 days of microstructure record exist (A0); until then pairs that pass
the structural rules carry status "liquidity_unmeasured" rather than a guess.

The signal watchlist (files/watchlist.json, CLAUDE.md §14) is NOT touched: the
universe is a separate file, trader/runtime/trading/universe.json.
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
OUT = TC.RUNTIME / "trading" / "universe.json"
HISTORY = TC.HISTORY
MIN_HISTORY_BARS = 2000
STABLE_BASES = frozenset({"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURI", "AEUR", "USDE", "XUSD",
                          "BFUSD", "PYUSD", "RLUSD", "USD1", "UST", "USTC", "GBP", "TRY", "BRL",
                          "ZAR", "UAH", "PLN", "RON", "ARS", "JPY", "MXN", "COP", "CZK"})
REQUIRED_TYPES = ("MARKET", "LIMIT", "STOP_LOSS_LIMIT")


def is_leveraged(base: str, bases: set) -> bool:
    for suf in ("UP", "DOWN", "BULL", "BEAR"):
        if base.endswith(suf) and len(base) > len(suf) and base[: -len(suf)] in bases:
            return True
    return False


def history_bars(sym: str) -> int:
    """Lines in the longest 15m history file of `sym` (0 if none)."""
    best = 0
    for fp in HISTORY.glob(f"{sym}_15m*.csv"):
        try:
            with io.open(fp, "rb") as fh:
                best = max(best, sum(1 for _ in fh) - 1)
        except Exception:
            pass
    return best


def build(xinfo: dict, liquidity: dict | None = None, thresholds: dict | None = None, hist=history_bars) -> dict:
    """liquidity: {sym: {"quote_vol": .., "spread_bps": .., "impact_bps": .., "divergence_bps": ..}} from A0."""
    syms = xinfo.get("symbols", [])
    bases = {s.get("baseAsset") for s in syms if s.get("quoteAsset") == "USDT"}
    pairs, excluded = [], {}
    for s in syms:
        sym, base = s.get("symbol"), s.get("baseAsset", "")
        if s.get("quoteAsset") != "USDT":
            continue
        why = None
        if s.get("status") != "TRADING":
            why = "status_" + str(s.get("status")).lower()
        elif not s.get("isSpotTradingAllowed", False):
            why = "spot_not_allowed"
        elif not s.get("ocoAllowed", False):
            why = "oco_not_allowed"
        elif not all(t in s.get("orderTypes", []) for t in REQUIRED_TYPES):
            why = "order_types_missing"
        elif base in STABLE_BASES:
            why = "stable_or_fiat"
        elif is_leveraged(base, bases):
            why = "leveraged_token"
        if why:
            excluded[sym] = why
            continue
        rec = {"symbol": sym, "base": base}
        n = hist(sym)
        rec["history_bars_15m"] = n
        if n < MIN_HISTORY_BARS:
            rec["status"] = "needs_history"
        elif not liquidity or not thresholds:
            rec["status"] = "liquidity_unmeasured"
        else:
            L = liquidity.get(sym)
            if L is None:
                rec["status"] = "liquidity_unmeasured"
            else:
                fails = [k for k, (field, op) in {"min_quote_vol": ("quote_vol", "ge"),
                                                  "max_spread_bps": ("spread_bps", "le"),
                                                  "max_impact_bps": ("impact_bps", "le"),
                                                  "max_divergence_bps": ("divergence_bps", "le")}.items()
                         if k in thresholds and L.get(field) is not None
                         and not (L[field] >= thresholds[k] if op == "ge" else L[field] <= thresholds[k])]
                rec["status"] = "illiquid:" + ",".join(fails) if fails else "tradable"
                rec["liquidity"] = L
        pairs.append(rec)
    from collections import Counter
    return {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "demo exchangeInfo", "thresholds": thresholds or None,
            "counts": {"pairs": len(pairs), "excluded": len(excluded),
                       "status": dict(Counter(p["status"].split(":")[0] for p in pairs)),
                       "excluded_by_reason": dict(Counter(excluded.values()))},
            "pairs": pairs, "excluded": excluded}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import audit
    audit.set_process("universe")
    import exchange_client as X
    import microstructure_recorder as M
    c = X.ExchangeClient(key="", secret="")        # exchangeInfo is public
    u = build(M.cached_exchange_info(c))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(u["counts"], ensure_ascii=False))
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
