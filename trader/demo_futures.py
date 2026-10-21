"""F0 -- readiness of the demo account for USD-M futures (carry-funding-money-1021-spec.md §10).

readiness(results) turns the observations of a smoke run into GO / NO_GO. It makes no request itself.
Items 1-9 of the spec are required; a key that was not checked counts as not verified. Items 10-11 only warn:
funding credit needs a held position across a funding time, and demo funding is computed from the demo book
(measured 2026-10-21: ETH 0.0000206 on demo vs 0.0000650 live), so demo carry money is not live carry money.
"""
from __future__ import annotations

MIN_FUTURES_BALANCE = 100.0
FUNDING_GAP_WARN_BP = 1.0

REQUIRED = ("account_ok", "futures_permission", "futures_balance_usdt", "one_way_mode", "isolated_ok", "leverage1_ok",
            "commission", "order_test_ok", "lot_match_ok")


def _ok(key, value) -> bool:
    if key == "futures_balance_usdt":
        return isinstance(value, (int, float)) and value >= MIN_FUTURES_BALANCE
    if key == "commission":
        return isinstance(value, dict) and isinstance(value.get("taker"), (int, float))
    return value is True


def readiness(results: dict) -> dict:
    missing = [k for k in REQUIRED if k not in results or not _ok(k, results[k])]
    warnings = []
    if results.get("funding_income_seen") is not True:
        warnings.append("funding_income not verified on the demo (needs a held position across a funding time)")
    gap = results.get("demo_vs_live_funding_gap_bp")
    if gap is None:
        warnings.append("demo_vs_live_funding_gap_bp not measured")
    elif abs(gap) > FUNDING_GAP_WARN_BP:
        warnings.append("demo_vs_live_funding_gap_bp = %.2f bp: demo carry money is not live carry money" % gap)
    return {"status": "GO" if not missing else "NO_GO", "missing": missing, "warnings": warnings}
