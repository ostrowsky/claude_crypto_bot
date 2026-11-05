"""Per-bucket market-order costs for paper books (wide-scan-shadow-w1-1028-spec.md, entry of 2026-11-05).

The books trade at the model's flat 8 bps per side. The mainnet calibration (cost-calibration-mainnet-1105-spec.md,
COST_CAL.json) prices a market order by liquidity bucket and size. Each closed trade gets
delta = (entry + exit - 16 bps) / 100 pp, and the book's daily R loses delta * cost / equity on the exit day.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cost_calibration as CC     # noqa: E402

UNRANKED = 10 ** 6                 # a pair without a rank is priced as the thinnest bucket


def load(path=CC.OUT):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not d.get("ranks"):                       # without ranks every pair would silently be priced as the tail
        raise ValueError("COST_CAL.json has no ranks: run cost_calibration.py ranks")
    return d["tables"]["live"], d["ranks"]


def _cost(table, b, side, size):
    x = table[b][side]
    return float((x.get(str(size)) or x.get(size))["median"])


def trade_delta_pp(sym: str, notional: float, table: dict, ranks: dict) -> float:
    b = CC.bucket(sym, ranks.get(sym, UNRANKED))
    s = CC.nearest_size(notional)
    return CC.trade_delta_pp(_cost(table, b, "buy", s), _cost(table, b, "sell", s))


def adjusted_series(series, exits, day_start_eq: dict, table: dict, ranks: dict) -> list:
    adj = {}
    for e in exits:
        d = str(e["exit_time"])[:10]
        eq = day_start_eq.get(d)
        if not eq:
            continue
        adj[d] = adj.get(d, 0.0) + trade_delta_pp(e["sym"], float(e["cost"]), table, ranks) * float(e["cost"]) / eq
    return [(d, r - adj.get(d, 0.0)) for d, r in series]


def book_exits(ledger_path: Path, book: str) -> list:
    out = []
    p = Path(ledger_path)
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("type") == "exit" and r.get("book") == book and r.get("cost") and r.get("exit_time"):
                out.append(r)
    return out


def day_start_equity(state_path: Path, book: str) -> dict:
    p = Path(state_path)
    if not p.exists():
        return {}
    st = json.loads(p.read_text(encoding="utf-8"))
    raw = ((st.get("books") or {}).get(book) or {}).get("day_start_eq") or {}
    return {str(k)[:10]: float(v) for k, v in raw.items()}
