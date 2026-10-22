"""FC_PASSIVE shadow book on live funding (fc-passive-next-1022-spec.md §A).

A paper book: spot long + USD-M perp short on BTCUSDT and ETHUSDT, L = 1 isolated, costs at level 1, 10 000 USDT,
from start day S (the first full UTC day after the first run; fixed thereafter). Every step recomputes the SAME
engine as the accepted 2026 test -- carry.simulate(experiment_fc.passive_config(), fees(1)) on [S, today) -- so the
book cannot drift from what was tested, a repeated step changes nothing and a step after a pause catches up.

Public data only (no key, no order), own append-only store in runtime/fc_shadow/data/. The research stores
(runtime/history, perp_history, funding) are not written.

  fc_shadow.py step     -- fetch new bars and funding, recompute, write state.json and ledger.jsonl
  fc_shadow.py readout  -- READOUT_FC.json: COLLECTING / KEEP / REJECT / INCONCLUSIVE / ANOMALY (§A.5)
Rollback: trader_config.FC_SHADOW_ENABLED = False (the step then does nothing).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trader_config as TC  # noqa: E402

FC_SHADOW_DIR = TC.RUNTIME / "fc_shadow"
DATA_DIR = FC_SHADOW_DIR / "data"
STATE = FC_SHADOW_DIR / "state.json"
LEDGER = FC_SHADOW_DIR / "ledger.jsonl"
READOUT = FC_SHADOW_DIR / "READOUT_FC.json"
SPEC = "fc-passive-next-1022-spec.md"

PAIRS = ("BTCUSDT", "ETHUSDT")
START_EQUITY = 10_000.0
STORE_LEAD_DAYS = 35            # the store starts 35 days before S (liquidity filter needs 20 days)
MIN_DAYS = 90                   # readout FC-SHADOW: no verdict before S + 90 full days
BAR_GAP_MAX_MS = 60 * 60 * 1000          # a hole of more than 1 h in a leg's 15m bars
FUNDING_GAP_MAX_MS = 12 * 60 * 60 * 1000  # 8 h interval + 4 h slack
PRICE_RATIO_MAX = 0.05          # perp / spot daily close outside +-5%
DAY_MS = 86_400_000
BAR_MS = 15 * 60 * 1000
LIVE_SPOT = "https://api.binance.com"


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def start_day(now: datetime) -> str:
    """First FULL UTC day at or after now: exactly 00:00:00 counts as that day."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight if now == midnight else midnight + timedelta(days=1)).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ store (public data, append-only)
def store_path(sym: str, leg: str) -> Path:
    return DATA_DIR / f"{sym}_{leg}.npz"


def _fetch_perp(client, sym: str, start_ms: int, end_ms: int, now_ms: int) -> None:
    import funding_store as FS
    import history_store as HS
    path = store_path(sym, "perp15m")
    have = HS.load(path)
    cursor = int(start_ms) if have is None or not len(have["t"]) else max(int(start_ms), int(have["t"][-1]) + BAR_MS)
    pending = []
    while cursor < end_ms:
        rows = FS._with_retry(lambda: client.futures_klines_from(sym, "15m", cursor, 1000))
        if not rows:
            break
        pending.extend(r for r in rows if int(r[0]) < end_ms and int(r[6]) < now_ms)
        if len(rows) < 1000:
            break
        cursor = int(rows[-1][0]) + BAR_MS
    if pending:
        have, _ = HS.merge(have, HS._rows_to_arr(pending))
        HS.save(path, have)


def _fetch_funding(client, sym: str, start_ms: int) -> None:
    import funding_store as FS
    path = store_path(sym, "funding")
    have = FS.load_funding(path)
    cursor = int(start_ms) if have is None or not len(have["t"]) else max(int(start_ms), int(have["t"][-1]) + 1)
    added = 0
    while True:
        rows = FS._with_retry(lambda: client.funding_rates(sym, cursor, 1000))
        if not rows:
            break
        have, a = FS.merge(have, FS.parse_funding(rows))
        added += a
        if len(rows) < 1000:
            break
        cursor = int(rows[-1]["fundingTime"]) + 1
    if added:
        FS.save_funding(path, have)


def load_data() -> dict:
    import funding_store as FS
    import history_store as HS
    out = {}
    for sym in PAIRS:
        out[sym] = {"spot": HS.load(store_path(sym, "spot15m")), "perp": HS.load(store_path(sym, "perp15m")),
                    "funding": FS.load_funding(store_path(sym, "funding"))}
    return out


def fetch_live(start_ms: int, now: datetime, client=None) -> dict:
    """Append new closed 15m bars (spot, perp) and funding events since start_ms; return the whole store.
    Raises on a persistent network error, so a half-updated store is never judged (the step aborts)."""
    import exchange_client as X
    import history_store as HS
    pub = client or X.ExchangeClient(key="", secret="")
    now_ms = int(now.timestamp() * 1000)
    for sym in PAIRS:
        HS.fetch_symbol(pub, sym, start_ms, now_ms, store_path(sym, "spot15m"), now_ms=now_ms)
        _fetch_perp(pub, sym, start_ms, now_ms, now_ms)
        _fetch_funding(pub, sym, start_ms)
    return load_data()


# ------------------------------------------------------------------ market and checks
def market_from(data: dict):
    import carry as C
    import xsmom as XS
    spot = XS.Panel.from_daily({s: XS.daily_from_15m(d["spot"]) for s, d in data.items()})
    perp = C.PerpPanel.from_daily({s: C.daily_perp(d["perp"]) for s, d in data.items()}, mult={})
    funding = {s: {"t": np.asarray(d["funding"]["t"], dtype=np.int64), "rate": np.asarray(d["funding"]["rate"], dtype=float)}
               for s, d in data.items()}
    return C.Market(spot=spot, perp=perp, funding=funding)


def _missing(data: dict) -> list:
    out = []
    for sym in PAIRS:
        d = (data or {}).get(sym) or {}
        for leg in ("spot", "perp", "funding"):
            a = d.get(leg)
            if a is None or not len(a["t"]):
                out.append({"kind": "no_data", "sym": sym, "leg": leg})
    return out


def _daily_close(a: dict, lo: int, hi: int) -> dict:
    t = np.asarray(a["t"], dtype=np.int64)
    m = (t >= lo) & (t < hi)
    t, c = t[m], np.asarray(a["c"], dtype=float)[m]
    if not len(t):
        return {}
    o = np.argsort(t, kind="stable")
    t, c = t[o], c[o]
    day = t // DAY_MS
    ends = np.r_[np.flatnonzero(day[1:] != day[:-1]), len(t) - 1]
    return {int(day[i]): float(c[i]) for i in ends}


def data_checks(data: dict, day_from: str, day_to: str) -> list:
    """§A.4 inside the window [day_from, day_to): bar holes > 1 h, funding pauses > 12 h, perp/spot close off by > 5%."""
    lo, hi = _ms(day_from), _ms(day_to)
    if hi <= lo:
        return []
    out = _missing(data)
    for sym in PAIRS:
        d = data.get(sym) or {}
        for leg in ("spot", "perp"):
            a = d.get(leg)
            if a is None or not len(a["t"]):
                continue
            t = np.asarray(a["t"], dtype=np.int64)
            t = np.sort(t[(t >= lo) & (t < hi)])
            pts = np.r_[lo - BAR_MS, t, hi]                      # virtual bars just outside the window
            hole = np.diff(pts) - BAR_MS
            for i in np.flatnonzero(hole > BAR_GAP_MAX_MS):
                out.append({"kind": "bar_gap", "sym": sym, "leg": leg, "from_ms": int(pts[i]) + BAR_MS,
                            "hours": round(float(hole[i]) / 3_600_000, 2)})
        f = d.get("funding")
        if f is not None and len(f["t"]):
            t = np.asarray(f["t"], dtype=np.int64)
            pts = np.r_[lo, np.sort(t[(t >= lo) & (t <= hi)]), hi]
            gap = np.diff(pts)
            for i in np.flatnonzero(gap > FUNDING_GAP_MAX_MS):
                out.append({"kind": "funding_gap", "sym": sym, "from_ms": int(pts[i]), "hours": round(float(gap[i]) / 3_600_000, 2)})
        if d.get("spot") is not None and d.get("perp") is not None:
            sc, pc = _daily_close(d["spot"], lo, hi), _daily_close(d["perp"], lo, hi)
            for day in sorted(set(sc) & set(pc)):
                ratio = pc[day] / sc[day] if sc[day] else float("inf")
                if abs(ratio - 1) > PRICE_RATIO_MAX:
                    out.append({"kind": "price_ratio", "sym": sym, "day": _day(day * DAY_MS), "ratio": round(ratio, 5)})
    return out


def book_checks(book: dict) -> list:
    """A liquidation is an anomaly; a margin stop is normal mechanics (it goes to the ledger as a trade)."""
    n = int(book.get("liquidations") or 0)
    return [{"kind": "liquidation", "n": n}] if n > 0 else []


def _key(rec: dict) -> str:
    return "|".join(str(rec.get(k)) for k in ("type", "kind", "sym", "leg", "day", "from_ms", "entry_day", "exit_day",
                                              "reason", "n"))


# ------------------------------------------------------------------ step
def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_json(path: Path, doc) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def _append_ledger(path: Path, recs: list) -> int:
    path = Path(path)
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["key"])
            except Exception:
                continue
    new = []
    for r in recs:
        k = _key(r)
        if k not in seen:
            seen.add(k)
            new.append(dict(r, key=k))
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for r in new:
                fh.write(json.dumps(r, default=str) + "\n")
    return len(new)


def _complete(data: dict, day_to: str) -> bool:
    """The store holds the last bar before day_to for every leg (nothing left to fetch for this window)."""
    end = _ms(day_to) - BAR_MS
    for sym in PAIRS:
        for leg in ("spot", "perp"):
            a = ((data or {}).get(sym) or {}).get(leg)
            if a is None or not len(a["t"]) or int(np.max(a["t"])) < end:
                return False
    return True


def step(now: datetime | None = None, fetch=None, state_path=STATE, ledger_path=LEDGER, enabled=None) -> dict:
    enabled = TC.FC_SHADOW_ENABLED if enabled is None else enabled
    if not enabled:
        return {"skipped": True, "reason": "FC_SHADOW_ENABLED is False"}
    now = now or datetime.now(timezone.utc)
    state = _read_json(state_path) or {}
    if not state.get("start_day"):
        state = {"spec": SPEC, "book": "FC_PASSIVE", "start_equity": START_EQUITY, "start_day": start_day(now),
                 "created_utc": now.isoformat(timespec="seconds"), "series": [], "marks": [], "n_trades": 0,
                 "liquidations": 0, "margin_stops": 0, "funding_income": 0.0, "anomalies": []}
    S = state["start_day"]
    today = now.strftime("%Y-%m-%d")
    state["last_step_utc"] = now.isoformat(timespec="seconds")
    if today <= S or (state.get("computed_to") == today and state.get("complete")):
        if today <= S:
            state.update(computed_to=today, complete=True)
        _write_json(state_path, state)
        return {"skipped": False, "days": len(state["series"]), "start_day": S, "unchanged": True}

    start_ms = _ms(S) - STORE_LEAD_DAYS * DAY_MS
    data = (fetch or fetch_live)(start_ms, now)
    found = data_checks(data, S, today)
    new_records = []
    if not _missing(data):
        import carry as C
        import experiment_fc as FC
        r = C.simulate(market_from(data), FC.passive_config(), C.fees(1), S, today, start_equity=START_EQUITY)
        trades = [t for t in r["trades"] if t["reason"] != "flush"]      # flush = window edge, not a trade
        eq, marks = START_EQUITY, [[S, START_EQUITY]]
        for d, v in r["series"]:
            eq *= 1 + v / 100
            marks.append([(datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"), eq])
        state.update(series=[list(x) for x in r["series"]], marks=marks, n_trades=len(trades),
                     liquidations=int(r["liquidations"]), margin_stops=int(r["margin_stops"]),
                     funding_income=float(r["funding_income"]), exposure_share=float(r["exposure_share"]))
        found += book_checks(r)
        new_records += [{"type": "trade", "sym": t["sym"], "entry_day": t["entry_day"], "exit_day": t["exit_day"],
                         "reason": t["reason"], "N": t["N"], "funding": t["funding"], "pnl": t["pnl"],
                         "ret_pct": t["ret_pct"]} for t in trades]
    # anomalies are sticky: once seen, the readout stays ANOMALY until the operator looks
    known = {_key(dict(a, type="anomaly")) for a in state.get("anomalies", [])}
    for a in found:
        if _key(dict(a, type="anomaly")) not in known:
            known.add(_key(dict(a, type="anomaly")))
            state["anomalies"].append(a)
    new_records += [dict(a, type="anomaly") for a in found]
    state.update(computed_to=today, complete=_complete(data, today) and not _missing(data))
    _write_json(state_path, state)
    added = _append_ledger(ledger_path, new_records)
    return {"skipped": False, "days": len(state["series"]), "start_day": S, "computed_to": today,
            "anomalies": len(state["anomalies"]), "ledger_added": added}


# ------------------------------------------------------------------ readout FC-SHADOW (§A.5, fixed before start)
def readout(state: dict) -> dict:
    import daily_revenue as DR
    series = [tuple(x) for x in state.get("series", [])]
    n = len(series)
    agg = DR.aggregate(series) if n else {"n_days": 0}
    liq = int(state.get("liquidations") or 0)
    anomalies = state.get("anomalies") or []
    S = state.get("start_day")
    due = (datetime.strptime(S, "%Y-%m-%d") + timedelta(days=MIN_DAYS)).strftime("%Y-%m-%d") if S else None
    if anomalies:
        status = "ANOMALY"
    elif n < MIN_DAYS:
        status = "COLLECTING"
    elif liq > 0 or agg["hi95"] < 0:
        status = "REJECT"
    elif agg["lo95"] > 0:
        status = "KEEP"
    else:
        status = "INCONCLUSIVE"
    return {"readout": "FC-SHADOW", "spec": SPEC, "status": status, "start_day": S, "due": due, "n_days": n,
            "min_days": MIN_DAYS, "agg": agg, "liquidations": liq, "margin_stops": state.get("margin_stops"),
            "funding_income": state.get("funding_income"), "anomalies": anomalies,
            "criteria": "KEEP: >= 90 days, lo95 of mean daily R > 0, 0 liquidations; REJECT: hi95 < 0 or any liquidation; "
                        "INCONCLUSIVE: extend the window, criteria never loosened; ANOMALY: any §A.4 check"}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "step":
        import audit as A
        A.set_process("fc_shadow")
        print("FC shadow step:", json.dumps(step(), default=str), flush=True)
        return 0
    if cmd == "readout":
        state = _read_json(STATE)
        if not state:
            print("FC shadow: no state yet")
            return 0
        r = readout(state)
        r["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json(READOUT, r)
        print("FC-SHADOW:", r["status"], "days", r["n_days"], "mean", r["agg"].get("mean_pct"), "lo95", r["agg"].get("lo95"),
              "due", r["due"])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
