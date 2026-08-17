"""Immutable later-EOD labels, built from exchange klines only.

`label_top20` is computed from the same rolling-24h snapshot that produces the
features, so a coin already up 14% at the snapshot is labelled a winner
*because* it is up 14%. That is how `top_gainer_model` reached AUC 0.99 and the
bandit reached "recall@20 = 100%", and it is why the North Star is marked
provisional (TH-03).

This module builds ground truth the other way round: one record per (symbol, UTC
day) from exchange OHLCV, never from anything the bot produced. A label derived
from the same snapshot as the features cannot be ground truth for those
features.

Two design points carry most of the value:

* **Strict UTC days.** `Europe/Budapest` gives 23- and 25-hour days at DST
  boundaries and disagrees with the exchange day — a silent denominator defect
  in exactly the metric built to be trustworthy.
* **A fixed early deadline.** `early_deadline_ts` is the first crossing of
  +2.5% from the open, not "half the realised move". The latter is computable
  only with hindsight and, for any move above +5%, lands *before* the +5% anchor
  it was supposed to follow — the v1 MoveEvent bug.

Immutable: a written record is never overwritten. Rebuilding must reproduce it,
or the store raises.

    pyembed\\python.exe files\\label_store.py --build
    pyembed\\python.exe files\\label_store.py --status

Spec: docs/specs/features/immutable-label-store-spec.md
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# UNCHANGED on purpose. `_identity` hashes builder_version, so bumping this
# would make every rebuild of the 19 502 written records raise
# ImmutableLabelError. The daily tier gets its own version instead.
BUILDER_VERSION = "label-store-v1"
DAILY_BUILDER_VERSION = "label-store-daily-v1"
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
MOVE_THRESHOLD_PCT = 5.0
EARLY_THRESHOLD_PCT = 2.5
MIN_BARS_COMPLETE = 20

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / ".runtime" / "labels"
DEFAULT_SOURCE = Path(__file__).resolve().parent / "_hourly_ohlcv_long.json"


class ImmutableLabelError(RuntimeError):
    """A written label was contradicted. Never resolved by overwriting."""


def _utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def build_day_record(symbol: str, day_start_ms: int, bars: list[list],
                     *, provenance: dict) -> dict[str, Any]:
    """One (symbol, UTC day) label from that day's bars.

    `bars` are [open_ts, open, high, low, close, volume], already restricted to
    the day and sorted. Partial days are marked incomplete and cannot qualify:
    absence of data is not evidence of a quiet day (TH-05).
    """
    if not bars:
        raise ValueError("cannot label a day with no bars")
    bars = sorted(bars, key=lambda b: b[0])
    day_open = float(bars[0][1])
    if day_open <= 0:
        raise ValueError("non-positive open price")

    high = max(float(b[2]) for b in bars)
    low = min(float(b[3]) for b in bars)
    close = float(bars[-1][4])
    complete = len(bars) >= MIN_BARS_COMPLETE

    max_move_pct = (high / day_open - 1.0) * 100.0
    eod_return_pct = (close / day_open - 1.0) * 100.0

    def first_crossing(threshold_pct: float) -> int | None:
        target = day_open * (1.0 + threshold_pct / 100.0)
        for bar in bars:
            if float(bar[2]) >= target:
                return int(bar[0])
        return None

    anchor_ts = first_crossing(MOVE_THRESHOLD_PCT)
    early_deadline_ts = first_crossing(EARLY_THRESHOLD_PCT)

    return {
        "symbol": symbol,
        "utc_day": _utc_day(day_start_ms),
        "open": day_open,
        "high": high,
        "low": low,
        "close": close,
        "eod_return_pct": round(eod_return_pct, 6),
        "max_move_pct": round(max_move_pct, 6),
        "qualifies_move5": bool(complete and max_move_pct >= MOVE_THRESHOLD_PCT),
        "anchor_ts": anchor_ts,
        "early_deadline_ts": early_deadline_ts,
        "bars_used": len(bars),
        "complete": complete,
        "label_mature_at": _utc_day(day_start_ms + DAY_MS),
        "provenance": dict(provenance, builder_version=BUILDER_VERSION),
    }


def build_day_record_daily(symbol: str, day_start_ms: int, bar: list,
                           *, provenance: dict,
                           now_ms: int | None = None) -> dict[str, Any]:
    """One (symbol, UTC day) label from that day's DAILY kline.

    This is the global ranking tier: enough to rank a finished day by return,
    and deliberately not enough to time anything inside it. `anchor_ts` and
    `early_deadline_ts` stay `None` — a daily bar knows the day's high but not
    when it was reached, and inventing a time would be worse than not having one.

    Completeness is decided by whether the UTC day has closed, not by a bar
    count: `MIN_BARS_COMPLETE` counts hourly bars, and applying it here would
    mark every global record incomplete and drop it from every consumer.
    """
    if not bar:
        raise ValueError("cannot label a day with no bar")
    day_open = float(bar[1])
    if day_open <= 0:
        raise ValueError("non-positive open price")

    high, low, close = float(bar[2]), float(bar[3]), float(bar[4])
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    complete = now_ms >= day_start_ms + DAY_MS

    max_move_pct = (high / day_open - 1.0) * 100.0
    eod_return_pct = (close / day_open - 1.0) * 100.0

    return {
        "symbol": symbol,
        "utc_day": _utc_day(day_start_ms),
        "open": day_open,
        "high": high,
        "low": low,
        "close": close,
        "eod_return_pct": round(eod_return_pct, 6),
        "max_move_pct": round(max_move_pct, 6),
        "qualifies_move5": bool(complete and max_move_pct >= MOVE_THRESHOLD_PCT),
        # Intraday timing does not exist at this resolution. None, never 0.
        "anchor_ts": None,
        "early_deadline_ts": None,
        "bars_used": 1,
        "complete": complete,
        "resolution": "1d",
        "label_mature_at": _utc_day(day_start_ms + DAY_MS),
        "provenance": dict(provenance, builder_version=DAILY_BUILDER_VERSION),
    }


def resolution_of(rec: dict) -> str:
    """Records written before the daily tier existed carry no `resolution`.
    They are all hourly, and defaulting here beats backfilling a field into
    immutable records — which is precisely what the store forbids."""
    return rec.get("resolution") or "1h"


def summarise_universe_build(*, resolved: list, failed: list) -> dict[str, Any]:
    """Build summary that names what it could not fetch.

    A universe list fetched today has no delisted pairs, so a past day's global
    ranking is missing coins that were live then. That absence looks exactly
    like a quiet market unless it is counted (TH-05).
    """
    return {
        "symbols_resolved": len(resolved),
        "symbols_failed": len(failed),
        "failed_symbols": sorted(failed)[:50],
        "caveat": "survivorship: the universe is fetched today, so pairs "
                  "delisted since a past day are absent from that day's rank",
    }


class LabelStore:
    """Append-only, keyed by (symbol, utc_day). A record is written once."""

    def __init__(self, root: Path = DEFAULT_STORE) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "move_events_v1.jsonl"
        self._index: dict[tuple[str, str], dict] | None = None

    def _load(self) -> dict[tuple[str, str], dict]:
        if self._index is None:
            self._index = {}
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._index[(rec["symbol"], rec["utc_day"])] = rec
        return self._index

    @staticmethod
    def _identity(rec: dict) -> str:
        """Everything that must not change, hashed. `built_at` is excluded —
        rebuilding at a different time is not a contradiction."""
        payload = {k: v for k, v in rec.items() if k != "provenance"}
        prov = rec.get("provenance") or {}
        payload["_source_sha256"] = prov.get("source_sha256")
        payload["_builder_version"] = prov.get("builder_version")
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         default=str).encode()).hexdigest()

    def put(self, rec: dict) -> bool:
        """Write once. Returns True if newly written, False if already present.
        Raises when the same key would carry different content."""
        index = self._load()
        key = (rec["symbol"], rec["utc_day"])
        existing = index.get(key)
        if existing is not None:
            if self._identity(existing) != self._identity(rec):
                raise ImmutableLabelError(
                    f"{key} already labelled with different content — a written "
                    f"label is never overwritten; investigate the source instead")
            return False
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        index[key] = rec
        return True

    def records(self) -> list[dict]:
        return list(self._load().values())


def build_from_klines(source: Path = DEFAULT_SOURCE,
                      store_root: Path = DEFAULT_STORE) -> dict[str, Any]:
    """Label every complete UTC day present in the kline cache.

    The newest day is skipped: it is still forming, and a label written before
    its close would be the very hindsight this store exists to remove.
    """
    raw = Path(source).read_bytes()
    provenance = {"source": Path(source).name,
                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    data = json.loads(raw.decode("utf-8"))
    store = LabelStore(store_root)

    written = skipped_incomplete = skipped_forming = conflicts = 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for symbol, bars in data.items():
        by_day: dict[int, list[list]] = {}
        for bar in bars:
            day_start = (int(bar[0]) // DAY_MS) * DAY_MS
            by_day.setdefault(day_start, []).append(bar)
        for day_start, day_bars in sorted(by_day.items()):
            rec = build_day_record(symbol, day_start, day_bars, provenance=provenance)
            if rec["utc_day"] >= today:
                skipped_forming += 1
                continue
            if not rec["complete"]:
                skipped_incomplete += 1
                continue
            try:
                if store.put(rec):
                    written += 1
            except ImmutableLabelError:
                conflicts += 1
    return {"written": written, "skipped_incomplete": skipped_incomplete,
            "skipped_still_forming": skipped_forming, "conflicts": conflicts,
            "total_in_store": len(store.records()), "path": str(store.path)}


WELL_COVERED_FRACTION = 0.80


def coverage_by_day(recs: list[dict]) -> dict[str, int]:
    """How many symbols carry a label on each UTC day."""
    out: dict[str, int] = {}
    for rec in recs:
        out[rec["utc_day"]] = out.get(rec["utc_day"], 0) + 1
    return out


def stale_symbols(recs: list[dict], *, behind_days: int = 14) -> list[dict]:
    """Symbols whose newest label is far behind the store's newest day.

    The kline fetch returns whatever history exists, so a delisted or renamed
    symbol silently contributes days from a different year. Those labels are
    real, but pooling them with current days mixes eras and universes — so they
    are named rather than averaged away.
    """
    if not recs:
        return []
    newest = max(r["utc_day"] for r in recs)
    last_by_symbol: dict[str, str] = {}
    for rec in recs:
        symbol = rec["symbol"]
        if rec["utc_day"] > last_by_symbol.get(symbol, ""):
            last_by_symbol[symbol] = rec["utc_day"]
    out = []
    for symbol, last in sorted(last_by_symbol.items()):
        if (datetime.fromisoformat(newest) - datetime.fromisoformat(last)).days > behind_days:
            out.append({"symbol": symbol, "last_label": last})
    return out


def status(store_root: Path = DEFAULT_STORE) -> dict[str, Any]:
    return status_from_records(LabelStore(store_root).records())


def status_from_records(recs: list[dict]) -> dict[str, Any]:
    if not recs:
        return {"records": 0}
    coverage = coverage_by_day(recs)
    symbols = len({r["symbol"] for r in recs})
    days = sorted(coverage)

    # Rates are reported only over days where most of the universe is present.
    # A "per day" figure computed across a union of eras is not a rate, it is an
    # artefact of which symbols happened to have history.
    threshold = max(1, int(symbols * WELL_COVERED_FRACTION))
    covered = sorted(d for d, n in coverage.items() if n >= threshold)
    scoped = [r for r in recs if r["utc_day"] in set(covered)]
    qualifying = [r for r in scoped if r["qualifies_move5"]]

    st: dict[str, Any] = {
        "records": len(recs),
        "symbols": symbols,
        "days_any_coverage": len(days),
        "full_window": f"{days[0]}..{days[-1]}",
    }
    if covered:
        st.update({
            "well_covered_days": len(covered),
            "well_covered_window": f"{covered[0]}..{covered[-1]}",
            "qualifying_move5": len(qualifying),
            "qualifying_per_day": round(len(qualifying) / len(covered), 2),
            "base_rate_pct": round(100 * len(qualifying) / len(scoped), 2),
        })
    else:
        st["well_covered_days"] = 0
        st["note"] = "no day carries enough of the universe — rates withheld"
    stale = stale_symbols(recs)
    if stale:
        st["stale_symbols"] = [f"{s['symbol']}@{s['last_label']}" for s in stale]
    return st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="immutable later-EOD label store")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = ap.parse_args(argv)

    if args.build:
        res = build_from_klines(args.source)
        print(f"written {res['written']} · already present "
              f"{res['total_in_store'] - res['written']} · "
              f"incomplete {res['skipped_incomplete']} · "
              f"still forming {res['skipped_still_forming']} · "
              f"conflicts {res['conflicts']}")
    st = status()
    print("=" * 66)
    print("Immutable label store")
    print("=" * 66)
    for key, value in st.items():
        print(f"  {key:<22}{value}")
    if st.get("records"):
        print()
        print("  Ground truth is exchange OHLCV — the bot's own snapshots are")
        print("  never consulted, which is what makes these labels usable as")
        print("  ground truth for features the bot computed.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
