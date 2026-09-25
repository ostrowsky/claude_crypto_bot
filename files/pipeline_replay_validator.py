"""L3 — replay validator keyed by config_key, not by the name of the rule.

WHY THIS EXISTS

Between 2026-06-21 and 2026-09-06 the weekly pipeline ran eleven times and L3
produced ZERO automatic verdicts: every one was `pending_manual_validation`.
`pipeline_validator.dispatch` looked validators up by `rule`, a free-form name
L2 invents ("trend_quality_forecast_floor_relax", "raise_lateness_cap_...",
"reduce_position_size_..."), and only three names were registered. Anything
else waited forever -- the one hypothesis still queued had waited a month.

L2 always supplies `config_key` and `diff.from -> diff.to`. For a threshold
whose compared value is written into bot_events.jsonl, that is enough to replay
the change on the bot's own decisions (TH-06):

    relax    rows the gate BLOCKED that the new value would have admitted
    tighten  rows the gate PASSED that the new value would have blocked

and to grade them against what the gate passes today by the thing the project
optimises -- the forward PEAK over the next 4 hours -- over the maximum period
the logs hold, broken out by month so one regime cannot carry it.

THE DECISION RULE, FIXED BEFORE ANY HYPOTHESIS IS GRADED

  band      the rows that change side under the proposed value
  control   rows that reached the gate and passed it under the current value
  mean_ratio  band mean peak / control mean peak
  tail_ratio  band share of peaks >= 5% / control share      (the goal is BIG moves)
  months    months with >= 20 band rows; consistency = share where band >= control

  relax    accept   n >= 60, mean_ratio >= 1.0, tail_ratio >= 1.0, consistency >= 0.6
           reject   mean_ratio < 1.0, tail_ratio < 1.0, consistency < 0.6
  tighten  accept   n >= 60, mean_ratio < 1.0, tail_ratio < 1.0, consistency < 0.4
           reject   mean_ratio >= 1.0, tail_ratio >= 1.0, consistency >= 0.6

  otherwise         needs_review -- the evidence disagrees with itself
  n < 30            needs_data   -- re-run automatically next week, never stuck

  Changed once, after the first run on 2026-09-18 and before any live verdict
  was written: reject originally ignored month consistency while accept used
  it, so a band that lost on the pooled mean but won in every month it could
  be scored in (the forecast-floor hypothesis, 100% over 3 months) was
  rejected outright. Both sides now require the months to agree.

Accept means "the evidence supports it", not "apply it": approval stays with
the operator. A key with no replay spec is REJECTED as unvalidatable, with the
reason, instead of being parked -- a change with no evidence path cannot pass
this project's max-period rule, and a queue that never drains is how the loop
stopped. `validatable_keys()` is handed to L2 so it can stop proposing them.
"""
from __future__ import annotations

import collections
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

EVENTS = HERE / "bot_events.jsonl"
HORIZON_H = 4
MIN_N = 30
ACCEPT_N = 60
MONTH_MIN = 20
CONSISTENCY = 0.6
BIG = 5.0

# Pipeline stage of each block reason. A row blocked at an EARLIER stage never
# reached the gate under test and belongs to neither side of the comparison.
STAGE = {
    "ml_zone": 1, "ml_proba_zone": 1,
    "trend_quality": 2, "trend_1h_chop": 2, "trend_chop": 2,
    "mode_range_quality": 3,
    "bandit": 4, "correlation_guard": 4, "clone_guard": 4,
    "clone_signal_guard": 4, "open_cluster_cap": 4, "late_continuation": 4,
    "rotation": 4, "impulse_guard": 4, "ranker_hard_veto": 4,
}

WEAK_TQ = re.compile(r"forecast (-?[\d.]+) < [\d.]+, vol ([\d.]+), ADX ([\d.]+), slope (-?[\d.]+)")


def _num(e, k):
    v = e.get(k)
    return float(v) if isinstance(v, (int, float)) else None


# --------------------------------------------------------------------------
# Replay specs: how the guard decides, from what the event log recorded.
# `blocks(e, cfg)` returns True/False, or None when the row lacks the inputs.
# --------------------------------------------------------------------------

def _tq_blocks(e, cfg):
    """monitor._trend_entry_quality_guard_reason, forecast/alt path only.

    Only rows the guard printed as `weak 15m trend` carry the forecast; the
    price-edge / RSI / daily-range checks run first and return earlier, so a
    row printed here already passed them."""
    m = WEAK_TQ.search(str(e.get("reason") or ""))
    if not m:
        return None
    fc, vol, adx, slope = map(float, m.groups())
    if fc >= cfg["TREND_15M_QUALITY_FORECAST_MIN"]:
        return False
    return not (vol >= cfg["TREND_15M_QUALITY_ALT_VOL_MIN"]
                and adx >= cfg["TREND_15M_QUALITY_ALT_ADX_MIN"]
                and slope >= cfg["TREND_15M_QUALITY_ALT_SLOPE_MIN"])


def _chop_blocks(e, cfg):
    """monitor._trend_1h_chop_guard_reason with the bull-day relax."""
    adx, slope, vol = _num(e, "adx"), _num(e, "slope_pct"), _num(e, "vol_x")
    if None in (adx, slope, vol):
        return None
    bull = bool(e.get("is_bull_day")) and cfg.get("TREND_1H_CHOP_USE_BULL_DAY_RELAX")
    sfx = "_BULL_DAY" if bull else ""
    return (adx < cfg["TREND_1H_CHOP_ADX_MIN" + sfx]
            or slope < cfg["TREND_1H_CHOP_SLOPE_MIN" + sfx]
            or vol < cfg["TREND_1H_CHOP_VOL_MIN" + sfx])


def _ml_blocks(e, cfg):
    p = _num(e, "ml_proba")
    if p is None:
        return None
    # Until the 2026-09-25 logging fix, blocks at trend_quality and later wrote
    # the candidate RANKER's quality_proba into ml_proba (monitor
    # _build_block_context). Such a row carries no ML score at all -- treat it
    # as unreplayable rather than grade the ranker's number against an ML floor.
    rq = _num(e, "ranker_quality_proba")
    if rq is not None and abs(rq - p) < 1e-12:
        return None
    floor = cfg["ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN" if e.get("is_bull_day")
                else "ML_GENERAL_HARD_BLOCK_MIN"]
    return p < floor


@dataclass(frozen=True)
class ReplaySpec:
    gate_codes: frozenset
    tf: Optional[str]
    blocks: Callable
    inputs: tuple            # every config key `blocks` reads
    directions: frozenset    # which of relax / tighten the log can replay
    note: str
    valid_since: str = ""    # rows before this are on a different scale / rule
    control_mode: str = ""   # if set, control = ENTRIES of this mode only


_TQ_KEYS = ("TREND_15M_QUALITY_FORECAST_MIN", "TREND_15M_QUALITY_ALT_VOL_MIN",
            "TREND_15M_QUALITY_ALT_ADX_MIN", "TREND_15M_QUALITY_ALT_SLOPE_MIN")
_CHOP_KEYS = ("TREND_1H_CHOP_ADX_MIN", "TREND_1H_CHOP_SLOPE_MIN", "TREND_1H_CHOP_VOL_MIN",
              "TREND_1H_CHOP_ADX_MIN_BULL_DAY", "TREND_1H_CHOP_SLOPE_MIN_BULL_DAY",
              "TREND_1H_CHOP_VOL_MIN_BULL_DAY", "TREND_1H_CHOP_USE_BULL_DAY_RELAX")
_ML_KEYS = ("ML_GENERAL_HARD_BLOCK_MIN", "ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN")

_TQ = ReplaySpec(frozenset({"trend_quality"}), "15m", _tq_blocks, _TQ_KEYS,
                 frozenset({"relax"}),
                 "passed rows do not log the forecast, so only relaxing replays; "
                 "control is trend/15m ENTRIES (only entries log the mode) -- they "
                 "also passed every later gate, which biases AGAINST relaxing",
                 control_mode="trend")
_CHOP = ReplaySpec(frozenset({"trend_1h_chop", "trend_chop"}), "1h", _chop_blocks,
                   _CHOP_KEYS, frozenset({"relax"}),
                   "passed rows do not log the entry mode, so tightening would "
                   "count modes the guard never judges; control is trend/1h "
                   "ENTRIES -- they also passed every later gate, which biases "
                   "AGAINST relaxing",
                   control_mode="trend")
_ML = ReplaySpec(frozenset({"ml_zone", "ml_proba_zone"}), None, _ml_blocks, _ML_KEYS,
                 frozenset({"relax", "tighten"}),
                 "ml_proba is logged on blocked and passed rows alike; only rows "
                 "since the peak-label model went live share today's scale",
                 valid_since="2026-09-07T19:31")

REPLAY_SPECS: dict[str, ReplaySpec] = {}
for _spec in (_TQ, _CHOP, _ML):
    for _k in _spec.inputs:
        if not _k.endswith("USE_BULL_DAY_RELAX"):
            REPLAY_SPECS[_k] = _spec


def validatable_keys() -> list[str]:
    return sorted(REPLAY_SPECS)


# --------------------------------------------------------------------------
# Outcome: forward peak from the long kline stores (see kline-long-store-spec)
# --------------------------------------------------------------------------

_BARS: dict = {}


def _bars(sym, tf):
    key = (sym, tf)
    if key not in _BARS:
        try:
            import _backtest_trend_start_detector as TD
            b = TD.load_bars(sym, "15m" if tf == "15m" else "1h")
        except Exception:
            b = []
        _BARS[key] = (b, {x[0]: i for i, x in enumerate(b)})
    return _BARS[key]


def forward_peak(sym, tf, dt, px, hours=HORIZON_H):
    bars, idx = _bars(sym, tf)
    step = 15 if tf == "15m" else 60
    bo = dt.replace(minute=(dt.minute // step) * step if step == 15 else 0,
                    second=0, microsecond=0)
    i = idx.get(bo)
    n = hours * (4 if tf == "15m" else 1)
    if i is None or px <= 0:
        return None
    fut = bars[i + 1: i + 1 + n]
    if len(fut) < n:
        return None
    return (max(b[2] for b in fut) / px - 1.0) * 100.0


# --------------------------------------------------------------------------

def _gate_of(e):
    if e.get("event") == "entry":
        return "ENTRY"
    return str(e.get("reason_code") or e.get("signal_type") or e.get("gate") or "")


def _iter_rows(spec: ReplaySpec, since: str):
    """Rows that REACHED the gate: blocked by it, blocked later, or entered.
    Deduplicated by (symbol, hour, gate) so a gate re-firing every poll does
    not weight its own opinion by loop frequency."""
    stage = min(STAGE[c] for c in spec.gate_codes)
    seen = set()
    with io.open(EVENTS, "rb") as fh:
        for raw in fh:
            if b'"blocked"' not in raw and b'"entry"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") not in ("blocked", "entry"):
                continue
            ts = str(e.get("ts") or "")
            if ts < since:
                continue
            tf = str(e.get("tf") or "")
            if spec.tf and tf != spec.tf:
                continue
            g = _gate_of(e)
            if g != "ENTRY" and g not in spec.gate_codes and STAGE.get(g, 0) <= stage:
                continue                      # stopped before reaching the gate
            px = e.get("price")
            if not e.get("sym") or not isinstance(px, (int, float)) or px <= 0:
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            key = (e["sym"], d.replace(minute=0, second=0, microsecond=0), g)
            if key in seen:
                continue
            seen.add(key)
            e["_dt"], e["_blocked_here"] = d, g in spec.gate_codes
            yield e


def _stats(v):
    if not v:
        return {"n": 0}
    s = sorted(v)
    return {"n": len(s), "median": round(s[len(s) // 2], 3),
            "mean": round(sum(s) / len(s), 3),
            "share_ge_5": round(sum(1 for x in s if x >= BIG) / len(s), 4)}


def validate(hyp: dict, since: str = "2026-01-01", cfg_module=None) -> dict:
    key = str(hyp.get("config_key") or "")
    diff = hyp.get("diff") or {}
    base = {"validator": "pipeline_replay_validator", "config_key": key}
    if cfg_module is None:
        import config as cfg_module
    if not key or not hasattr(cfg_module, key):
        return {**base, "verdict": "reject",
                "reason": f"config_key '{key}' does not exist in config.py"}
    spec = REPLAY_SPECS.get(key)
    if spec is None:
        return {**base, "verdict": "reject",
                "reason": (f"unvalidatable: no replay for '{key}' -- the event log "
                           f"does not record what it is compared against. Add a "
                           f"ReplaySpec in pipeline_replay_validator.py, or propose "
                           f"a key from validatable_keys()."),
                "validatable_keys": validatable_keys()}
    try:
        new_v = float(diff.get("to"))
    except (TypeError, ValueError):
        return {**base, "verdict": "reject", "reason": "diff.to is not a number"}
    cur = {k: getattr(cfg_module, k) for k in spec.inputs if hasattr(cfg_module, k)}
    cur_v = float(cur[key])
    new = dict(cur, **{key: new_v})
    if new_v == cur_v:
        return {**base, "verdict": "reject", "reason": "diff.to equals the live value"}

    # Both sides are judged by applying the CURRENT and the PROPOSED value to
    # the same logged inputs -- never by what the gate decided on the day. The
    # first version used the day's decision, and because floors and even the
    # model's scale moved over the period, rows blocked under an old rule were
    # counted as "admitted by the new value": it graded raising the ml floor
    # 0.15 -> 0.30 as a RELAX and accepted it (TH-04).
    window = max(since, spec.valid_since or since)
    relaxed_in, tightened_out, control, unreplayable = [], [], [], 0
    for e in _iter_rows(spec, window):
        b_cur, b_new = spec.blocks(e, cur), spec.blocks(e, new)
        if b_cur is None or b_new is None:
            if e["_blocked_here"]:
                unreplayable += 1
            else:
                # passed the gate on the day and its inputs were not logged
                # (e.g. the forecast on a passed trend row): treated as passing
                # under the current value too -- stated in the spec's note
                control.append(e)
            continue
        if b_cur and not b_new:
            relaxed_in.append(e)
        elif b_new and not b_cur:
            tightened_out.append(e)
        elif not b_cur:
            control.append(e)
    if spec.control_mode:
        # The first run compared trend-only blocked rows with every mode that
        # passed -- an unmatched control that graded relaxing the 1h chop ADX
        # floor 25 -> 20 as ACCEPT. Only entries record their mode.
        control = [e for e in control if e.get("event") == "entry"
                   and (e.get("signal_mode") or e.get("mode")) == spec.control_mode]
    # direction comes from where rows actually moved, not from the name L2 chose
    direction = "relax" if len(relaxed_in) >= len(tightened_out) else "tighten"
    moved_rows = relaxed_in if direction == "relax" else tightened_out
    if direction not in spec.directions:
        return {**base, "verdict": "reject",
                "reason": f"'{key}' can only be replayed as {sorted(spec.directions)}: {spec.note}"}

    def peaks(rows):
        out, by_m = [], collections.defaultdict(list)
        for r in rows:
            p = forward_peak(r["sym"], str(r.get("tf")), r["_dt"], float(r["price"]))
            if p is not None:
                out.append(p)
                by_m[r["_dt"].strftime("%Y-%m")].append(p)
        return out, by_m

    band, band_m = peaks(moved_rows)
    ctrl, ctrl_m = peaks(control)
    sb, sc = _stats(band), _stats(ctrl)
    report = {**base, "direction": direction, "from": cur_v, "to": new_v,
              "horizon_hours": HORIZON_H, "since": window,
              "band": sb, "control": sc, "unreplayable_rows": unreplayable,
              "note": spec.note}
    if sb["n"] < MIN_N or sc["n"] < MIN_N:
        return {**report, "verdict": "needs_data",
                "reason": (f"band n={sb['n']}, control n={sc['n']} (need {MIN_N} each); "
                           f"re-evaluated automatically on the next run")}
    mean_ratio = sb["mean"] / sc["mean"] if sc["mean"] else float("inf")
    tail_ratio = (sb["share_ge_5"] / sc["share_ge_5"]) if sc["share_ge_5"] else (
        float("inf") if sb["share_ge_5"] else 1.0)
    months = [m for m, v in band_m.items() if len(v) >= MONTH_MIN and ctrl_m.get(m)]
    wins = sum(1 for m in months
               if sum(band_m[m]) / len(band_m[m]) >= sum(ctrl_m[m]) / len(ctrl_m[m]))
    consistency = wins / len(months) if months else 0.0
    report.update(mean_ratio=round(mean_ratio, 3), tail_ratio=round(tail_ratio, 3),
                  months_scored=len(months), month_consistency=round(consistency, 3))
    better = mean_ratio >= 1.0 and tail_ratio >= 1.0
    worse = mean_ratio < 1.0 and tail_ratio < 1.0
    desc = (f"{direction} {key} {cur_v} -> {new_v}: {sb['n']} rows change side, "
            f"mean 4h peak {sb['mean']:.2f}% vs {sc['mean']:.2f}% passed "
            f"({mean_ratio:.2f}x), >=5% share {sb['share_ge_5']:.1%} vs "
            f"{sc['share_ge_5']:.1%} ({tail_ratio:.2f}x), month consistency "
            f"{consistency:.0%} over {len(months)}")
    if direction == "relax":
        if better and sb["n"] >= ACCEPT_N and consistency >= CONSISTENCY:
            v = "accept"
        elif worse and consistency < CONSISTENCY:
            v = "reject"
        else:
            v = "needs_review"
    else:
        if worse and sb["n"] >= ACCEPT_N and consistency <= 1.0 - CONSISTENCY:
            v = "accept"
        elif better and consistency >= CONSISTENCY:
            v = "reject"
        else:
            v = "needs_review"
    return {**report, "verdict": v, "reason": desc}


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--to", type=float, required=True)
    ap.add_argument("--since", default="2026-01-01")
    a = ap.parse_args()
    print(json.dumps(validate({"config_key": a.key, "diff": {"to": a.to}}, a.since),
                     ensure_ascii=False, indent=1))
