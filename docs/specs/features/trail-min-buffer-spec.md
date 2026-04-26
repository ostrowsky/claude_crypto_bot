# Trail-stop minimum buffer (anti-whipsaw)

- **Slug:** `trail-min-buffer`
- **Status:** shipped
- **Owner:** Claude (with @ostr)
- **Created:** 2026-04-26
- **Shipped:** 2026-04-26
- **Related:** ALGOUSDT 04-22 → 04-26 trade post-mortem; bandit `tight` arm
  selection on `impulse_speed/15m`; user feedback "бот неправильно рано вышел".

---

## 1. Problem

On `impulse_speed` and `strong_trend` entries the LinUCB trail-bandit
sometimes selects the `tight` (×0.85) or `very_tight` (×0.70) arm. With
**ATR-only** trail buffers, those multipliers leave the stop within
~0.5 – 0.9 % of price even when *daily range* is 6 – 10 %. Result:
ordinary intra-bar dip kicks the stop, P&L exits at −0.3 … −0.9 %
within 2 – 4 bars, and price then continues up.

Concrete cases (last 14 d, ALGOUSDT, paired entry → exit):
- 04-22 `impulse_speed/15m`, trail_k=2.12 (`tight`), exit = ATR-trail hit at −0.41 %, hold = 47 min, daily_range ≈ 7 %.
- 04-26 same setup, exit at −0.62 %, hold = 33 min, then price moved +5 % over the next 4 h.

Backtest (30 d, all paired trades, `_backtest_trail_arm_pnl.py`) confirms
the structural pattern: on `impulse_speed/15m` and `strong_trend/1h`,
trail-stop hits dominate exits (~70 %) and arms tighter than `default`
account for the bulk of −0.3 … −0.7 % whipsaws on high-volatility days
(daily_range ≥ 5 %).

## 2. Success metric

**Primary:** on `impulse_speed`, `strong_trend`, `impulse` modes, fraction
of trades exiting via trail-stop with P&L ∈ (−1 %, 0 %) drops by ≥ 30 %
relative to the prior 30-day baseline, **without** lowering `recall@top20`.

**Secondary:** average hold (bars) on those modes ↑ by ≥ 1 bar.

## 3. Scope

### In scope
- New config block: per-mode minimum trail-stop buffer expressed as a
  fraction of price.
- Two helpers in `files/monitor.py`:
  `_trail_min_buffer_pct(mode)` and `_compute_trail_buffer(...)`.
- Apply the floor at **both** sites: initial trail (entry) and per-bar
  trail update.
- Log line `TRAIL FLOOR …` when the floor exceeds the ATR-buffer.

### Out of scope
- Changing bandit arm multipliers or removing arms.
- Changing `MAX_HOLD_BARS` or the fast-loss-exit logic.
- Volatility-aware sizing (separate spec).

## 4. Behaviour / design

For every entry and every trail update, the **effective buffer** is:

```
buffer = max( trail_k * ATR ,  min_pct(mode) * price )
trail  = price - buffer
```

`min_pct` defaults are conservative — non-zero only for the three modes
that demonstrably whipsaw on volatile names:

```python
# files/config.py  (new block at L425)
TRAIL_MIN_BUFFER_PCT_ENABLED: bool = True
TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED: float = 0.015  # 1.5 %
TRAIL_MIN_BUFFER_PCT_STRONG_TREND:  float = 0.015
TRAIL_MIN_BUFFER_PCT_IMPULSE:       float = 0.012
TRAIL_MIN_BUFFER_PCT_TREND:         float = 0.0
TRAIL_MIN_BUFFER_PCT_ALIGNMENT:     float = 0.0
TRAIL_MIN_BUFFER_PCT_RETEST:        float = 0.0
TRAIL_MIN_BUFFER_PCT_BREAKOUT:      float = 0.0
TRAIL_MIN_BUFFER_PCT_DEFAULT:       float = 0.0
```

Helpers (`files/monitor.py` ~ L1705):

```python
def _trail_min_buffer_pct(mode: str) -> float:
    if not getattr(config, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    if mode == "impulse_speed":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED", 0.015))
    # …per mode…
    return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_DEFAULT", 0.0))

def _compute_trail_buffer(*, price, trail_k, atr, mode):
    atr_buf = trail_k * atr if atr > 0 else 0.0
    min_pct = _trail_min_buffer_pct(mode)
    min_buf = min_pct * price if (price > 0 and min_pct > 0) else 0.0
    return max(atr_buf, min_buf)
```

Apply at:
- `init_trail` site (~ L4491) — entry creation.
- Trail-update site (~ L4830) — both for `pos.trail_k` and the
  `effective_trail_k` cap.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `TRAIL_MIN_BUFFER_PCT_ENABLED` | `True` | Master switch — `False` reverts to ATR-only buffer |
| `TRAIL_MIN_BUFFER_PCT_<MODE>` | see §4 | Per-mode floor as fraction of price |

**Rollback:** set `TRAIL_MIN_BUFFER_PCT_ENABLED = False` and restart.
Open positions keep their current trail; new positions get pure
`trail_k * ATR`.

## 6. Risks

- **Wider stops → larger losses on actual reversals.** Mitigation:
  caps are small (≤ 1.5 %), only on three impulse modes, and
  `MAX_HOLD_BARS` + fast-loss-exit still bound downside.
- **Bandit confusion.** The trail-bandit sees the *requested* `trail_k`,
  not the realised buffer. If the floor dominates, the bandit may keep
  picking `tight` because the realised P&L improves regardless.
  Acceptable for now — revisit when adding `realised_buffer_pct` to
  the context vector.
- **Mode mislabeling.** `signal_mode` on `pos` is set at entry; if the
  attribute is missing we fall back to `"trend"` (zero floor) — safe.

## 7. Verification

- [x] Backtest: `pyembed\python.exe files\_backtest_trail_arm_pnl.py`
      — confirmed `tight` arm whipsaw cluster on impulse_speed/strong_trend
      with daily_range ≥ 5 %.
- [x] Bot restarted 2026-04-26 13:26:25, clean startup, no crash on
      first poll.
- [ ] 7-day live: re-run the backtest and compare trail-hit-loss rate
      on the three target modes vs the prior 30 d baseline. (Pending —
      open this spec back up once data accrues.)

### Results (initial)

```
== Trade-by-arm breakdown, last 30d (pre-fix baseline) ==
impulse_speed/15m  tight  n=…  avg_pnl=+0.21%  trail_exit%=…
strong_trend/1h    tight  n=…  avg_pnl=…
…
```
(Baseline numbers captured pre-deploy; post-deploy comparison after
≥ 50 paired trades on target modes.)

## 8. Follow-ups

- Add `realised_buffer_pct` (= max(ATR%, floor%)) to the bandit context
  vector so arm selection internalises the floor.
- Consider an upper cap (e.g. 3 % of price) so the floor never dwarfs
  ATR on extreme-volatility days.
- Volatility-aware sizing spec — if buffer grows, position size should
  shrink to keep $-risk constant.
