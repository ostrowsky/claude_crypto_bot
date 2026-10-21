# RSI-divergence hold — SHADOW readout WEAK-HOLD-SHADOW (2026-10-20, approved and live 2026-10-21)

Status: **SHADOW live.** Designed 2026-10-20; the operator approved this design as written on
2026-10-21 ("тесты, потом код, флаг RSI_DIV_HOLD_SHADOW_ENABLED=True, перезапуск бота detached").
Tests: `files/test_weak_exit_shadow_1020.py`. The bot's exits are unchanged; only the event is new.
Deviation from §2 stated: the flag ships as True (the design's "False until approval" is satisfied).
Parent: `weak-exits-1005-spec.md` (rule accepted on the Jul–Oct half of a time split; that spec asks
for a clean confirmation in shadow after the X-9c readout).
Precondition met: X-9c readout = **KEEP** (2026-10-14, 14 days, rocket-day 15m trades n=52,
+3.976 % [+1.984, +6.289] against the +1.70 % threshold, no anomalies, switched share 36.2 %).
Truth-harness rules: TH-03 (data the replay did not use), TH-05 (unresolved holds are pending, not
zeros), TH-06 (the bot's own exits), TH-07 (shadow first, stated rollback), TH-10.

## 1. The rule under test (unchanged from the accepted one)

A **15m** position that would exit on `⚠️ WEAK: RSI дивергенция …` (`strategy.py:775`) with profit at
that bar's close **≥ +0.5 %** — a FIXED threshold for every mode — is held on the trail instead.
The per-mode H5 thresholds (`_h5_break_even_pct`: 0.3 / 0.5 / 0.7 %) are **not** used: that would ship
a rule that was never tested.

Maximum-period evidence (from the parent spec): the replay covered every real 15m trade since
2026-03-01, which is the start of the 15m long store, so it is the maximum available period
(4 313 trades). The rule was chosen on 03..06 and accepted on 07..10-05. This shadow adds the
unseen days from 10-06 on.

## 2. What changes live in shadow: only one log event

The bot exits exactly as today. At the moment it actually exits (after the MIN_WEAK_EXIT_BARS,
trend-hold and H5 checks, i.e. at the start of the `if reason:` exit block in
the position branch of `_poll_coin`), if the exit reason is an RSI divergence on 15m, it writes:

```
event = "weak_exit_shadow"
sym, tf, mode, bar_ts, price, pnl_pct      # pnl at the bar close, same basis as the replay
would_hold                                  # pnl_pct >= RSI_DIV_HOLD_MIN_PNL_PCT
threshold, trail_stop, trail_k, leader_mode # leader_mode must always be False (see §4)
reason
```

Config (new):
```python
RSI_DIV_HOLD_SHADOW_ENABLED = False   # True on operator approval; rollback = False + restart
RSI_DIV_HOLD_MIN_PNL_PCT = 0.5        # fixed for all modes -- NOT _h5_break_even_pct
```
No live `RSI_DIV_HOLD_ENABLED` flag is added now: a flag the live path never reads would be a claim
the code does not keep (CLAUDE.md §0a rule 13). It comes with the deploy, if the readout says KEEP.

### monitor.py — helper next to `_h5_should_suppress`
```python
def _rsi_div_hold_decision(reason: Optional[str], pnl_pct: float, tf: Optional[str]) -> Optional[bool]:
    """Weak-exit rule (weak-exits-1005-spec.md), SHADOW: None = not in scope,
    True = the accepted rule would hold this RSI-divergence exit on the trail.
    The threshold is FIXED for all modes on purpose -- not _h5_break_even_pct."""
    if tf != "15m" or not reason:
        return None
    r = reason.lower()
    if not ("rsi" in r and ("дивергенц" in r or "diverg" in r)):
        return None
    return float(pnl_pct) >= float(getattr(config, "RSI_DIV_HOLD_MIN_PNL_PCT", 0.5))
```
(The match is the same as `_backtest_weak_exits.category(...) == "rsi_divergence"`.)

### monitor.py — hook, first lines of the existing `if reason:` exit block
```python
    if reason:
        price = float(c[i])
        pnl   = pos.pnl_pct(price)
        # RSI-divergence hold, SHADOW (weak-exit-shadow-1020-spec.md): the bot still
        # exits below; the event records whether the accepted rule would have held.
        if bool(getattr(config, "RSI_DIV_HOLD_SHADOW_ENABLED", False)):
            _rdh = _rsi_div_hold_decision(reason, pnl, tf)
            if _rdh is not None:
                try:
                    botlog.log_weak_exit_shadow(
                        sym, tf, getattr(pos, "signal_mode", None), current_bar_ts, price, pnl, _rdh,
                        float(getattr(config, "RSI_DIV_HOLD_MIN_PNL_PCT", 0.5)),
                        float(getattr(pos, "trail_stop", 0.0) or 0.0), float(getattr(pos, "trail_k", 2.0)),
                        bool(getattr(pos, "leader_mode", False)), reason)
                except Exception as _rdh_err:
                    log.warning("weak exit shadow log failed for %s: %s", sym, _rdh_err)
        pnl_icon = ...   # unchanged from here on
```
The hook sets nothing on `pos`, never touches `reason`, never returns.

## 3. Counterfactual and readout (computed after the fact)

`readouts.readout_weak_exit_shadow` (new; `fn` in the registry):

- **Population:** every real 15m trade (entry→exit pairing of `_backtest_weak_exits.load_trades_with_reason`)
  whose entry day is a comparable day (bot up all day) in the window. Unchanged trades contribute a
  difference of 0 — the same pairing as the offline replay, so `all trades` and `rocket-day trades`
  mean the same thing as in the accepted test.
- **Changed trades:** exit category `rsi_divergence` AND profit at the exit bar's grid close ≥ +0.5 %.
- **Current arm:** close of the exit bar (grid close), as offline.
- **Hold arm, primary — P-1 trail + leader mode:** the P-1 plain trail (trade's trail_k, the mode's
  floor) ratcheted from the entry bar, as `hold_on_trail`; from bar ix+1 on, if the coin enters the
  leader set (`grid.lead(LEADER_EXIT_RANK_MAX, LEADER_EXIT_MIN_RET)`), the position switches to
  `exit_validator.wide_from` with the live `leader_params(config)`. This is what the rule would do if
  deployed today: since 2026-09-28 a held position can switch to leader mode.
- **Hold arm, secondary — plain P-1** (`_backtest_weak_exits.hold_on_trail`): descriptive, the
  number comparable with the 10-05 acceptance (+0.89 pp [+0.27, +1.65]).
- **Pending holds (TH-05):** a hold that has not crossed its stop and has fewer than 96 bars of data
  after the exit bar is *pending* — excluded and counted, never scored as its last close.
- **Labels:** rocket-days and winner-days from `exit_validator.day_labels()` (immutable label store).
- **Statistics and verdict:** `exit_validator.paired` + `exit_validator.decide` on the primary arm,
  unchanged: rocket-day lower 95 % > 0 AND all-trade lower 95 % ≥ −0.10 pp AND ≥ 60 % of months with
  ≥ 5 rocket-day trades ≥ 0. Mapping: `accept` → **KEEP** (= propose deploying the rule; a separate
  operator decision), `reject` → **ROLLBACK_SUGGESTED** (= do not deploy; shadow off), `needs_data` /
  `needs_review` → **INCONCLUSIVE** (= do not deploy; a longer window is a NEW registry entry, the
  criterion is never relaxed).
- **Anomalies (reported at once):** a `weak_exit_shadow` event with `leader_mode = True`; an event
  with tf ≠ 15m; an event with no 15m exit of the same symbol within its bar; after the switch-on day,
  disagreement > 5 % between the events' `would_hold` and the replay's "changed" set (live pnl vs grid
  close basis drift).

### Window and minimums (fixed now, before any number of the window is looked at)

| field | value | why |
|---|---|---|
| `window_from` | **2026-10-06** | the rule and the criterion were fixed on 10-05; trades from 10-06 were not in either half of the replay. The readout reconstructs them from exit events + 15m history, so the 14 days before the switch-on are usable; the shadow event cross-checks the days after it. |
| `min_days` | 42 comparable days | ~6 weeks; spans two calendar months for the month rule |
| `min_trades` (`_n_trades`) | 40 **changed** rocket-day trades | the effect lives there; leader mode now absorbs part of the rocket-day trades (diff 0) |
| also required by `decide` | all trades ≥ 200, rocket-day ≥ 30, changed ≥ 30 | `exit_validator` constants, unchanged |
| `due` | **2026-11-30** | before it the status is COLLECTING; after it TOO_EARLY until the minimums are met |

Rates behind the minimums (offline test half, pre-leader): ~17 trades/day, ~1.7 rocket-day
trades/day, ~2.5 changed trades/day. No live count since 10-06 was taken, to keep the window blind.
Power caveat stated in advance: at the offline effect size, ~40 changed rocket-day trades may give an
interval that still touches 0 — INCONCLUSIVE is a plausible and acceptable outcome.

## 4. Overlap with X-9c leader mode

- **Structurally disjoint at the exit decision.** `leader_exit.step` runs before
  `check_exit_conditions`; for a position in leader mode it returns `hold` or `exit` and
  the position branch of `_poll_coin` returns, so a leader-mode position never reaches the RSI-divergence exit
  and never produces a `weak_exit_shadow` event. An event with `leader_mode = True` is an ANOMALY.
- **The population is therefore not the offline one (TH-06).** The accepted half (Jul–Oct) was almost
  entirely before leader mode (09-28); now ~36 % of 15m trades switch, and they are disproportionately
  rocket-day trades. Part of what the rule gained offline may already be collected by leader mode.
  That is exactly the question this readout answers: what the rule adds **on top of** X-9c.
- **During the hold the two meet.** A held position can become a leader later; the primary hold arm
  switches to the leader trail at that bar (§3). The share of changed trades that switch is reported.
- **No contamination of X-9c.** X-9c is final (KEEP); in shadow nothing changes, so no live trade's
  outcome is altered.
- Not modelled (stated, as offline): other soft exits after the RSI-divergence bar (volume
  exhaustion, EMA patterns under H5, time limit, RL exit).

## 5. Tests (written before the code; `files/test_weak_exit_shadow_1020.py`)

1. Config: `RSI_DIV_HOLD_SHADOW_ENABLED` exists; `RSI_DIV_HOLD_MIN_PNL_PCT == 0.5`.
2. Helper: the exact `strategy.py` RSI-divergence string at pnl 0.50 → True, 0.49 → False; on `1h` → None;
   volume exhaustion / EMA fan / RSI overbought / ATR trail / H5 soft reasons → None.
3. Fixed threshold: with `H5_BREAK_EVEN_PCT_IMPULSE_SPEED = 0.3` and `..._STRONG_TREND = 0.7` patched,
   the helper's answers at pnl 0.4 and 0.6 do not change (it takes no mode at all).
4. Agreement with the replay: for a list of reasons, `helper is not None` ⇔
   `_backtest_weak_exits.category(reason) == "rsi_divergence"`.
5. Wiring (source): the hook sits inside the `if reason:` exit block after the H5 `elif`, before
   `await send(`; the hook's lines contain no `reason = None`, no `return`, no assignment to `pos.`.
6. Ordering (source): in the position branch of `_poll_coin`, `leader_exit.step` and its `return` on `hold`
   come before `check_exit_conditions`.
7. Event: `botlog.log_weak_exit_shadow` writes `event = "weak_exit_shadow"` with all fields of §2.
8. Readout on a synthetic grid: unchanged trades contribute 0; a changed trade whose trail runs up
   gives a positive difference; a changed trade that enters the leader set after the exit bar takes
   `wide_from`; an unresolved hold is pending, not scored; days before 2026-10-06 are excluded.
9. Verdict mapping accept/reject/needs_* → KEEP/ROLLBACK_SUGGESTED/INCONCLUSIVE; each anomaly of §3
   raises ANOMALY; below the minimums `status_of` gives COLLECTING / TOO_EARLY.
10. Registry: entry `WEAK-HOLD-SHADOW` exists, its `fn` is in `readouts.FNS`, fields of §6.
11. Docs: CLAUDE.md and PROJECT_CONTEXT.md mention the flag (same commit, §16).

## 6. Registry entry (draft)

```json
{
 "id": "WEAK-HOLD-SHADOW",
 "title": "RSI-divergence exit with profit >= +0.5% held on the trail (shadow, 15m)",
 "kind": "computed",
 "fn": "readout_weak_exit_shadow",
 "flag": "RSI_DIV_HOLD_SHADOW_ENABLED",
 "switched_on_utc": "<set at switch-on>",
 "window_from": "2026-10-06",
 "due": "2026-11-30",
 "min_days": 42,
 "min_trades": 40,
 "spec": "docs/specs/features/weak-exit-shadow-1020-spec.md",
 "expected": {
  "offline_test_rocket_day_pp": "+0.89 [+0.27, +1.65] (Jul-Oct half, n=163, plain P-1, pre-leader mode)",
  "offline_test_all_trades_pp": "+0.011 [-0.075, +0.105] (n=1633, 245 changed)",
  "note": "leader mode (X-9c, live since 09-28) now takes ~36% of 15m trades; the live effect may be smaller"
 },
 "criteria": {
  "keep": "primary arm (P-1 + leader switch): rocket-day lower 95% > 0 AND all-trade lower 95% >= -0.10 pp AND >= 60% of months (>= 5 rocket-day trades) >= 0 -> propose deploying the rule to the operator",
  "rollback": "all-trade lower 95% < -0.10 pp OR rocket-day upper 95% < 0 -> do not deploy, shadow off",
  "otherwise": "inconclusive -> do not deploy; a longer window is a new entry, the criterion is not relaxed"
 },
 "rollback": "nothing to undo while in shadow; RSI_DIV_HOLD_SHADOW_ENABLED = False + restart stops the logging"
}
```

## 7. Order of work on approval

tests (§5, red) → `config.py` flag + threshold → `botlog.log_weak_exit_shadow` → helper + hook in
`monitor.py` → `readout_weak_exit_shadow` + `FNS` + registry entry → tests green, `truth_harness.py
change --staged` → CLAUDE.md + PROJECT_CONTEXT.md → restart the bot detached (WMI,
`start_bot_detached.ps1`, `bot_stderr.log` backed up) → first `weak_exit_shadow` event checked in
`bot_events.jsonl`.
