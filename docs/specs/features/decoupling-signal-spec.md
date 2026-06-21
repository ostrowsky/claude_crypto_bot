# Decoupling signal — spec (2026-05-07)

## Status
SHADOW deployed (`DECOUPLING_SHADOW_ENABLED=True`). Gate NOT implemented
(`DECOUPLING_GATE_ENABLED=False`, reserved). No decision impact yet.

## Premise (validated)
A watchlist coin that is BOTH high-volatility AND decoupled from the
market (low trailing correlation to the equal-weight watchlist basket) is
~2× more likely to make a forward idiosyncratic big move (top-gainer
rocket) than a coin moving with the market.

Validation: `docs/reports/2026-05-07-decoupling-validation.md`
- 365d × 1h, train/holdout, no lookahead.
- Vol-controlled lift (high-vol tercile): +1.85pp train / +3.14pp holdout.
- Strict-rocket flag (top-5/day or daily ≥12%): precision ~10–13% vs base
  ~5.5% → LIFT ×1.87 holdout / ×2.25 train, recall 11–22%.
- It is a PRIORITY signal (~8 misses/hit), NOT a detector → intended as a
  bandit/ranker context feature, never a hard gate.

## Computation
`files/decoupling_signal.py`
- basket = equal-weight mean log-return across the watched coins.
- per coin: `trailing_corr` (Pearson to basket, 168×1h window),
  `vol` (realized), `vol_pctile` (cross-sectional rank).
- `decoupling_score = clamp((1-corr)/2, 0, 1) * vol_pctile` ∈ [0,1].
- `flag = vol_pctile ≥ DECOUPLING_VOL_PCTILE_MIN AND trailing_corr ≤ DECOUPLING_CORR_MAX`.
- `scores_from_rets()` is pure/testable; `__main__` self-test asserts a
  synthetic decoupled high-vol coin flags and a coupled one does not.

## Wiring (shadow)
- `monitor.py` monitoring loop: every `DECOUPLING_REFRESH_MIN` minutes,
  `decoupling_signal.compute_scores(session, watched_syms)` →
  `config._decoupling_scores` (fail-open). Reuses `correlation_guard`'s
  close-fetcher — no new fetch infra.
- `botlog.log_entry`: nullable `decoupling_score` / `decoupling_flag` /
  `decoupling_corr` written to `bot_events.jsonl` entry records.
- Call site `monitor.py` ~L5346 looks up `config._decoupling_scores[sym]`.

## Config (`config.py`)
```python
DECOUPLING_SHADOW_ENABLED = True      # rollback = False
DECOUPLING_TF = "1h"
DECOUPLING_WINDOW_BARS = 168          # ~7d (validated window)
DECOUPLING_VOL_PCTILE_MIN = 0.66
DECOUPLING_CORR_MAX = 0.60
DECOUPLING_REFRESH_MIN = 15
DECOUPLING_GATE_ENABLED = False       # reserved; needs shadow-replay first
```

## Acceptance before promoting to a bandit feature / gate
Per §4a / RM-22 pattern:
1. ≥7d shadow with entry records carrying `decoupling_*` fields.
2. Bandit shadow-replay: adding `decoupling_score` to context must improve
   early top-20 capture (NS) WITHOUT recall loss.
3. Only then add to `contextual_bandit` context vector (dim-migrate saved
   A/b like RM-22 Step B) and/or ranker features. Never a hard gate
   (precision too low — §7 over-block risk).

## Rollback
`DECOUPLING_SHADOW_ENABLED = False` — stops compute + logging, zero
behavior change (it was shadow anyway).
