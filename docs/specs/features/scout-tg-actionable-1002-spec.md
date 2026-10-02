# trend-scout Telegram: only changes the goal backs (2026-10-02)

- **Slug:** `scout-tg-actionable-1002`
- **Status:** SHIPPED 2026-10-02 (operator: «да» to sending scout reports only when actionable)
- **Truth harness:** TH-02 / TH-11 (a proxy finding is not put to the operator as
  evidence), TH-07 (flag + rollback)
- **Rollback:** `TREND_SCOUT_TG_ONLY_ACTIONABLE = False` + RL-worker restart
- **Maximum period / shadow:** not applicable — messaging only; scout's analysis,
  log and changelog are unchanged

## Why

The 4-hourly scout report said «Монет с трендом: 52 из 67 … Пропустил: 48» and
asked to confirm `COOLDOWN_BARS 19 → 15` (risk high; scout backtest n = 496,
ret5 +0.07%, win 50%). "Trend" is scout's own score, not a day's winner; the
backtest is a 5-bar proxy on a 4-hour window; cooldown has no goal replay (no
blocked events) and a close idea (re-entering a leader inside the cooldown, X-8)
was refuted on 2026-09-28 (−0.08%/trade). The message was noise that read like an
alarm, against the operator's "no junk".

## Change (`files/trend_scout.py`)

- A report goes to Telegram only when a change was applied — which since
  2026-09-29 already needs an L3 `accept` (`TREND_SCOUT_AUTO_APPLY_REQUIRES_L3`) —
  or a medium/high-risk proposal (never auto-applied) has an L3 `accept`
  (`l3_verdict`, cached 7 days). Only those proposals are listed under
  «Требует подтверждения».
- Everything else stays in the log, `trend_scout_changelog.jsonl` and the scout
  state file, as before.
- Active after an RL-worker restart (scout runs inside it).

## Tests

`files/test_scout_tg_1002.py` (6).
