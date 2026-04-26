# <Feature title>

- **Slug:** `<short-kebab-slug>`
- **Status:** draft | in-progress | shipped | reverted
- **Owner:** <name / agent>
- **Created:** YYYY-MM-DD
- **Shipped:** YYYY-MM-DD (fill in when merged)
- **Related:** links to issues, prior specs, log lines, screenshots

---

## 1. Problem

What is broken / missing / suboptimal? Be specific. Cite numbers from
`bot_events.jsonl`, backtests, or live observation. Avoid adjectives.

## 2. Success metric

How will we know this worked? One primary metric, optional secondary.
Examples:
- Reduce trail-hit-loss rate on `impulse_speed/15m` from X % to ≤ Y %
- Recall@top20 stays ≥ Z while false-entry rate drops by W

## 3. Scope

### In scope
- …

### Out of scope
- … (call out tempting adjacent changes that we explicitly defer)

## 4. Behaviour / design

What changes, where, and how. Reference exact file paths and line ranges
when known. Include the new config keys with default values.

```python
# files/config.py
NEW_FLAG_ENABLED: bool = True
NEW_THRESHOLD: float = 0.015
```

Diagrams or pseudo-code if it helps. Keep it short — code is the truth.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `…` | `…` | `…` |

**Rollback:** flip `<FLAG>=False` and restart. No data migration.

## 6. Risks

- What can go wrong? (over-blocking, regressions, latency, cost)
- Mitigation for each.

## 7. Verification

What must pass before we ship.

- [ ] Backtest: `python files/_backtest_<slug>.py` — paste summary below
- [ ] Live monitor: ≥ 30 min with no crash, expected log lines visible
- [ ] No regression in `recall@top20` (compare last-day vs prior 7-day avg)

### Results

(paste numbers / log excerpts here after running)

## 8. Follow-ups

Things deferred to a later spec.
