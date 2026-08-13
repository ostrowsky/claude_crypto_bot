# Health-report integrity and baseline recovery

- **Slug:** `health-report-integrity`
- **Status:** shipped
- **Owner:** Codex
- **Created:** 2026-08-13
- **Shipped:** 2026-08-13
- **Related:** `files/bot_health_report.py`, `.runtime/pipeline/baselines/`

---

## 1. Problem

The 2026-08-13 morning report ended with `Тревог нет` although no critic for
the report date had been loaded: `deployment_health.available=false` and the
North Star status was `unknown`.  It also compared the first-ever and latest
North-Star records across denominator-definition changes, labelled a 7% versus
7% result as flat, and called silent misses the main bottleneck even though the
current North-Star decomposition was coverage 61.5%, capture 17.0%, and time
lead 70.8%.

Production decisions `curtail_fallback_to_trend` and `alert_during_cooldown`
had no pinned pre-deploy baseline, so attribution returned `no_baseline` while
the Telegram summary said that their effect was still being calculated.

## 2. Success metric

- A morning report uses the latest completed final critic when today's final
  critic does not yet exist.
- `Тревог нет` is impossible when deployment evidence is missing, partial, or
  stale, or when more than the expected current partial day is absent.
- The progress verdict uses comparable, de-duplicated North-Star records only.
- Attribution made exclusively from `insufficient_data` is reported as
  unmeasurable, never as helped or harmed.
- The main bottleneck corresponds to the smallest current North-Star component.

## 3. Scope

### In scope

- Critic selection and evidence-freshness metadata.
- Data-quality red flags and defensive Telegram rendering.
- Comparable North-Star trend selection and bottleneck wording.
- Honest rendering of attribution outcomes.
- Runtime backfill of missing approved-decision baselines and a fresh
  attribution run.

### Out of scope

- Trading-policy, entry, exit, or mode changes.
- Fabricating pre-deploy observations before health reporting existed.
- Adding historical runtime baselines or generated reports to Git.

## 4. Behaviour / design

1. Critic priority is today's final, today's midday, then the newest final from
   the previous three days.  The selected critic records its target day and age.
2. A missing critic, a midday-only critic, or a final critic older than the most
   recently completed day creates a data-quality red flag.
3. North-Star history prefers rows carrying the current 14-day uptime-adjusted
   schema; dates are de-duplicated and only the recent comparable window is
   used for the progress verdict.  If endpoint windows differ by more than two
   full working days, the verdict is `РАНО СУДИТЬ` instead of a false trend.
4. The report displays capture/monetisation as the main bottleneck when it is
   lower than coverage and time lead.  Silent misses remain visible separately.
5. Attribution with no measurable expected metric is rendered as `нет данных
   для оценки`, including when the pipeline's outer verdict is `miss`.

## 5. Config flags & rollback

No new config flags.

**Rollback:** revert the health-report commit.  Runtime baseline JSON can remain;
it is generated evidence and has no live trading effect.

## 6. Risks

- A prior-day critic could be mistaken for same-day data.  Mitigation: expose
  `critic_target_date` and print it in the summary.
- Uptime warnings can include the current incomplete day.  Mitigation: tolerate
  one partial day and alert only on additional gaps.
- Older North-Star history can be incomparable.  Mitigation: prefer rows with
  the explicit uptime-adjusted schema and use a bounded legacy fallback only
  when fewer than two comparable rows exist.

## 7. Verification

- [x] Focused unit tests for critic fallback, missing/partial/stale evidence,
  comparable progress, bottleneck rendering, and insufficient attribution.
- [x] Existing bot-health critic-phase tests.
- [x] Fresh attribution after `pipeline_baseline.py --backfill-approved`.
- [x] Regenerated 2026-08-13 health JSON/Markdown/Telegram report.
- [x] `git diff --check`.

### Results

- Baseline backfill created the missing runtime snapshots.  The two production
  decisions now have 14 pre-deploy rows and all seven generic tracked metrics.
  The 2026-05-07 shadow decision predates health history and cannot be
  reconstructed (zero pre rows); shadow status does not require impact
  attribution.
- Fresh attribution `attribution-2026-08-13T094611Z.json`: 3 regression,
  2 `insufficient_data`, 1 `needs_data`.  In particular,
  fallback-to-trend and cooldown are now correctly `insufficient_data`, not
  `no_baseline` or `miss`.
- Tests: health integrity 9/9, critic phase 6/6, attribution 17/17.
- Regenerated report uses final critic 2026-08-12 and reports three critical
  alerts: uptime gap, early capture 0% for the completed critic day, and a
  73.33pp training-to-live gap.  The rolling headline is `РАНО СУДИТЬ` because
  its endpoints contain 5 versus 10 full working days.

## 8. Follow-ups

- Add decision-specific live metrics for cooldown re-alert precision/recall and
  fallback-to-trend net P&L; historical generic health baselines cannot
  reconstruct metrics that were never recorded.
