# Judgment checklist

Use this checklist after `truth_harness.py`; include the applicable TH IDs in
the audit or feature spec.

| ID | Evidence to inspect | Fail when |
|----|---------------------|-----------|
| TH-01 | numerator, denominator, base rate, lift, N | a standalone percentage is used as evidence |
| TH-02 | canonical metric map and claim wording | proxy/training value is called business progress |
| TH-03 | feature timestamp, immutable label timestamp, feature list/ablation | a feature sees the answer, label is contemporaneous, or rolling-24h membership is called EOD truth |
| TH-04 | train/val/test date ranges and day grouping | score is in-sample, post-fit, random-split, or splits a UTC day |
| TH-05 | schema, denominator, uptime and endpoint sample sizes | windows differ or downtime is scored as performance |
| TH-06 | population definition and maximum available period | a gate is tested only market-wide, not on bot candidates |
| TH-07 | flag, rollback, shadow/canary, risk anti-metrics | behaviour changes without a bounded reversible rollout |
| TH-08 | durable report/decision with period, N, metrics, verdict | a negative result can be forgotten or silently re-proposed |
| TH-09 | current MD claim against config, source and runtime | documentation describes a different bot |
| TH-10 | generated_at, source date, phase, coverage, unknown handling | stale/partial evidence supports a definitive conclusion |
| TH-11 | portfolio `alpha_vs_buy_and_hold_pct`; canonical ZigZag EX1 provenance | another PnL proxy answers “does the bot make money?” or deprecated EX1 proxy answers exit quality |
| TH-12 | spec, tests, verification, living-spec update | a change cannot be traced from requirement to evidence |

For an intentional exception, require a spec waiver containing: TH-ID, reason,
risk, owner and expiry. An undocumented exception is a failed audit.
