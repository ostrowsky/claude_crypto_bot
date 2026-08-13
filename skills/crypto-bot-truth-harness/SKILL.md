---
name: crypto-bot-truth-harness
description: Audit the claude_crypto_bot for truthful metrics, evidence provenance, MD/spec compliance, and safe change validation. Use for bot audits, report or metric reviews, claims of learning/progress, model or backtest results, roadmap decisions, and before completing any change to trading behaviour, gates, models, metrics, reports, or the auto-improvement loop.
---

# Crypto Bot Truth Harness

Treat a good-looking metric as unproven until its denominator, provenance,
timing and comparison contract are verified.

## Workflow

1. Locate the `claude_crypto_bot` root. Read `AGENTS.md`, `CLAUDE.md` §0/§0a,
   `PROJECT_CONTEXT.md`, `docs/specs/features/truth-harness-spec.md`, and the
   feature spec affected by the task.
2. Run the mechanical full profile before analysis:

   ```powershell
   pyembed\python.exe files\truth_harness.py full
   ```

   Use `--json .runtime/truth_harness/latest.json` when structured findings are
   useful. Never commit this runtime output.
3. Inspect every blocking finding at its source. Do not waive a check because a
   headline seems plausible. Distinguish current, stale, partial and unknown
   evidence.
4. Apply the judgment checks in
   [references/audit-checklist.md](references/audit-checklist.md). Mechanical
   success is necessary, not sufficient.
5. For a code change, verify the staged scope before handoff:

   ```powershell
   pyembed\python.exe files\truth_harness.py change --staged
   git diff --check
   ```

6. For any trading-policy relaxation, require a maximum-available-period,
   time-separated replay on the bot's actual candidate population, guardrails,
   rollback and shadow/canary evidence. Do not promote on a proxy improvement.
7. Report `PASS`, `FAIL`, or `UNKNOWN`. List violations with TH-ID, direct
   evidence, impact on the conclusion and the smallest safe remediation. Never
   claim compliance for an unchecked surface.

## Non-negotiable distinctions

- Training/proxy metric ≠ business outcome.
- Same-snapshot label recognition ≠ forward prediction.
- Rolling-24h snapshot membership ≠ immutable later-EOD top-20 ground truth.
- In-sample/post-fit score ≠ holdout achievement.
- Missing runtime data ≠ miss and ≠ success.
- Per-mode or per-trade PnL ≠ portfolio alpha.
- Deprecated proxy EX1 ≠ canonical ZigZag EX1.
- A ratio without numerator, denominator, base rate and lift is diagnostic-only.

If the Harness itself fails internally, return `UNKNOWN` and repair it before
using its output to approve another change.
