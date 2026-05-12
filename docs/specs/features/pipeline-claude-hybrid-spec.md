# Pipeline Claude Hybrid — L2 / L5 / L6 with Anthropic API

**Created:** 2026-05-11
**Status:** Implemented, fallback-default (no API key required to run)

## Why

L1, L3, L7 are deterministic and need to stay that way: они быстрые, дешёвые
и поведение полностью объяснимо. Слои L2/L5/L6 же выигрывают от добавления
интеллектуального второго мнения — без замены детерминированной логики.

Гибридная схема:

| Layer | Deterministic core | Claude role | Off-mode behaviour |
|-------|--------------------|-------------|--------------------|
| L1    | `pipeline_run.py`           | — none —                                | unchanged |
| L2    | `pipeline_hypothesis.py` rules | augment with extra hypotheses        | rules only |
| L3    | `pipeline_validator.py`     | — none —                                | unchanged |
| L4    | `pipeline_shadow.py`        | — none —                                | unchanged |
| L5    | `pipeline_blind_critic.py`  | parallel BLIND critic, agreement check  | rule verdict only |
| L6    | `pipeline_approve.py`       | advisor before y/n prompt (sees all)    | no advice shown   |
| L7    | `pipeline_monitor.py`       | — none —                                | unchanged |

## Configuration

The hybrid client (`pipeline_claude_client.py`) auto-detects an API key from
any of these sources (first match wins):

1. `ANTHROPIC_API_KEY` env var
2. `CRYPTOBOT_PIPELINE_CLAUDE_KEY` env var
3. `.runtime/pipeline/.claude_api_key` (first non-empty, non-comment line)

To temporarily disable Claude even if a key is configured:
- env `CRYPTOBOT_PIPELINE_NO_CLAUDE=1`, or
- CLI flag `--no-claude` (L2 only — L5/L6 use the env var)

To verify the client works:
```powershell
pyembed\python.exe files\pipeline_claude_client.py
```
Exit message either reports "DISABLED — no API key" (fallback path) or "OK — response: ..." with a tiny test call.

## Audit trail

Every Claude call is appended to `.runtime/pipeline/claude_calls.jsonl` with:
- timestamp, layer, purpose
- model, cost estimate, latency
- truncated prompt + response

L6 advisory recommendations are also written to
`.runtime/pipeline/approve_advisory.jsonl` (one record per `cmd_approve` call)
so we can later compare the advisor's recommendation with actual outcome (via L7).

## Layer details

### L2 augmentation
- After rule-based hypotheses are generated, Claude is given:
  - persistent red flags from L1 (days_red ≥ MIN_PERSISTENT_DAYS)
  - the rule layer's proposals (so it doesn't duplicate)
  - the last 50 already_tried entries
  - the full do_not_touch list
- Claude returns ≤3 additional hypotheses, tagged `generator: "claude"`.
- `filter_already_tried` + `filter_locked_keys` apply to both sources equally.
- A `dedup_by_key` pass drops duplicates with the same `(rule, config_key)`,
  keeping the rule version (deterministic priority).
- The final `_summary-<date>.json` now includes a `claude_augmentation` block
  with `n_rule`, `n_claude`, `n_after_dedup`.

### L5 hybrid blind critique
- Two critics run on the **same blind snapshot** (numbers before/after only).
- Both verdicts are stored in the critic record:
  - `rule_verdict`   — deterministic comparison
  - `claude_verdict` — Claude's independent read
  - `agreement`     — boolean
- If they **disagree**, the top-level `verdict` is downgraded to `needs_review`
  with a `reason` containing both verdicts. This is the value-add of running
  both: catching cases either alone would miss.
- Claude's input is built by `_build_blind_snapshot()` and contains ONLY:
  - the change description (config_key, diff, ts, expected_range)
  - the tracked metric values before/after
  - nothing from `hypothesis_id`, `rationale`, `validation_report`, `shadow_report`

### L6 advisory
- After `safety_checks()` runs (and either passes or is force-overridden),
  Claude is given EVERYTHING the human sees: hypothesis, validation, shadow,
  safety output.
- It returns one of `approve | reject | needs_review` plus:
  - `confidence`: low | medium | high
  - `justification`: one-paragraph reasoning
  - `watch_after_apply`: list of metrics to monitor for the first 24h after rollout
- In interactive mode, the advice is **printed before** the y/n prompt.
- In non-interactive mode (`--approve`), the advice is **logged but not printed**
  — the operator must read it from `approve_advisory.jsonl` if they care.
- Safety checks remain the **only** gate. Claude's recommendation is advisory.
- The decision record now includes a `claude_advice` field summarising the recommendation.

## Cost ceiling

Per-run cost estimates (Opus 4.7 pricing as of 2026-05):
- L2 weekly: ~3000 input + 1500 output tokens → ~$0.16
- L5 per decision: ~1500 input + 500 output → ~$0.06
- L6 per approval: ~2500 input + 700 output → ~$0.09

Realistic monthly: $0.50–$2.00. If costs spike, `CRYPTOBOT_PIPELINE_NO_CLAUDE=1`
flips everything back to rule mode without code change.

## Stress test compatibility

`pipeline_stress_test.py` sets `CRYPTOBOT_PIPELINE_NAMESPACE=stress_test` BEFORE
importing pipeline_lib, so all synthetic hypotheses and decisions go to an
isolated subdirectory. The Claude client is **not** namespace-aware (its log
goes to `.runtime/pipeline/claude_calls.jsonl` regardless), but its calls
during stress tests are tagged `layer=L6` / `purpose=approval_advice` and easy
to filter out.

The stress test does NOT depend on Claude — its three test cases exercise L3
validator and L6 safety_checks, both of which are rule-based.

## Rollback procedure

1. **All-off:** `set CRYPTOBOT_PIPELINE_NO_CLAUDE=1` (or unset
   `ANTHROPIC_API_KEY`). All three layers revert to rule-only behaviour.
2. **Per-layer:** delete the `pipeline_claude_client` import line at the top
   of the affected layer file; that layer reverts but others keep working.

## Verifying the integration

```powershell
# 1. Fallback path (no key) — must run cleanly
pyembed\python.exe files\pipeline_stress_test.py
# expected: 3/3 PASS, no Claude calls in claude_calls.jsonl

# 2. Smoke test (only if API key is configured)
pyembed\python.exe files\pipeline_claude_client.py

# 3. L2 with Claude (only if API key is configured)
pyembed\python.exe files\pipeline_hypothesis.py --print-summary
# inspect _summary-<date>.json: claude_augmentation.enabled should be True
```
