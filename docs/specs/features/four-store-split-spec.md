# Four-store split — severing research memory from the execution channel

- **Slug:** `four-store-split`
- **Status:** spec → implementation
- **Created:** 2026-08-14
- **Parent:** [`continuous-improvement-agent`](continuous-improvement-agent-spec.md) §6.1
- **Truth-harness invariants:** TH-09 (docs match reality), TH-12 (traceable)
- **Priority:** P0 — this is defect #0 in [`docs/ARCHITECTURE.md`](../../ARCHITECTURE.md)

## Problem

`.runtime/pipeline/decisions/decisions.jsonl` is described everywhere as the
loop's *memory*. It is also *executable*:

```
append an approved record
  → _config_runtime_overrides.apply_overrides() applies diff.to
    at every `import config`
  → pipeline_approve._maybe_auto_restart spawns restart_bot.bat
```

Live at the time of writing: `ENTRY_SCORE_MIN_15M` 40.0 → 35.0 and
`IMPULSE_SPEED_15M_HIGH_MOMENTUM_BYPASS_ENABLED` False → True. Both are gating
constants, and both differ from what `config.py` declares.

This is not hypothetical: **the most recent approved record in that file was
written by an LLM** during the 2026-08-13 bandit fix. It was a no-op and
operator-approved, but the write path was exercised. Anything able to append an
approved record can change live gating and trigger a restart — the definition of
a confused deputy.

Hardening (`_NEVER_OVERRIDABLE`, WARNING logging) made the channel visible and
harder to subvert. It did not separate memory from execution. This spec does.

## Design — four stores, one execution path

| Store | Path | Writer | Executable |
|---|---|---|---|
| Research ledger | `.runtime/control_plane/research_ledger.jsonl` | research tooling, incl. LLM | **no** |
| Promotion requests | `.runtime/control_plane/promotion_requests.jsonl` | governor, from a verified result | no |
| Signed approvals | `.runtime/control_plane/signed_approvals.jsonl` | governor + operator signature | no — it authorises, it does not act |
| **Runtime override store** | `.runtime/release/runtime_overrides.json` | `release_overrides.py` only | **yes — the only one `config.py` reads** |

The severing property is structural, not procedural: **`config.py` no longer
reads `decisions.jsonl` at all.** There is no code path from the research ledger
to the override store. A record appended to research memory — by an agent, by a
script, by anyone — cannot become behaviour without passing through
`release_overrides.py`, which accepts only a signed approval or an explicit,
recorded operator migration.

### Honest boundary of the guarantee

"Not readable/writable by the agent" is enforced here by **tooling boundaries,
not OS permissions**: one process, one user, one filesystem. What is guaranteed
is that no function reachable from the research path writes the override store,
and that this is asserted by test. Real credential separation belongs to a later
phase and is not claimed now.

### Signatures

A `SignedApproval` carries `HMAC-SHA256(operator_key, canonical_payload)`. The
key lives at `.runtime/release/operator.key`, is created by the operator, is
gitignored, and is never read or printed by any tool other than the verifier.
Without a key, new approvals cannot be signed and the release refuses them —
fail closed.

### Legacy entries, and why they are not silently grandfathered

The two live overrides predate this mechanism and carry no signature. Deleting
them would change live gating; accepting them silently would make the signature
requirement decorative. So each migrated entry records:

```
source = "legacy_decisions_jsonl"   provenance = [decision_id, …]
review_by = 2026-09-14              signature = null
```

They keep working, they are visibly unsigned, and the release tool reports them
as debt on every run. Re-approving them with a signature — or letting them
lapse — is an operator decision, not a silent default.

## Migration, and the property that must not break

**Live gating must be identical before and after.** The order is:

1. materialise the current effective overrides into the release store, with
   provenance;
2. prove equality: `config` loaded the old way and the new way must agree on
   every key, including the two live values;
3. only then switch the reader.

`AUTO_APPLY_OVERRIDES_ENABLED` keeps its meaning and its default (`True`), so
the operator's existing off-switch still works. `_NEVER_OVERRIDABLE` still
applies — a store entry cannot re-enable the mechanism, change the watchlist or
set the token.

## Verification

`test_four_store_split.py` asserts:

1. **the confused-deputy path is severed** — appending an approved record to the
   research ledger, with a real config key and a concrete value, does not change
   `config` after reload;
2. `config.py` contains no reference to `decisions.jsonl`;
3. a store entry for a protected key is refused;
4. an approval with a bad signature is refused by the release tool;
5. an approval with no signature is refused unless explicitly marked legacy;
6. legacy entries are reported as debt with their `review_by`;
7. migration is idempotent, and the effective config after migration equals the
   effective config before it — the two live overrides survive byte-identically;
8. the research ledger is append-only and never referenced by `config.py`.

## Behaviour safety (TH-06 / TH-07)

**Backtest: не применимо, and here is why that is not an evasion.** This change
asserts the opposite of a behaviour change — it claims live gating is *identical*
before and after. A backtest answers "is the new behaviour better"; there is no
new behaviour to compare. The correct evidence for an equivalence claim is an
equivalence proof, and it was run over the **maximum** affected set rather than a
sample: every key the override mechanism can touch.

The only edit to `config.py` is two comments pointing at the new store; no
constant changed. Procedure:

1. baseline captured from the live process before the switch
   (`.runtime/pre_split_config_snapshot.json`);
2. reader switched to the release store;
3. every key re-read and compared.

Result — all identical, including the two constants that were actually
overridden:

```
ENTRY_SCORE_MIN_15M                             35.0  ->  35.0   OK
IMPULSE_SPEED_15M_HIGH_MOMENTUM_BYPASS_ENABLED  True  ->  True   OK
AUTO_APPLY_OVERRIDES_ENABLED                    True  ->  True   OK
BANDIT_FORWARD_REWARD_ENABLED                   True  ->  True   OK
MAX_POLL_PER_CYCLE                                45  ->  45     OK
```

**Shadow / canary: не применимо.** Shadow mode exists to observe a new
behaviour before enforcing it; there is no new behaviour here. A canary
allocates exposure between two policies; both policies are the same policy. What
this change alters is *who may write* the gating values, not what they are —
so the meaningful safety property is the equivalence above plus the rollback
below, not a staged exposure.

If a future change to the store *does* move a gating value, it is an ordinary
behaviour change and inherits the full requirement: maximum-period backtest,
shadow first, flag and rollback.

## Findings from the architecture review

**A caller was left assuming the channel still worked.** `pipeline_approve.
_maybe_auto_restart` appended an approved record, printed "the bot will load the
new override", and spawned `restart_bot.bat`. After the split it would restart
the bot for nothing and report success — a silent no-op wearing a confident
message, which is worse than an error. It now refuses, explains that a decision
record no longer changes gating, and prints the real path: sign an approval, run
`release_overrides.py --apply`, restart. Removing the mechanism without
converting its callers would have been half a fix.

**The migration found seven effective overrides, not two.** The old reader
evaluated all seven and only *recorded* the two that differed from `config.py`;
the other five happened to equal their defaults. Carrying all seven is therefore
not a behaviour change — five are no-ops — and it makes the full set visible for
the first time. All seven are unsigned legacy debt with a review date.

**A test of mine was wrong, not the code.** The first version forbade the string
`decisions.jsonl` anywhere in the reader, which failed on the docstring
explaining that the file is no longer read. It now checks string constants the
code actually evaluates, docstrings excluded — use, not mention.

## Rollback

Delete `.runtime/release/runtime_overrides.json` and set
`AUTO_APPLY_OVERRIDES_ENABLED=False` to run on pure `config.py` defaults; or
restore the previous `_config_runtime_overrides.py` to read `decisions.jsonl`
again. The research ledger and decisions history are append-only and are never
deleted by any step here.

## Not in scope

Removing `pipeline_approve._maybe_auto_restart` (it spawns a restart but can no
longer change gating on its own), OS-level credential separation, and the
promotion governor's own logic. This spec severs the channel; it does not build
the loop that would use it.
