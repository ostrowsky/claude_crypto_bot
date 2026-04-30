# AGENTS.md — Spec-first workflow for AI agents

This repository uses a **spec-first** workflow inspired by
[ostrowsky/spec-first-bootstrap](https://github.com/ostrowsky/spec-first-bootstrap).

For project-specific architecture, runtime, and known issues, read
[`CLAUDE.md`](./CLAUDE.md) and [`PROJECT_CONTEXT.md`](./PROJECT_CONTEXT.md) first.
This file is the **process** guide; those are the **system** guide.

---

## Three-layer model

```
Product Spec  ──►  Implementation  ──►  Verification
(docs/specs/)      (files/*.py)         (qa/, tests, backtests)
```

- **Product Spec** — *what* and *why*. Lives in `docs/specs/features/<feature>-spec.md`.
  Captures the problem, success metric, scope, behaviour, risks, rollback.
- **Implementation** — *how*. Code in `files/`. Each non-trivial change links back
  to its spec from the commit message.
- **Verification** — *did it work*. Backtest scripts, unit tests, live-event analysis.
  Results recorded in the spec under "Verification".

---

## Workflow for ANY non-trivial change

1. **Spec first.** Copy `docs/specs/templates/feature-spec.md` to
   `docs/specs/features/<short-slug>-spec.md`. Fill in problem, scope,
   acceptance criteria, rollback. Commit (or stage) the spec **before** code.
2. **Implement.** Touch only what the spec covers. If the change grows beyond
   the spec, stop and update the spec first.
3. **Verify.** Run the verification listed in the spec
   (backtest / unit tests / live monitor for N minutes). Paste the result
   into the spec's "Verification" section.
4. **Bump version** in `files/build_info.py`:
   - Increment `BUILD_VERSION` (semver: patch / minor / major).
   - Set `BUILD_APPLIED_AT_UTC` to the **current UTC ISO timestamp**
     (this is *when the new version is applied*, not when the bot
     starts and not the file mtime).
   - Update `BUILD_NOTES` with one-line summary.
   - `_append_version_history()` will record (version, applied_at)
     in `.runtime/version_history.jsonl` automatically on first
     import after the change.
5. **Commit.** Reference the spec path in the commit body:
   `Spec: docs/specs/features/<slug>-spec.md`. Include the version
   bump in the same commit.
6. **Update CLAUDE.md / PROJECT_CONTEXT.md** if the change touches sections
   2 / 4 / 5 / 7 / 8 / 11 of `CLAUDE.md` (architecture, ML, schedules,
   known issues, rotation, config flags).

### Version bump rules (semver)

- **patch** (x.y.**Z**): logger fixes, log-line wording, comment-only,
  retraining of an existing model, doc-only follow-ups in code.
- **minor** (x.**Y**.0): new feature behind a flag, new gate, new
  config tunable, new spec implementation that changes runtime
  behaviour for some path.
- **major** (**X**.0.0): breaking change in public API (rare for this
  project), restructure of pipeline, removal of an entry mode.

### When a spec is NOT required

- One-line typo or comment fix.
- Pure refactor with no behaviour change.
- Diagnostic / throwaway script (prefix file with `_diag_` or `_backtest_`).
- Emergency hotfix — write the spec **after**, same day, in the same branch.

---

## File conventions

| Path | Purpose |
|------|---------|
| `docs/specs/README.md` | Index of all feature specs |
| `docs/specs/templates/feature-spec.md` | Template — copy, do not edit in place |
| `docs/specs/features/*.md` | One file per feature/fix |
| `files/_diag_*.py` | Throwaway diagnostic scripts (gitignored OK) |
| `files/_backtest_*.py` | Throwaway backtest scripts (keep if reusable) |

Spec slugs: kebab-case, short, descriptive. Examples:
`trail-min-buffer-spec.md`, `fast-reversal-guard-spec.md`,
`rotation-ml-gate-spec.md`.

---

## Commit message format

```
<type>(<scope>): <short summary>

<body — what & why, link to spec>

Spec: docs/specs/features/<slug>-spec.md
```

Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.
Scopes match the codebase: `trail`, `rotation`, `bandit`, `ranker`,
`monitor`, `config`, `specs`.

---

## Anti-patterns (do not do)

- Writing 200+ lines of code with no spec, then back-filling docs.
- Hardcoding tunables in `monitor.py` instead of `config.py`.
- Tightening a filter without a Pareto sweep (see `CLAUDE.md` §4 lesson —
  three filters once blocked **100 %** of top gainers).
- Killing PIDs without checking cmdline (the sibling `gpt_crypto_bot`
  uses Python too).
- Reading the large jsonl/model blobs directly — they are deny-listed.
  Write a streaming aggregator instead.
