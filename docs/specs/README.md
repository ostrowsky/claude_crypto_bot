# Specs index

Spec-first workflow. See [`../../AGENTS.md`](../../AGENTS.md) for the process.

## Templates

- [`templates/feature-spec.md`](./templates/feature-spec.md) — feature/fix template

## Features

| Slug | Status | Summary |
|------|--------|---------|
| [`features/trail-min-buffer-spec.md`](./features/trail-min-buffer-spec.md) | shipped 2026-04-26 | Per-mode minimum %-of-price floor on ATR trail-stop buffer to prevent whipsaw exits on volatile entries (impulse_speed, strong_trend, impulse). |

## How to add a new spec

```
cp docs/specs/templates/feature-spec.md docs/specs/features/<slug>-spec.md
$EDITOR docs/specs/features/<slug>-spec.md
git add docs/specs/features/<slug>-spec.md
# … then implement, verify, commit code with `Spec: docs/specs/features/<slug>-spec.md` in the body
```
