"""Stable codes for the free-text block reasons in `bot_events.jsonl`.

Blocked events carry `reason` as a human sentence, in two languages and several
spellings of the same thing — `MTF: 1м MACD` uses a Cyrillic `м` while
`MTF: 1m retest` uses Latin `m`, `<=` and `≤` both appear, and 449 rows say
`????????: портфель полон` where a cp1251 write mangled the prefix. 310 distinct
templates exist across 213 621 blocked rows.

Classifying by substring is fragile — rename a log line and the taxonomy
silently reclassifies — so this module is a bridge, not a destination. The
destination is a structured `reason_code` emitted at the decision site
(roadmap item E5); until then every report that groups blocks must go through
here rather than inventing its own matching, or two reports will disagree about
the same day.

`UNKNOWN` is deliberately visible in output: an unmatched reason is evidence
that a gate changed its wording, and hiding it as "other" is how a taxonomy
rots.
"""
from __future__ import annotations

import re

UNKNOWN = "unclassified"

# Order matters: the first match wins, so put the specific ones first.
# Each entry is (code, compiled pattern). Patterns are matched against the
# lower-cased "<signal_type> <reason>" string.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("portfolio_full", re.compile(r"портфель полон|portfolio full|лимит .*позици")),
    ("open_cluster_cap", re.compile(r"open cluster cap")),
    ("clone_signal_guard", re.compile(r"clone signal guard")),
    ("correlation_guard", re.compile(r"correlation guard|corr guard")),
    ("symbol_cooldown", re.compile(r"cooldown")),
    ("impulse_speed_curtail", re.compile(r"regime-curtailed|curtail")),
    ("impulse_speed_guard", re.compile(r"impulse_speed guard")),
    ("late_impulse_rotation", re.compile(r"late impulse_speed rotation")),
    ("late_continuation", re.compile(r"late .*continuation")),
    ("ranker_hard_veto", re.compile(r"ranker hard veto")),
    ("ranker_veto", re.compile(r"ranker veto")),
    ("bandit_skip", re.compile(r"bandit skip")),
    ("ml_proba_zone", re.compile(r"ml proba .*zone|outside profitable zone")),
    ("ml_quality", re.compile(r"^ml .*quality|ml trend\|")),
    ("entry_score", re.compile(r"entry score")),
    ("trend_quality", re.compile(r"trend quality guard")),
    ("trend_1h_chop", re.compile(r"chop:")),
    ("mode_range_quality", re.compile(r"mode_range_quality")),
    ("impulse_guard", re.compile(r"weak \d+[mhм] impulse|impulse guard")),
    ("mtf", re.compile(r"^mtf|mtf:|коррекц")),
    ("time_block", re.compile(r"time block")),
    ("fast_reversal_risk", re.compile(r"fast_reversal|fast reversal")),
]


# Two datasets describe the same block differently. `bot_events.jsonl` carries
# the free-text sentence the rules above match; `critic_dataset.jsonl` carries
# `decision.reason_code`, already short (22 distinct values). Feeding the latter
# through the regexes silently produced `unclassified` for every blocked top-20
# winner, so the harm table read "unclassified: 4" and named no gate at all.
_CODE_ALIASES: dict[str, str] = {
    "ml_zone": "ml_proba_zone",
    "cooldown": "symbol_cooldown",
    "portfolio": "portfolio_full",
    "correlation_guard_shadow": "correlation_guard",
    # `bot_events.jsonl` writes `trend_chop` (15 626 rows) for the same gate the
    # free text calls `trend/1h chop:`. Found by the event-store parity check,
    # which reported 7 517 blocks as `trend_1h_chop` from one field and
    # `unclassified` from the other — the same events, two spellings.
    "trend_chop": "trend_1h_chop",
    # not blocks; mapped so they can never be mistaken for one
    "entry_score_soft_pass": "entry_score_soft_pass",
    "near_miss": "near_miss",
    "take": "take",
    "rule_signal": "rule_signal",
    "bootstrap_ml_dataset": "bootstrap_ml_dataset",
}


def normalize_block_reason(reason: str = "", signal_type: str = "") -> str:
    """Map one block reason — free text or an existing code — onto a stable code.

    Unmatched text returns `unclassified` rather than a guess — see module
    docstring for why that is louder than an "other" bucket.
    """
    raw = str(reason or "").strip()
    token = raw.lower()
    if token in _CODE_ALIASES:
        return _CODE_ALIASES[token]
    if token in {code for code, _ in _RULES}:
        return token

    text = f"{str(signal_type or '').strip()} {raw}".lower()
    if not text.strip():
        return UNKNOWN
    for code, pattern in _RULES:
        if pattern.search(text):
            return code
    return UNKNOWN


def known_codes() -> list[str]:
    """Every code this module can emit, for report headers and tests."""
    return [code for code, _ in _RULES] + [UNKNOWN]
