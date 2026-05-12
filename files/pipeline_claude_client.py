"""Anthropic Claude API client for hybrid pipeline layers (L2/L5/L6).

Design goals:
  - **Optional**: if no API key is configured, every helper here returns None
    and callers fall back to deterministic logic. The pipeline must keep
    working with zero external dependencies.
  - **Auditable**: every call (prompt + response + cost estimate) is appended
    to .runtime/pipeline/claude_calls.jsonl so post-hoc review is possible.
  - **No SDK dependency**: uses urllib.request directly so we don't have to
    install anthropic into pyembed.

Configuration (any one of these works):
  - env ANTHROPIC_API_KEY
  - env CRYPTOBOT_PIPELINE_CLAUDE_KEY
  - file .runtime/pipeline/.claude_api_key   (first non-empty line)

Disable explicitly (forces deterministic mode even if key present):
  - env CRYPTOBOT_PIPELINE_NO_CLAUDE=1
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pipeline_lib as PL

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-opus-4-7"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

CALLS_LOG = PL.PIPELINE / "claude_calls.jsonl"

# Approximate $/Mtok (Opus 4.7 list price as of 2026-05). Used for audit only —
# never blocks calls. If pricing changes, update here.
PRICE_PER_MTOK_INPUT = {
    "claude-opus-4-7":                5.0,
    "claude-sonnet-4-6":              3.0,
    "claude-haiku-4-5-20251001":      1.0,
}
PRICE_PER_MTOK_OUTPUT = {
    "claude-opus-4-7":               25.0,
    "claude-sonnet-4-6":             15.0,
    "claude-haiku-4-5-20251001":      5.0,
}


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def _read_key_file() -> str | None:
    p = PL.PIPELINE / ".claude_api_key"
    if not p.exists():
        return None
    try:
        # utf-8-sig drops the BOM that PowerShell's `Out-File -Encoding utf8`
        # writes by default — without this the key string carries ﻿ and
        # urllib's HTTP layer chokes ("latin-1 can't encode ﻿").
        text = p.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for line in text.splitlines():
        # Strip BOM/zero-width chars in case the file was concatenated
        line = line.lstrip("﻿​").strip()
        if line and not line.startswith("#"):
            return line
    return None


def _sanitize_key(key: str | None) -> str | None:
    if not key:
        return None
    key = key.strip().lstrip("﻿​").strip()
    # ASCII-only enforcement; api keys are sk-ant-... pure ASCII anyway
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        return None
    return key or None


def get_api_key() -> str | None:
    if os.environ.get("CRYPTOBOT_PIPELINE_NO_CLAUDE", "").strip() == "1":
        return None
    raw = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("CRYPTOBOT_PIPELINE_CLAUDE_KEY")
        or _read_key_file()
    )
    return _sanitize_key(raw)


def is_enabled() -> bool:
    return bool(get_api_key())


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    pi = PRICE_PER_MTOK_INPUT.get(model, 0.0)
    po = PRICE_PER_MTOK_OUTPUT.get(model, 0.0)
    return round((in_tok / 1_000_000) * pi + (out_tok / 1_000_000) * po, 6)


def call_claude(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
    layer: str = "unknown",
    purpose: str = "unspecified",
    timeout_s: int = 60,
) -> dict | None:
    """Call the Anthropic Messages API.

    Returns a dict with keys: {text, raw, usage, model, cost_usd, latency_ms}
    or None if Claude is disabled / network failed. Callers must handle None.

    Every call is logged to CALLS_LOG with the prompt, response and usage so
    we can audit what Claude was asked and what it answered.
    """
    key = get_api_key()
    if not key:
        return None

    body = {
        "model":     model,
        "max_tokens": max_tokens,
        "system":    system,
        "messages":  [{"role": "user", "content": user}],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=data, method="POST",
        headers={
            "x-api-key":         key,
            "anthropic-version": API_VERSION,
            "content-type":      "application/json",
        },
    )

    started = datetime.now(timezone.utc)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Capture the response body — it contains Anthropic's structured error
        # ({"type":"error","error":{"type":"...","message":"..."}}) which is
        # what we actually need to diagnose 400/401/429/etc.
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass
        _log_call(layer, purpose, model, system, user,
                  error=f"HTTP {e.code}: {e.reason} | body={body}", started=started)
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        _log_call(layer, purpose, model, system, user, error=f"network: {e!r}", started=started)
        return None
    except json.JSONDecodeError as e:
        _log_call(layer, purpose, model, system, user, error=f"json: {e!r}", started=started)
        return None

    text = ""
    for block in payload.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")

    usage = payload.get("usage", {}) or {}
    in_tok  = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    cost = _estimate_cost(model, in_tok, out_tok)
    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    result = {
        "text":       text,
        "raw":        payload,
        "usage":      usage,
        "model":      model,
        "cost_usd":   cost,
        "latency_ms": latency_ms,
    }
    _log_call(layer, purpose, model, system, user, result=result, started=started)
    return result


def _log_call(
    layer: str,
    purpose: str,
    model: str,
    system: str,
    user: str,
    *,
    result: dict | None = None,
    error: str | None = None,
    started: datetime,
) -> None:
    rec = {
        "ts":      PL.utc_now_iso(),
        "layer":   layer,
        "purpose": purpose,
        "model":   model,
        "input": {
            "system_chars": len(system),
            "user_chars":   len(user),
            "system":       _truncate(system, 4000),
            "user":         _truncate(user, 8000),
        },
    }
    if error:
        rec["error"] = error
    if result:
        rec["output"] = {
            "text":       _truncate(result["text"], 8000),
            "usage":      result["usage"],
            "cost_usd":   result["cost_usd"],
            "latency_ms": result["latency_ms"],
        }
    PL.append_jsonl(CALLS_LOG, rec)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 30] + f"... [truncated {len(s) - limit} chars]"


# ---------------------------------------------------------------------------
# JSON-mode helper
# ---------------------------------------------------------------------------


def call_claude_json(
    system: str,
    user: str,
    *,
    schema_hint: str = "",
    **kwargs: Any,
) -> dict | None:
    """Wrapper around call_claude that asks for strict JSON and parses it.

    Returns the parsed dict or None on failure.  `schema_hint` is appended to
    `system` to nudge the model toward returning JSON-only output.
    """
    enforced = system.rstrip() + "\n\nRespond with a single valid JSON object only. No prose, no markdown fences, no comments."
    if schema_hint:
        enforced += f"\nExpected JSON shape: {schema_hint}"
    res = call_claude(enforced, user, **kwargs)
    if not res:
        return None
    text = (res.get("text") or "").strip()
    # Be lenient: strip markdown fences if model wraps anyway
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first ``` and any leading language hint, drop trailing ```
        while lines and lines[0].startswith("```"):
            lines.pop(0)
        while lines and lines[-1].startswith("```"):
            lines.pop()
        text = "\n".join(lines)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # one more rescue: find first '{' and last '}'
        i, j = text.find("{"), text.rfind("}")
        if i == -1 or j == -1 or j <= i:
            return None
        try:
            parsed = json.loads(text[i : j + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    parsed["__claude_meta__"] = {
        "model":      res["model"],
        "cost_usd":   res["cost_usd"],
        "latency_ms": res["latency_ms"],
        "usage":      res["usage"],
    }
    return parsed


# ---------------------------------------------------------------------------
# CLI: smoke test
# ---------------------------------------------------------------------------


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Claude client smoke test")
    ap.add_argument("--prompt", default="Respond with exactly the JSON: {\"ok\": true}")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not is_enabled():
        print("[claude-client] DISABLED — no API key found.")
        print("  Set ANTHROPIC_API_KEY env or write the key to:")
        print(f"  {PL.PIPELINE / '.claude_api_key'}")
        return

    print(f"[claude-client] key found, calling {args.model}...")
    res = call_claude_json(
        "You are a JSON-only responder.",
        args.prompt,
        model=args.model,
        max_tokens=100,
        layer="smoke_test",
        purpose="connectivity",
    )
    if res is None:
        # Pull the most recent error from the audit log and surface a useful hint
        last_err = ""
        if CALLS_LOG.exists():
            for rec in PL.iter_jsonl(CALLS_LOG):
                if rec.get("error"):
                    last_err = rec["error"]
        print("[claude-client] call failed.")
        if last_err:
            print(f"  reason: {last_err[:300]}")
            if "credit balance is too low" in last_err.lower():
                print("")
                print("  Anthropic API billing is separate from Claude.ai / Claude Code.")
                print("  Add credits at: https://console.anthropic.com/settings/billing")
            elif "invalid x-api-key" in last_err.lower() or "401" in last_err:
                print("  -> The API key is rejected. Generate a new one at:")
                print("     https://console.anthropic.com/settings/keys")
            elif "model" in last_err.lower() and "not_found" in last_err.lower():
                print("  -> Try --model claude-haiku-4-5-20251001 (cheaper, broadly available)")
        else:
            print("  see .runtime/pipeline/claude_calls.jsonl")
        return
    print(f"[claude-client] OK — response: {res}")
    print(f"  cost: ${res['__claude_meta__']['cost_usd']}, latency {res['__claude_meta__']['latency_ms']}ms")


if __name__ == "__main__":
    _main()
