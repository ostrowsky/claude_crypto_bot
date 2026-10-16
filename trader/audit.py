"""Trader audit log: every Binance request and every decision (trader-audit-log-1012-spec.md).

Two append-only journals, one file per UTC day:
  trader/runtime/audit/exchange/YYYY-MM-DD.jsonl    one line per HTTP request (audit.exchange.v1)
  trader/runtime/audit/decisions/YYYY-MM-DD.jsonl   one line per decision      (audit.decision.v1)
A failed write never breaks the request or the decision: it is counted in `failures`.
Keys, secrets and signatures never reach a line (exchange_client strips them before calling here).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import trader_config as TC

ROOT = TC.RUNTIME / "audit"
SCHEMA = {"exchange": "audit.exchange.v1", "decisions": "audit.decision.v1"}
# endpoints whose response body IS the decision or the money (spec §2.1); others: size only
BODY_ENDPOINTS = ("/api/v3/order", "/api/v3/orderList", "/api/v3/account", "/api/v3/myTrades",
                  "/api/v3/openOrders", "/api/v3/openOrderList")
SECRET_PARAMS = ("signature", "apiKey", "apikey")
_PROCESS = {"name": os.environ.get("TRADER_PROCESS", "unknown")}
_VERSION = {}


def set_process(name: str) -> None:
    _PROCESS["name"] = name


def app_version() -> str | None:
    if "v" not in _VERSION:
        try:
            _VERSION["v"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(TC.REPO),
                                           capture_output=True, text=True, timeout=5).stdout.strip() or None
        except Exception:
            _VERSION["v"] = None
    return _VERSION["v"]


def decision_id(book, sym, tf, ts, kind) -> str:
    return hashlib.sha256(f"{book}|{sym}|{tf}|{ts}|{kind}".encode()).hexdigest()[:20]


def redact_params(params: dict) -> dict:
    return {k: v for k, v in (params or {}).items() if k not in SECRET_PARAMS}


def keep_body(path: str) -> bool:
    return any(path.startswith(p) for p in BODY_ENDPOINTS)


def summarize_body(path: str, body):
    """account: non-zero balances + canTrade only; other money endpoints: the whole body."""
    if path.startswith("/api/v3/account") and path == "/api/v3/account" and isinstance(body, dict):
        bal = [b for b in body.get("balances", []) if float(b.get("free") or 0) + float(b.get("locked") or 0) > 0]
        return {"canTrade": body.get("canTrade"), "accountType": body.get("accountType"), "balances": bal}
    return body


class AuditWriter:
    def __init__(self, kind: str, process: str | None = None, root: Path | None = None, run_id: str | None = None):
        self.kind, self.root = kind, Path(root) if root is not None else ROOT
        self.process = process
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.failures = 0

    def write(self, rec: dict, now: datetime | None = None) -> bool:
        try:
            now = now or datetime.now(timezone.utc)
            line = {"schema": SCHEMA.get(self.kind, self.kind), "ts": now.isoformat(timespec="milliseconds"),
                    "process": self.process or _PROCESS["name"], "run_id": self.run_id, "app_version": app_version(), **rec}
            d = self.root / self.kind
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            return True
        except Exception:
            self.failures += 1
            return False


_DEFAULT = {}


def default_writer(kind: str) -> AuditWriter | None:
    """Process-wide writer honouring the AUDIT_* flags (None when disabled)."""
    flag = {"exchange": "AUDIT_EXCHANGE_ENABLED", "decisions": "AUDIT_DECISIONS_ENABLED"}[kind]
    if os.environ.get("TRADER_AUDIT_DISABLED") == "1" or not getattr(TC, flag, False):   # tests: run_tests.py
        return None
    if kind not in _DEFAULT:
        _DEFAULT[kind] = AuditWriter(kind)
    return _DEFAULT[kind]
