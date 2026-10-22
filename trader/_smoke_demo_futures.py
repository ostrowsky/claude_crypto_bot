"""F0 smoke of the DEMO account's USD-M futures (fc-passive-next-1022-spec.md §B).

Without --go: prints the plan and sends NOTHING. With --go (the operator's run):
  reads the demo futures account, sets one-way mode (only if hedge), ISOLATED margin and leverage 1 on
  BTCUSDT and ETHUSDT, sends one POST /fapi/v1/order/test (validated, never executed), compares demo and
  live funding. No position is opened. Never prints the key, the secret or a signature.
Writes trader/runtime/futures/F0_<UTC>.json. Exit code 0 = GO (or the plan), 1 = NO_GO.
Run from trader/:  ..\\pyembed\\python.exe _smoke_demo_futures.py --go
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit as A  # noqa: E402
import demo_futures as DF  # noqa: E402


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    go = "--go" in argv
    if go:
        A.set_process("smoke_demo_futures")
    out = DF.run_f0(go=go)
    if out["status"] == "PLAN":
        print("\n".join(out["plan"]))
        print("Run with --go to send these requests to the DEMO futures account.")
        return 0
    for s in out["steps"]:
        print(("  ok  " if s["ok"] else "  FAIL") + " " + s["step"] + ("" if s["ok"] else "  " + json.dumps(
            {k: v for k, v in s.items() if k not in ("step", "ok")}, default=str)))
    print(json.dumps({"results": out["results"], "readiness": out["readiness"]}, indent=1, default=str))
    print("F0:", out["status"], "->", out["file"])
    return 0 if out["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
