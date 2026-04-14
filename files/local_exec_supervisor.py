from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG = logging.getLogger("local_exec_supervisor")


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
JOB_ROOT = RUNTIME / "codex_jobs"
QUEUE_DIR = JOB_ROOT / "queue"
STATUS_DIR = JOB_ROOT / "status"
SUPERVISOR_STATUS = RUNTIME / "codex_exec_supervisor_status.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    for path in (RUNTIME, JOB_ROOT, QUEUE_DIR, STATUS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def update_supervisor_status(*, state: str, last_job_id: str = "", last_error: str = "") -> None:
    payload = {
        "state": state,
        "pid": os.getpid(),
        "updated_at": utc_now_iso(),
        "last_job_id": last_job_id,
        "last_error": last_error,
        "queue_dir": str(QUEUE_DIR),
        "status_dir": str(STATUS_DIR),
    }
    write_json(SUPERVISOR_STATUS, payload)


@dataclass
class JobResult:
    ok: bool
    summary: str
    exit_code: int = 0
    output_tail: str = ""
    spawned_pid: int = 0


def _tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_sync(command: list[str], cwd: Path | None = None, timeout: int = 3600) -> JobResult:
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    combined = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    return JobResult(
        ok=proc.returncode == 0,
        summary=f"exit={proc.returncode}",
        exit_code=proc.returncode,
        output_tail=_tail_text(combined.strip()),
    )


def _spawn_detached(command: list[str], cwd: Path | None = None) -> JobResult:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    return JobResult(ok=True, summary="spawned", spawned_pid=proc.pid)


def _handle_run_rl_headless(job: dict[str, Any]) -> JobResult:
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "start_rl_worker_bg.ps1"),
    ]
    return _run_sync(cmd, cwd=ROOT, timeout=180)


def _handle_stop_rl_headless(job: dict[str, Any]) -> JobResult:
    cmd = ["cmd.exe", "/c", str(ROOT / "stop_rl_headless.bat")]
    return _run_sync(cmd, cwd=ROOT, timeout=120)


def _handle_run_rl_train_once(job: dict[str, Any]) -> JobResult:
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "headless_train_once.ps1"),
    ]
    return _run_sync(cmd, cwd=ROOT, timeout=3600)


def _handle_run_top_gainer_critic(job: dict[str, Any]) -> JobResult:
    cmd = [
        str(ROOT / "pyembed" / "python.exe"),
        str(ROOT / "files" / "top_gainer_critic.py"),
        "--phase",
        str(job.get("phase") or "midday"),
    ]
    if job.get("date"):
        cmd.extend(["--date", str(job["date"])])
    return _run_sync(cmd, cwd=ROOT / "files", timeout=3600)


def _handle_check_telegram_connectivity(job: dict[str, Any]) -> JobResult:
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "$ErrorActionPreference='Stop';"
            "$tcp=Test-NetConnection api.telegram.org -Port 443 -WarningAction SilentlyContinue;"
            "$http=$null;"
            "try { $resp=Invoke-WebRequest https://api.telegram.org -Method Head -TimeoutSec 15 -UseBasicParsing; "
            "      $http=@{ ok=$true; status=[int]$resp.StatusCode } } "
            "catch { $http=@{ ok=$false; error=$_.Exception.Message } };"
            "$out=@{"
            "  computer='api.telegram.org';"
            "  tcp_success=[bool]$tcp.TcpTestSucceeded;"
            "  remote_address=$tcp.RemoteAddress.IPAddressToString;"
            "  remote_port=$tcp.RemotePort;"
            "  http=$http"
            "};"
            "$out | ConvertTo-Json -Depth 8"
        ),
    ]
    return _run_sync(cmd, cwd=ROOT, timeout=60)


def _handle_install_catboost(job: dict[str, Any]) -> JobResult:
    cmd = [
        str(ROOT / "pyembed" / "python.exe"),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "six",
        "python-dateutil",
        "pytz",
        "tzdata",
        "pandas",
        "catboost",
    ]
    return _run_sync(cmd, cwd=ROOT, timeout=7200)


def _load_telegram_token() -> str:
    env_path = ROOT / "files" / ".env"
    if not env_path.exists():
        raise RuntimeError(f".env not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if token:
                return token
    raise RuntimeError("TELEGRAM_BOT_TOKEN not found in files/.env")


def _handle_run_trade_bot(job: dict[str, Any]) -> JobResult:
    token = _load_telegram_token()
    pid_file = ROOT / ".runtime" / "bot_bg.json"
    launcher_log = ROOT / ".runtime" / "start_bot_bg.log"
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "start_bot_bg.ps1"),
        "-Token",
        token,
    ]
    result = _run_sync(cmd, cwd=ROOT, timeout=120)

    if pid_file.exists():
        try:
            payload = read_json(pid_file)
            wrapper_pid = int(payload.get("wrapper_pid") or 0)
            python_pid = int(payload.get("python_pid") or 0)
            if wrapper_pid > 0:
                return JobResult(
                    ok=True,
                    summary="spawned_and_confirmed" if python_pid > 0 else "spawned_wrapper",
                    exit_code=result.exit_code,
                    output_tail=json.dumps(
                        {
                            "wrapper_pid": wrapper_pid,
                            "python_pid": python_pid if python_pid > 0 else None,
                            "started_at": payload.get("started_at", ""),
                        },
                        ensure_ascii=False,
                    ),
                    spawned_pid=python_pid if python_pid > 0 else wrapper_pid,
                )
        except Exception:
            pass

    launcher_tail = ""
    if launcher_log.exists():
        try:
            launcher_tail = _tail_text(launcher_log.read_text(encoding="utf-8", errors="replace").strip())
        except Exception:
            launcher_tail = ""

    combined_tail = result.output_tail.strip()
    if launcher_tail:
        combined_tail = ((combined_tail + "\n\n" + launcher_tail).strip() if combined_tail else launcher_tail)

    return JobResult(
        ok=False,
        summary="bot_not_confirmed",
        exit_code=result.exit_code or 1,
        output_tail=combined_tail,
        spawned_pid=0,
    )


def _handle_stop_trade_bot(job: dict[str, Any]) -> JobResult:
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "$root='D:\\Projects\\gpt_crypto_bot';"
            "$pidFile=Join-Path $root '.runtime\\bot_bg.json';"
            "$ids=@();"
            "if (Test-Path $pidFile) {"
            "  try { $state=Get-Content $pidFile -Raw | ConvertFrom-Json; $ids += @($state.python_pid,$state.wrapper_pid) } catch {}"
            "}"
            "$ids=$ids | Where-Object { $_ -and [int]$_ -gt 0 } | ForEach-Object { [int]$_ } | Select-Object -Unique;"
            "foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue };"
            "if (Test-Path $pidFile) { Remove-Item $pidFile -Force -ErrorAction SilentlyContinue };"
            "Write-Host ('Stopped trade bot PIDs: ' + ($(if ($ids) { $ids -join ', ' } else { 'none' })))"
        ),
    ]
    return _run_sync(cmd, cwd=ROOT, timeout=120)


JOB_HANDLERS = {
    "run_rl_headless": _handle_run_rl_headless,
    "stop_rl_headless": _handle_stop_rl_headless,
    "run_rl_train_once": _handle_run_rl_train_once,
    "run_top_gainer_critic": _handle_run_top_gainer_critic,
    "check_telegram_connectivity": _handle_check_telegram_connectivity,
    "install_catboost": _handle_install_catboost,
    "run_trade_bot": _handle_run_trade_bot,
    "stop_trade_bot": _handle_stop_trade_bot,
}


def _list_queue_files() -> list[Path]:
    return sorted(
        QUEUE_DIR.glob("*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _load_job(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    payload.setdefault("job_id", path.stem)
    payload.setdefault("job_type", "")
    return payload


def _job_status_path(job_id: str) -> Path:
    return STATUS_DIR / f"{job_id}.json"


def _write_job_status(job: dict[str, Any], *, state: str, result: JobResult | None = None, error: str = "") -> None:
    payload: dict[str, Any] = {
        "job_id": job.get("job_id", ""),
        "job_type": job.get("job_type", ""),
        "state": state,
        "requested_at": job.get("requested_at", ""),
        "updated_at": utc_now_iso(),
        "request": job,
        "error": error,
    }
    if state == "running":
        payload["started_at"] = utc_now_iso()
    if result is not None:
        payload["result"] = {
            "ok": result.ok,
            "summary": result.summary,
            "exit_code": result.exit_code,
            "output_tail": result.output_tail,
            "spawned_pid": result.spawned_pid,
        }
    write_json(_job_status_path(str(job.get("job_id", ""))), payload)


def process_job(path: Path) -> bool:
    job = _load_job(path)
    job_type = str(job.get("job_type") or "")
    handler = JOB_HANDLERS.get(job_type)
    if handler is None:
        _write_job_status(job, state="failed", error=f"unsupported job_type: {job_type}")
        path.unlink(missing_ok=True)
        LOG.warning("Unsupported job_type=%s", job_type)
        return False

    _write_job_status(job, state="running")
    update_supervisor_status(state="running_job", last_job_id=str(job.get("job_id", "")))
    try:
        result = handler(job)
        _write_job_status(job, state="completed" if result.ok else "failed", result=result)
        path.unlink(missing_ok=True)
        update_supervisor_status(
            state="idle",
            last_job_id=str(job.get("job_id", "")),
            last_error="" if result.ok else result.summary,
        )
        LOG.info("Processed job_id=%s type=%s ok=%s", job.get("job_id", ""), job_type, result.ok)
        return result.ok
    except Exception as exc:
        _write_job_status(job, state="failed", error=str(exc))
        path.unlink(missing_ok=True)
        update_supervisor_status(state="idle", last_job_id=str(job.get("job_id", "")), last_error=str(exc))
        LOG.exception("Job failed job_id=%s type=%s", job.get("job_id", ""), job_type)
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local execution supervisor for Codex-triggered jobs.")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    parser.add_argument("--poll-sec", type=float, default=5.0, help="Queue polling interval in seconds.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    ensure_dirs()
    update_supervisor_status(state="idle")
    LOG.info("Local exec supervisor started poll_sec=%s", args.poll_sec)

    while True:
        jobs = _list_queue_files()
        if jobs:
            process_job(jobs[0])
            if args.once:
                break
        else:
            update_supervisor_status(state="idle")
            if args.once:
                break
            time.sleep(max(args.poll_sec, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
