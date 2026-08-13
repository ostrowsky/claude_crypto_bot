"""Truth harness — mechanical part of CLAUDE.md §0a.

Checks the claims in CLAUDE.md against the code, the config, the datasets and the
running bot, then prints the checklist a human still has to answer. Exit code 1
if anything drifted, so it can gate a commit.

Every check here exists because the corresponding rule was violated in this repo
and cost real time:

  A. flags/sizes in CLAUDE.md vs reality           (rule 9)
  B. live bot matches what CLAUDE.md advertises    (rule 9)
  C. ratio metrics publish a base rate / lift      (rule 1)
  D. metric windows are uptime-aware               (rule 5)
  E. refuted backtests keep their verdict          (rule 8)

  pyembed\\python.exe files\\_harness_check.py
"""
from __future__ import annotations
import io, json, os, re, subprocess, sys
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
sys.path.insert(0, str(FILES))

problems: list[str] = []
notes: list[str] = []


def section(t: str) -> None:
    print("\n" + t)
    print("-" * len(t))


# --- A. CLAUDE.md vs config/datasets -------------------------------------
section("A. CLAUDE.md против config.py и датасетов (правило 9)")
r = subprocess.run([sys.executable, str(FILES / "_audit_md_vs_config.py")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
out = (r.stdout or "") + (r.stderr or "")
for line in out.splitlines():
    if line.strip().startswith("✗") or "расхождений нет" in line:
        print("  " + line.strip())
if r.returncode != 0:
    problems.append("CLAUDE.md описывает не тот бот, который работает (см. A)")

# --- B. live bot vs advertised behaviour ---------------------------------
section("B. Работающий бот против заявленного (правило 9)")
try:
    import config  # noqa: E402
    log = ROOT / "bot_stderr.log"
    txt = io.open(log, encoding="utf-8", errors="replace").read()[-4_000_000:] if log.exists() else ""
    hb = re.findall(r"monitoring_loop alive: (\d+) coins", txt)
    if getattr(config, "MONITOR_FULL_WATCHLIST", False):
        wl = len(config.load_watchlist())
        if hb:
            last = int(hb[-1])
            ok = last >= wl * 0.9
            print(f"  MONITOR_FULL_WATCHLIST=True · в наблюдении {last} из {wl} "
                  f"{'— ок' if ok else '— НЕ СООТВЕТСТВУЕТ'}")
            if not ok:
                problems.append(f"заявлен полный watchlist, а наблюдается {last} из {wl}")
        else:
            notes.append("нет heartbeat в логе — состояние наблюдения неизвестно")
    fresh = None
    if txt:
        m = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", txt, re.M)
        if m:
            fresh = datetime.strptime(m[-1], "%Y-%m-%d %H:%M:%S")
            age = datetime.now() - fresh
            print(f"  последняя запись в логе: {m[-1]} ({age.total_seconds()/60:.0f} мин назад)")
            if age > timedelta(minutes=30):
                problems.append(f"бот молчит {age.total_seconds()/60:.0f} мин — возможно, не работает")
except Exception as e:
    notes.append(f"проверка живого бота не выполнена: {e}")

# --- C. ratio metrics must publish a base rate ---------------------------
section("C. Доли публикуются с базой/лифтом (правило 1)")
RATIO = re.compile(r"\brecall|coverage_pct|hit_rate|silent_miss_pct", re.I)
BASE = re.compile(r"base\s*rate|базов|baseline|lift|лифт|precision|точност", re.I)
offenders = []
STRINGS = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'')
for p in sorted(FILES.glob("_backtest_*.py")) + sorted(FILES.glob("_compute_*.py")):
    src = io.open(p, encoding="utf-8", errors="replace").read()
    # Look at ALL string literals, not one line per print: a multi-line f-string
    # kept the base rate on a continuation line and the naive version still
    # flagged it.
    printed = "\n".join(STRINGS.findall(src))
    if RATIO.search(printed) and not BASE.search(printed):
        offenders.append(p.name)

def _staged() -> set[str]:
    """Files in this commit — the rule blocks NEW work, legacy debt only warns.
    Freezing the repo over pre-existing scripts would get the hook disabled,
    which is worse than the debt."""
    try:
        r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                           cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return {Path(x.strip()).name for x in (r.stdout or "").splitlines() if x.strip()}
    except Exception:
        return set()


staged = _staged()
if offenders:
    blocking = [o for o in offenders if o in staged]
    for o in offenders:
        mark = "✗" if o in blocking else "⚠"
        print(f"  {mark} {o}: печатает долю без базовой ставки/лифта"
              + (" (в этом коммите)" if o in blocking else " (легаси)"))
    if blocking:
        problems.append(f"{len(blocking)} изменяемый(х) скрипт(ов) печатают долю "
                        f"без базы (правило 1)")
    else:
        notes.append(f"{len(offenders)} легаси-скрипт(ов) печатают долю без базы — "
                     f"долг, чинить при следующем касании")
else:
    print("  все скрипты с долями печатают базу или лифт")

# --- D. uptime-aware metric windows --------------------------------------
section("D. Метрики учитывают простой (правило 5)")
for name in ("_compute_early_capture.py", "_backtest_top20_coverage_funnel.py"):
    src = io.open(FILES / name, encoding="utf-8", errors="replace").read()
    ok = "days_full" in src and "MIN_ACTIVE_HOURS" in src
    print(f"  {'ок' if ok else '✗'} {name}")
    if not ok:
        problems.append(f"{name} не учитывает нерабочие дни (правило 5)")

# --- E. refuted backtests keep their verdict -----------------------------
section("E. Отрицательные результаты зафиксированы (правило 8)")
no_verdict = []
for p in sorted(FILES.glob("_backtest_*.py")):
    src = io.open(p, encoding="utf-8", errors="replace").read()
    if not re.search(r"VERDICT|ВЕРДИКТ|RESULT|REFUTED|FINDING", src, re.I):
        no_verdict.append(p.name)
print(f"  бэктестов: {len(list(FILES.glob('_backtest_*.py')))}, "
      f"без записанного результата: {len(no_verdict)}")
if no_verdict:
    print("   " + ", ".join(no_verdict[:6]) + (" ..." if len(no_verdict) > 6 else ""))
    notes.append(f"{len(no_verdict)} бэктест(ов) без вердикта в файле — "
                 f"результат будет забыт и перепроверен заново")

# --- verdict --------------------------------------------------------------
print("\n" + "=" * 72)
if problems:
    print("НЕ СООТВЕТСТВУЕТ MD:")
    for p_ in problems:
        print("  ✗ " + p_)
else:
    print("Механические проверки пройдены.")
for n in notes:
    print("  ⚠ " + n)

print("""
Ответить руками (CLAUDE.md §0a) — то, что скрипт проверить не может:
  1. доля опубликована с базой и лифтом?
  2. названы признаки, которые могут содержать ответ?
  3. числа — из holdout, разбитого ПО ВРЕМЕНИ?
  4. окна сравнения сопоставимы по объёму данных?
  6. гейт проверен на СОБСТВЕННЫХ входах бота, а не на рынке?
  7. изменение поведения — за флагом, с откатом, сначала в shadow?
  8. отрицательный результат записан в репозиторий?
 10. отчёт не утверждает больше, чем показывают данные?
""")
sys.exit(1 if problems else 0)
