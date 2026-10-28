"""Daily operator report of the demo trader (demo-ledger-d-1013-spec.md §5).

Written to trader/runtime/reports/YYYY-MM-DD.txt for the previous UTC day. Delivery to the
operator is a Claude scheduled task at 09:00 local that reads this file and posts it to the chat;
this module only writes the file. Direction words follow the metric's rule: a direction only when
the CI excludes zero, otherwise "рано судить".

  (from trader/) ../pyembed/python.exe trader_report.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import daily_revenue as DR        # noqa: E402
import trader_config as TC        # noqa: E402

COUNTED = ("FULL", "BOT_DOWN")
STATUS_RU = {"FULL": "полный день", "BOT_DOWN": "бот жил < 95% времени (в результат счёта входит)",
             "RESET": "сброс или пополнение баланса (исключён)", "UNKNOWN": "нет отметки капитала (исключён)",
             "RECONCILE_FAIL": "учёт не сходится со счётом (исключён, тревога)"}


def _f(v, nd=3, sign=True):
    if v is None:
        return "н/д"
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def _on_day(ts, day: str) -> bool:
    return bool(ts) and str(ts)[:10] == day and str(ts)[10:11] in ("T", " ")


def _window(rows: list, day: str, n: int | None) -> str:
    rows = sorted((r for r in rows if r["day"] <= day), key=lambda r: r["day"])
    if n:
        start = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=n - 1)).strftime("%Y-%m-%d")
        rows = [r for r in rows if r["day"] >= start]
    series, fallback = [], 0
    for r in rows:
        v = r.get("R_policy_pct")
        if v is None and r.get("R_pct") is not None:      # snapshot without holdings: stablecoin split unknown
            v, fallback = r["R_pct"], fallback + (r.get("status") in COUNTED)
        if r.get("status") in COUNTED and v is not None:
            series.append((r["day"], float(v)))
    excluded = len(rows) - len(series)
    a = DR.aggregate(series, excluded=excluded)
    label = f"{n} дн" if n else "всё"
    if not a.get("n_days"):
        return f"  {label}: нет засчитанных дней (исключено {excluded})"
    fb = f", из них {fallback} по R счёта (стейблкоины не отделены)" if fallback else ""
    # below two bootstrap blocks the interval is degenerate (n = 7, block 5 -> 3 block starts), so no bounds
    ci = (f"[{_f(a['lo95'])}, {_f(a['hi95'])}]" if a["n_days"] >= DR.MIN_DAYS_DIRECTION
          else f"ДИ не определён (< {DR.MIN_DAYS_DIRECTION} дней)")
    return (f"  {label}: {_f(a['mean_pct'])} %/день {ci}, дней {a['n_days']}{fb}, "
            f"исключено {excluded}, итог {_f(a['compound_pct'], 2)}% -> {DR.verdict_text(a)}")


def _executor_block(day: str, rows) -> list:
    """Trades of the phase-C executor that day, from the decisions journal (C1 paper or C2 demo)."""
    rows = [r for r in (rows or []) if _on_day(r.get("ts"), day)]
    ent = [r for r in rows if r.get("kind") == "real_entry"]
    ex = [r for r in rows if r.get("kind") == "real_exit" and r.get("action") == "exit"]
    al = [r for r in rows if r.get("kind") == "alert"]
    modes = sorted({str(r.get("mode")) for r in rows if r.get("mode")})
    L = [f"  Исполнитель этапа C ({', '.join(modes) or 'нет записей'}): входов {sum(1 for r in ent if r.get('action') == 'enter')}, "
         f"пропусков {sum(1 for r in ent if r.get('action') == 'skip')}, выходов {len(ex)}"]
    skips = {}
    for r in ent:
        if r.get("action") == "skip":
            skips[r.get("reason")] = skips.get(r.get("reason"), 0) + 1
    if skips:
        L.append("    пропуски: " + ", ".join(f"{k} {v}" for k, v in sorted(skips.items(), key=lambda kv: -kv[1])))
    for r in sorted(ex, key=lambda r: r["ts"]):
        pnl, ret = r.get("pnl"), r.get("ret_pct")
        pnl_s = f"{pnl:+.2f} USDT" if pnl is not None else "н/д"
        ret_s = f", {ret:+.2f}%" if ret is not None else ""
        L.append(f"    {r.get('sym')}: выход {str(r['ts'])[11:16]}, {r.get('reason')}, {pnl_s}{ret_s}")
    if ex:
        tot = sum(float(r.get("pnl") or 0) for r in ex)
        L.append(f"    итог закрытых: {tot:+.2f} USDT")
    if al:
        codes = {}
        for r in al:
            codes[r.get("code")] = codes.get(r.get("code"), 0) + 1
        L.append("    тревоги: " + ", ".join(f"{k} {v}" for k, v in sorted(codes.items(), key=lambda kv: -kv[1])))
    return L


def _pct(v, nd=1):
    return "н/д" if v is None else f"{v * 100:.{nd}f}%"


def _seen_line(label: str, es) -> str:
    if not es or not es.get("n_leaders"):
        return f"  {label}: лидеров нет"
    return (f"  {label}: помечены до +2.5% -- {es['flagged_before_cross']} из {es['n_leaders']}, "
            f"помечены вообще -- {es['n_flagged_leaders']}; к моменту метки прошло (медиана) "
            f"{_pct(es.get('move_done_at_flag_median'), 0)} движения дня; "
            f"база: помечено пар {es['n_flagged_pairs']} из {es.get('n_pairs_day') or 'н/д'}, из них лидеров "
            f"{_pct(es.get('flagged_hit_share'))}, доля лидеров среди всех пар {_pct(es.get('base_rate'))}")


def _wide_block(w) -> list:
    """W1 scanner of the whole spot market (wide-scan-shadow-w1-1028-spec.md §3)."""
    L = ["5. Весь рынок (сканер W1, все пары спота)"]
    if not w or not w.get("coverage") or not w.get("leaders"):
        L.append("  нет данных сканера за эти сутки (сутки не полные или сканер не работал)")
        return L
    c = w["coverage"]
    L.append(f"  покрытие: баров {_pct(c.get('bars_share'))}, пар на баре {_pct(c.get('pair_share'))}, "
             f"задержка от закрытия бара p50 {c.get('latency_p50', 0):.0f} c / p95 {c.get('latency_p95', 0):.0f} c")
    L.append(_seen_line(f"топ-{len(w['leaders'])} рынка", w.get("early_seen")))
    L.append(_seen_line("из них вне 93 монет бота", w.get("early_seen_outside_bot")))
    L.append(f"  теневая книга x4_wide (все пары): R за сутки {_f(w.get('x4_wide_day'))}%; "
             f"x4_floor8 (93 монеты бота): {_f(w.get('x4_floor8_day'))}%")
    return L


def build(day: str, day_rows, shadow_rows=None, demo_fills=None, exchange_rows=None, snapshot_share=None,
          shadow_R=None, executor_rows=None, wide=None) -> str:
    rows = list(day_rows.values()) if isinstance(day_rows, dict) else list(day_rows or [])
    cur = next((r for r in sorted(rows, key=lambda r: r.get("revision", 0), reverse=True) if r["day"] == day), None)
    L = [f"Трейдер, демо-счёт: отчёт за {day} (сутки UTC)", ""]

    L.append("1. Выручка дня")
    if cur is None:
        L.append("  расчёт дня отсутствует (daily_revenue.py compute не отработал)")
    else:
        L.append(f"  статус: {cur.get('status')} -- {STATUS_RU.get(cur.get('status'), '')}")
        bnb = cur.get("fee_reserve_contrib_pct")
        L.append(f"  R счёта {_f(cur.get('R_pct'), 4)}%, вклад стейблкоинов {_f(cur.get('stable_contrib_pct'), 4)}%, "
                 + (f"вклад запаса BNB {_f(bnb, 4)}%, " if bnb else "")
                 + f"выручка политики {_f(cur.get('R_policy_pct'), 4)}%")
        if cur.get("reason"):
            L.append(f"  примечание: {cur['reason']}")
        L.append(f"  сверка счёт - учёт: {_f(cur.get('P_minus_L_pct'), 4)}% капитала")
        L.append(f"  бенчмарки: BTC {_f(cur.get('bench_btc_pct'), 2)}%, универсум (равные веса) "
                 f"{_f(cur.get('bench_universe_pct'), 2)}%, USDT 0")
    L.append("")
    L.append("2. Окна (выручка политики, только FULL + BOT_DOWN, 95% ДИ)")
    for n in (7, 30, None):
        L.append(_window(rows, day, n))
    L.append(f"  Направление называется, только если дней >= {DR.MIN_DAYS_DIRECTION} и ДИ не включает 0; "
             "иначе -- рано судить.")
    L.append("")

    L.append("3. Сделки предыдущего дня")
    fills = [f for f in (demo_fills or []) if _on_day(f.get("time"), day)]
    if not fills:
        L.append("  демо-счёт: нет сделок (ордера на демо-счёт ещё не включены)")
    else:
        fee = sum(float(f.get("commission_usdt") or 0) for f in fills)
        L.append(f"  демо-счёт: {len(fills)} исполнений, комиссии {fee:.2f} USDT")
        for f in sorted(fills, key=lambda f: f["time"]):
            L.append(f"    {f['time'][11:19]} {f['side']:4} {f['symbol']} {f['qty']} @ {f['price']}"
                     f" ({f.get('decision_id') or 'без решения'})")
    if executor_rows is not None:
        L.extend(_executor_block(day, executor_rows))
    exits = [r for r in (shadow_rows or []) if r.get("type") == "exit" and _on_day(r.get("exit_time"), day)]
    books = sorted({r["book"] for r in exits} | set(shadow_R or {}))
    if not books:
        L.append("  теневые книги E7: закрытых сделок нет")
    for b in books:
        bx = sorted((r for r in exits if r["book"] == b), key=lambda r: r["exit_time"])
        r_day = (shadow_R or {}).get(b, {}).get(day)
        head = f"  теневая книга {b}: закрыто {len(bx)}"
        if bx:
            wins = sum(1 for r in bx if float(r.get("pnl") or 0) > 0)
            head += (f", прибыльных {wins}, итог {sum(float(r.get('pnl') or 0) for r in bx):+.2f} USDT, "
                     f"средняя {sum(float(r['ret_pct']) for r in bx) / len(bx):+.2f}% после издержек")
        if r_day is not None:
            head += f"; R книги за сутки {r_day:+.3f}%"
        L.append(head)
        for r in bx:
            L.append(f"    {r['sym']}: вход {str(r.get('entry_ts'))[5:16]}, выход {str(r['exit_time'])[5:16]}, "
                     f"{r.get('reason')}, {float(r['ret_pct']):+.2f}%")
    L.append("")

    L.append("4. Здоровье")
    L.append(f"  доля 5-минутных снимков капитала: {_f(snapshot_share * 100 if snapshot_share is not None else None, 1, False)}%")
    ex = [r for r in (exchange_rows or []) if _on_day(r.get("ts"), day)]
    errs = [r for r in ex if not r.get("ok")]
    L.append(f"  запросов к бирже: {len(ex)}, ошибок {len(errs)}")
    codes = {}
    for r in errs:
        k = f"{r.get('path')} {r.get('status')}/{r.get('error_code')}"
        codes[k] = codes.get(k, 0) + 1
    for k, v in sorted(codes.items(), key=lambda kv: -kv[1])[:5]:
        L.append(f"    {k}: {v}")
    if wide is not None:
        L.append("")
        L.extend(_wide_block(wide))
    return "\n".join(L) + "\n"


def _jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _wide_report(day: str):
    """The W1 scanner's day (wide_scan.report); a failure there never breaks the daily report."""
    try:
        import wide_scan
        return wide_scan.report(day)
    except Exception as ex:
        return {"day": day, "coverage": None, "leaders": None, "error": repr(ex)[:200]}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import audit_report as AR
    import shadow_readout as SR
    argv = sys.argv[1:] if argv is None else argv
    day = argv[0] if argv else (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = DR.latest_by_day(TC.RUNTIME / "metrics" / "daily_revenue.jsonl")
    shadow_R = {}
    st_path = TC.RUNTIME / "shadow" / "state.json"
    if st_path.exists():
        st = json.loads(st_path.read_text(encoding="utf-8"))
        for name, b in (st.get("books") or {}).items():
            shadow_R[name] = dict(SR.series_from_marks(b.get("day_start_eq") or {}, st.get("started_utc")))
    cur = rows.get(day) or {}
    txt = build(day, rows, shadow_rows=_jsonl(TC.RUNTIME / "shadow" / "ledger.jsonl"),
                demo_fills=_jsonl(TC.RUNTIME / "trading" / "ledger.jsonl"),
                exchange_rows=AR.load("exchange", 3), snapshot_share=cur.get("snapshot_share"), shadow_R=shadow_R,
                executor_rows=[r for r in AR.load("decisions", 3) if r.get("process") == "executor"
                               or r.get("book") == "exec_x4"], wide=_wide_report(day))
    out = TC.RUNTIME / "reports" / f"{day}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
