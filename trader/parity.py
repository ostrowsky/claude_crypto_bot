"""Demo vs paper parity of the phase-C executor (demo-paper-parity-1029-spec.md). Measurement only.

  pairs(decisions)          one row per executor key: real entry + real exit + the paper result of the same trade;
                            d = real ret - paper ret (pp), money gap, and its split into entry / fees / exit-and-rest
  coverage(decisions, day)  paper entries, entered, skips by reason, paper exits without a position, exchange stops
  windows(pairs, day)       7 / 30 / all days of the daily mean d, block bootstrap; a direction only with >= 10 days
  block(decisions, day)     the "6. Демо против бумаги" lines of the daily report
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import daily_revenue as DR        # noqa: E402
import fill_model as FM           # noqa: E402

BOOK = "exec_x4"
MODE = "demo"
_M = FM.model("taker", 1)                          # the paper book's model (executor.build)
MODEL_ENTRY_BPS = _M.half_spread * 1e4             # 8 bp from the signal price
MODEL_FEE_PCT = _M.taker_fee * 100                 # 0.10 % per side
BNB_FEE_PCT = FM.FEES["bnb"][1] * 100              # 0.075 % per side when paid in BNB
PAPER_RET_LOGGED_FROM = "2026-10-22"            # exits before it carry no paper_ret_pct (fix of 10-21, test_paper_replay_1021)
OUTLIER_PP = 1.0
MAX_OUTLIERS = 5


def _num(v):
    return float(v) if isinstance(v, (int, float)) else None


def _mine(r) -> bool:
    return r.get("book") == BOOK and r.get("mode") == MODE


def pairs(decisions) -> list:
    rows = sorted((r for r in decisions if _mine(r)), key=lambda r: str(r.get("ts")))
    out = {}
    for r in rows:
        k, kind, act = r.get("key"), r.get("kind"), r.get("action")
        if kind == "real_entry" and act == "enter":
            out[k] = {"key": k, "sym": r.get("sym"), "entry_ts": r.get("ts"), "cost": _num(r.get("cost")),
                      "slippage_bps": _num(r.get("slippage_bps")), "latency_sec": _num(r.get("latency_sec")),
                      "bnb_fee": (_num(r.get("fee_usdt")) or 0) > 0 and not r.get("fee_estimated"),
                      "exit_ts": None, "exit_reason": None, "ret_pct": None, "pnl": None, "paper_ret_pct": None}
        elif k in out and kind == "real_exit":
            p = out[k]
            if act == "exit":
                p.update(exit_ts=r.get("ts"), exit_reason=r.get("reason"), ret_pct=_num(r.get("ret_pct")),
                         pnl=_num(r.get("pnl")))
            if _num(r.get("paper_ret_pct")) is not None and (act == "exit" or r.get("reason") == "already_closed"):
                p["paper_ret_pct"] = _num(r.get("paper_ret_pct"))
    res = []
    for p in out.values():
        if p["exit_ts"] is None:
            p["state"] = "open"
        elif p["paper_ret_pct"] is None or p["ret_pct"] is None:
            p["state"] = "no_paper"
        else:
            p["state"] = "full"
            d = p["ret_pct"] - p["paper_ret_pct"]
            p["d_pp"] = d
            p["gap_usdt"] = (p["pnl"] - p["paper_ret_pct"] / 100 * p["cost"]) if (p["pnl"] is not None and p["cost"]) else None
            p["entry_pp"] = (MODEL_ENTRY_BPS - p["slippage_bps"]) / 100 if p["slippage_bps"] is not None else 0.0
            p["fee_pp"] = 2 * (MODEL_FEE_PCT - BNB_FEE_PCT) if p["bnb_fee"] else 0.0
            p["rest_pp"] = d - p["entry_pp"] - p["fee_pp"]
        p["exit_day"] = str(p["exit_ts"])[:10] if p["exit_ts"] else None
        p["entry_day"] = str(p["entry_ts"])[:10]
        res.append(p)
    return res


def coverage(decisions, day: str) -> dict:
    rows = [r for r in decisions if _mine(r) and str(r.get("ts", ""))[:10] == day]
    ent = [r for r in rows if r.get("kind") == "real_entry"]
    skips = {}
    for r in ent:
        if r.get("action") == "skip":
            skips[r.get("reason")] = skips.get(r.get("reason"), 0) + 1
    return {"paper_entries": len(ent), "entered": sum(1 for r in ent if r.get("action") == "enter"), "skips": skips,
            "no_real_position": sum(1 for r in rows if r.get("kind") == "real_exit" and r.get("action") == "skip"
                                    and r.get("reason") == "no_real_position"),
            "exchange_stops": sum(1 for r in rows if r.get("kind") == "execution_divergence")}


def _series(ps, day_from, day_to) -> list:
    by = {}
    for p in ps:
        if p["state"] == "full" and p["exit_day"] and day_from <= p["exit_day"] <= day_to:
            by.setdefault(p["exit_day"], []).append(p["d_pp"])
    return sorted((d, sum(v) / len(v)) for d, v in by.items())


def _verdict(a: dict) -> str:
    n = a.get("n_days", 0)
    if n < DR.MIN_DAYS_DIRECTION:
        return "рано судить"
    if a["lo95"] > 0:
        return "демо систематически лучше бумаги"
    if a["hi95"] < 0:
        return "демо систематически хуже бумаги"
    return "разница не отличима от 0"


def windows(ps, day: str) -> dict:
    d1 = date.fromisoformat(day)
    out = {}
    for name, n in ((7, 7), (30, 30), ("all", None)):
        d0 = "0000-00-00" if n is None else (d1 - timedelta(days=n - 1)).isoformat()
        a = DR.aggregate(_series(ps, d0, day))
        a["verdict"] = _verdict(a)
        out[name] = a
    return out


def _f(v, nd=2):
    return "н/д" if v is None else f"{v:+.{nd}f}"


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def block(decisions, day: str) -> list:
    L = [f"6. Демо против бумаги (книга {BOOK}, режим demo; d = демо − бумага, п.п. на сделку; "
         f"модель бумаги: вход {MODEL_ENTRY_BPS:.0f} б.п. от цены сигнала, комиссия {MODEL_FEE_PCT:.2f}% на сторону)"]
    ps = pairs(decisions)
    if not ps:
        L.append("  нет демо-сделок")
        return L
    full = [p for p in ps if p["state"] == "full" and p["exit_day"] == day]
    if full:
        n = len(full)
        m = lambda k: sum(p[k] for p in full) / n          # noqa: E731
        gap = sum(p["gap_usdt"] for p in full if p["gap_usdt"] is not None)
        L.append(f"  день: пар {n}, средняя d {_f(m('d_pp'))} п.п. (вход {_f(m('entry_pp'))}, комиссии {_f(m('fee_pp'))}, "
                 f"выход и прочее {_f(m('rest_pp'))}), разница в деньгах {gap:+.2f} USDT")
    else:
        L.append("  день: полных пар нет")
    ent = [p for p in ps if p["entry_day"] == day and p["slippage_bps"] is not None]
    if ent:
        sl = [p["slippage_bps"] for p in ent]
        lat = [p["latency_sec"] for p in ent if p["latency_sec"] is not None]
        L.append(f"  входы дня: {len(ent)}, проскальзывание от цены сигнала: среднее {sum(sl) / len(sl):+.1f} б.п., "
                 f"медиана {_q(sl, 0.5):+.1f}, p90 {_q(sl, 0.9):+.1f}; задержка медиана {_q(lat, 0.5) or 0:.0f} с")
    c = coverage(decisions, day)
    sk = ", ".join(f"{k} {v}" for k, v in sorted(c["skips"].items(), key=lambda kv: -kv[1])) or "нет"
    n_open = sum(1 for p in ps if p["state"] == "open")
    n_old = sum(1 for p in ps if p["state"] == "no_paper" and p["exit_day"] < PAPER_RET_LOGGED_FROM)
    n_np = sum(1 for p in ps if p["state"] == "no_paper") - n_old
    L.append(f"  покрытие дня: бумажных входов {c['paper_entries']}, исполнено {c['entered']}, пропуски: {sk}; "
             f"бумажных выходов без позиции {c['no_real_position']}; срабатываний биржевого стопа {c['exchange_stops']}")
    L.append(f"  сейчас: открытых пар {n_open}, закрытых без бумажного результата {n_np}"
             + (f" (и {n_old} до {PAPER_RET_LOGGED_FROM}: журнал тогда не писал бумажный результат)" if n_old else ""))
    w = windows(ps, day)
    for name, label in ((7, "7 дн"), (30, "30 дн"), ("all", "всё")):
        a = w[name]
        if not a.get("n_days"):
            L.append(f"  {label}: нет полных пар")
            continue
        ci = (f"[{a['lo95']:+.3f}, {a['hi95']:+.3f}]" if a["n_days"] >= DR.MIN_DAYS_DIRECTION
              else "ДИ не определён (< 10 дней)")
        L.append(f"  {label}: средняя d по дням {a['mean_pct']:+.3f} п.п. {ci}, дней {a['n_days']} -> {a['verdict']}")
    out = sorted((p for p in full if abs(p["d_pp"]) > OUTLIER_PP), key=lambda p: -abs(p["d_pp"]))[:MAX_OUTLIERS]
    for p in out:
        L.append(f"    выброс: {p['sym']} {p['exit_reason']}: демо {p['ret_pct']:+.2f}%, бумага {p['paper_ret_pct']:+.2f}%, "
                 f"d {p['d_pp']:+.2f} (вход {p['entry_pp']:+.2f}, выход и прочее {p['rest_pp']:+.2f})")
    return L
