"""Model health: input drift that says something, quality against the model's OWN target, run manifests.

WHY (2026-10-05, prod-tools plan step 2, model-health-1005-spec.md)
`pipeline_drift.py` (RM-14) has run daily since 2026-05-19 and answered
"drift_detected" on 164 of 166 runs: a two-sample KS test on thousands of rows
calls every small shift significant. It tracked model OUTPUTS (ml_proba, scores),
which move whenever a model is retrained -- every night -- and `ranker_proba`,
which critic_dataset never logs (0 of 166 runs). Nothing consumed its output.
Nothing measured whether ml_proba still does the job it is trained for, and
nothing recorded what each nightly retrain changed.

First version of this module (same day) repeated two of those mistakes, and the
backfill showed it: PSI on model outputs alerted on 55% of 139 days (mostly model
changes, e.g. the 09-07 label switch, PSI 2.16), and ml_proba "quality" was
graded against ret_10 although the model has been trained since 2026-09-07 on
the PEAK label (high within 5 bars >= +2%) -- the wrong yardstick (TH-11).

WHAT IT MEASURES NOW (population: critic rows that reached the ML gate, i.e.
`decision.ml_proba` is not None -- the bot's own candidates, TH-06)
  input drift  PSI of the market features the models read (ATR, daily range,
               volume ratio, RSI, ADX, slope, EMA distance, BTC trend, candle
               body), last 7 days vs the 30 before. < 0.10 stable, 0.10-0.25 watch,
               >= 0.25 shift. ALERT only when a shift persists 3 days in a row.
  quality      AUC of ml_proba for its own training target -- the peak label,
               resolved with ml_signal_model._peak_ahead (same horizon/threshold
               as config) -- per day, pooled over 14 days with a day bootstrap.
               ALERT when the interval lies below 0.5 (ranks worse than chance);
               "watch" when it contains 0.5. Reported beside it, never alerted on:
               the Spearman IC with the forward 10-bar return and the AUC for
               "coin ends the day in the immutable global top-20" with its base
               rate (TH-01; partly visible at decision time, TH-02).
  manifests    one line per nightly retrain: model hash before/after, git HEAD
               and dirty-file count, config.py hash, dataset size, trainer summary.

Reporting only: no trading decision reads this module. Flag MODEL_HEALTH_ENABLED.
CLI:  python model_health.py                         -> today's report (logged)
      python model_health.py --backfill 2026-05-20   -> one report per day, not logged
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pipeline_lib as PL

HERE = Path(__file__).resolve().parent
CRITIC = HERE / "critic_dataset.jsonl"
OUT_DIR = PL.PIPELINE / "model_health"
HEALTH_LOG = OUT_DIR / "health.jsonl"
MANIFESTS = OUT_DIR / "run_manifests.jsonl"

INPUTS = ("atr_pct", "daily_range", "vol_x", "rsi", "adx", "slope", "close_vs_ema50", "btc_vs_ema50", "body_pct")
# Regime indicators are reported but never alert: "BTC above/below its EMA50" changes by nature, and in the
# 139-day backfill it alone held the alert on 99 days (model-health-1005-spec.md).
REGIME_INDICATORS = ("btc_vs_ema50",)
RECENT_DAYS, REFERENCE_DAYS, QUALITY_DAYS = 7, 30, 14
PSI_WATCH, PSI_SHIFT, PERSIST_DAYS = 0.10, 0.25, 3
MIN_ROWS = 200
MIN_DAY_ROWS = 40
BOOT = 2000


# ---------------------------------------------------------------- data

def load_rows(path: Path = CRITIC, since: date | None = None) -> list[dict]:
    """Compact rows of candidates that reached the ML gate (ml_proba logged)."""
    out = []
    cut = since.isoformat() if since else ""
    with io.open(path, "rb") as fh:
        for raw in fh:
            if b'"ml_proba"' not in raw:
                continue
            try:
                e = json.loads(raw)
            except Exception:
                continue
            d = e.get("decision") or {}
            if d.get("ml_proba") is None:
                continue
            ts = str(e.get("ts_signal") or "")
            day = ts[:10]
            if not day or day < cut:
                continue
            f = e.get("f") or {}
            lab = e.get("labels") or {}
            row = {"day": day, "ts": ts, "sym": e.get("sym"), "tf": e.get("tf") or "15m",
                   "ml_proba": d.get("ml_proba"), "ret_10": lab.get("ret_10")}
            for k in INPUTS:
                row[k] = f.get(k)
            out.append(row)
    return out


def attach_peaks(rows: list) -> int:
    """row['peak'] = the trainer's peak label value (% high within horizon bars), or None."""
    import config
    import ml_signal_model as MSM
    horizon = int(getattr(config, "ML_PEAK_LABEL_HORIZON", 5))
    try:
        cache = MSM._peak_cache()                       # read-only use of the trainer's cache
    except Exception:
        cache = {}
    n = 0
    for r in rows:
        try:
            ts = MSM._parse_ts(r["ts"])
        except Exception:
            r["peak"] = None
            continue
        key = "%s|%s|%s|%d" % (r["sym"], r["tf"], ts.isoformat(), horizon)
        p = cache.get(key)
        if p is None:
            p = MSM._peak_ahead(str(r["sym"]), str(r["tf"]), ts, horizon)
        r["peak"] = p
        n += p is not None
    return n


def winner_days() -> set:
    try:
        import immutable_labels as IL
        import _compute_early_capture as E
        win, _ = IL.winners_by_day(top_n=20, watchlist=E.load_watchlist(), rank_before_filter=True)
        return set(win)
    except Exception:
        return set()


# ---------------------------------------------------------------- statistics

def psi(reference: list, recent: list, bins: int = 10) -> float | None:
    """Population stability index with reference-quantile bins."""
    ref = sorted(x for x in reference if isinstance(x, (int, float)) and math.isfinite(x))
    rec = [x for x in recent if isinstance(x, (int, float)) and math.isfinite(x)]
    if len(ref) < MIN_ROWS or len(rec) < MIN_ROWS:
        return None
    edges = sorted({ref[min(len(ref) - 1, int(len(ref) * k / bins))] for k in range(1, bins)})

    def shares(xs):
        cnt = [0] * (len(edges) + 1)
        for x in xs:
            lo, hi = 0, len(edges)
            while lo < hi:                       # first edge > x
                mid = (lo + hi) // 2
                if edges[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            cnt[lo] += 1
        return [max(c / len(xs), 1e-4) for c in cnt]

    a, b = shares(ref), shares(rec)
    return round(sum((q - p) * math.log(q / p) for p, q in zip(a, b)), 4)


def _ranks(v: list) -> list:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x: list, y: list) -> float | None:
    if len(x) < 30:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def auc(scores: list, labels: list, min_class: int = 10) -> float | None:
    pos = sum(1 for l in labels if l)
    neg = len(labels) - pos
    if pos < min_class or neg < min_class:
        return None
    r = _ranks(scores)                                   # 0-based average ranks
    rank_sum_pos = sum(ri + 1 for ri, l in zip(r, labels) if l)
    return (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)   # Mann-Whitney U / (pos * neg)


def _boot(values: list, seed: int) -> tuple:
    rnd = random.Random(seed)
    b = sorted(sum(rnd.choices(values, k=len(values))) / len(values) for _ in range(BOOT))
    return sum(values) / len(values), b[int(0.025 * BOOT)], b[int(0.975 * BOOT) - 1]


# ---------------------------------------------------------------- report

def report_for(rows: list, today: date, winners: set, threshold_pct: float = 2.0) -> dict:
    """Health as known at the start of `today` (rows of earlier days only); alerts are set by finalize()."""
    d0 = today - timedelta(days=RECENT_DAYS)
    d1 = d0 - timedelta(days=REFERENCE_DAYS)
    q0 = today - timedelta(days=QUALITY_DAYS)
    t, s0, s1, sq = today.isoformat(), d0.isoformat(), d1.isoformat(), q0.isoformat()
    recent = [r for r in rows if s0 <= r["day"] < t]
    ref = [r for r in rows if s1 <= r["day"] < s0]
    drift = {}
    for f in INPUTS:
        v = psi([r[f] for r in ref], [r[f] for r in recent])
        drift[f] = None if v is None else v
    shifted = sorted(f for f, v in drift.items() if v is not None and v >= PSI_SHIFT)
    watched = sorted(f for f, v in drift.items() if v is not None and PSI_WATCH <= v < PSI_SHIFT)

    qrows = [r for r in rows if sq <= r["day"] < t]
    by_day = {}
    for r in qrows:
        by_day.setdefault(r["day"], []).append(r)
    daily = {}
    for d, rs in sorted(by_day.items()):
        lab = [(r, r["peak"] >= threshold_pct) for r in rs if isinstance(r.get("peak"), (int, float))]
        a = auc([r["ml_proba"] for r, _ in lab], [y for _, y in lab], min_class=5) if len(lab) >= MIN_DAY_ROWS else None
        ret = [r for r in rs if isinstance(r.get("ret_10"), (int, float))]
        ic = spearman([r["ml_proba"] for r in ret], [r["ret_10"] for r in ret])
        daily[d] = {"auc_own": None if a is None else round(a, 4), "ic_ret10": None if ic is None else round(ic, 4),
                    "n": len(rs), "n_peak": len(lab), "pos_rate": round(sum(y for _, y in lab) / len(lab), 4) if lab else None}
    aucs = [v["auc_own"] for v in daily.values() if v["auc_own"] is not None]
    ics = [v["ic_ret10"] for v in daily.values() if v["ic_ret10"] is not None]
    quality = {"target": f"peak >= {threshold_pct:g}% within ML_PEAK_LABEL_HORIZON bars (ml_signal_model's own label)",
               "days": len(aucs), "rows": len(qrows), "daily": daily}
    if len(aucs) >= 7:
        m, lo, hi = _boot(aucs, 5)
        quality.update(auc_mean=round(m, 4), auc_lo95=round(lo, 4), auc_hi95=round(hi, 4),
                       level="inverted" if hi < 0.5 else "no_signal" if lo <= 0.5 else "ok")
        pr = [v["pos_rate"] for v in daily.values() if v["pos_rate"] is not None]
        quality["target_base_rate"] = round(sum(pr) / len(pr), 4) if pr else None
    else:
        quality["level"] = "insufficient_data"
    if len(ics) >= 7:
        m, lo, hi = _boot(ics, 6)
        quality["ic_ret10"] = {"mean": round(m, 4), "lo95": round(lo, 4), "hi95": round(hi, 4),
                               "note": "reference only: ret_10 is not the model's target"}
    if winners:
        wr = [r for r in qrows if r.get("sym")]
        labels = [(r["day"], r["sym"]) in winners for r in wr]
        a = auc([r["ml_proba"] for r in wr], labels)
        quality["winner_day_auc"] = None if a is None else round(a, 4)
        quality["winner_day_base_rate"] = round(sum(labels) / len(labels), 4) if labels else None
    return {"day": t, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_drift": drift, "shifted": shifted, "watched": watched, "quality": quality,
            "logging_gaps": ["ranker_proba is not logged in critic_dataset (pipeline_drift has had 0 rows for it)"]}


def shift_streak(history: list, feature: str) -> int:
    n = 0
    for h in reversed(history):
        if feature in (h.get("shifted") or []):
            n += 1
        else:
            break
    return n


def finalize(rep: dict, history: list) -> dict:
    """Alerts need the previous days: an input shift alerts only when it persists PERSIST_DAYS days."""
    seq = history + [rep]
    feats = [f for f in rep["shifted"] if f not in REGIME_INDICATORS]
    rep["alerts"] = [f"input_drift:{f}" for f in feats if shift_streak(seq, f) >= PERSIST_DAYS]
    rep["watch"] = [f"input_drift:{f}" for f in feats if f"input_drift:{f}" not in rep["alerts"]]
    lvl = rep["quality"].get("level")
    if lvl == "inverted":
        rep["alerts"].append("quality:inverted")
    elif lvl == "no_signal":
        rep["watch"].append("quality:no_signal")
    return rep


# ---------------------------------------------------------------- run manifests

def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=20, cwd=str(HERE.parent))
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def record_retrain(name: str, model_path, before_sha: str | None, dataset_path=None, result: dict | None = None,
                   path: Path = MANIFESTS) -> dict:
    """Append one manifest line for a nightly retrain. Never raises."""
    try:
        after = sha256_file(model_path)
        status = _git("status", "--porcelain")
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": name,
               "model_path": str(Path(model_path).name), "sha_before": before_sha, "sha_after": after,
               "changed": bool(after and after != before_sha),
               "git_head": _git("rev-parse", "HEAD"),
               "git_dirty_files": None if status is None else len([l for l in status.splitlines() if l.strip()]),
               "config_sha256": sha256_file(HERE / "config.py"),
               "dataset": None, "result": result}
        if dataset_path and Path(dataset_path).exists():
            st = Path(dataset_path).stat()
            rec["dataset"] = {"path": Path(dataset_path).name, "bytes": st.st_size,
                              "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec
    except Exception as exc:                               # a manifest must never break training
        return {"model": name, "error": f"{type(exc).__name__}: {exc}"}


def last_manifests(path: Path = MANIFESTS, hours: int = 30) -> list:
    if not path.exists():
        return []
    cut = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get("ts", "")) >= cut:
            out.append(r)
    return out


# ---------------------------------------------------------------- rendering

def render_block(rep: dict, history: list | None = None, manifests: list | None = None) -> str | None:
    if not rep:
        return None
    seq = (history or []) + [rep]
    q = rep.get("quality") or {}
    lines = ["<b>Здоровье моделей</b> (ML-фильтр, кандидаты бота)"]
    if q.get("level") not in (None, "insufficient_data"):
        txt = {"ok": "работает", "no_signal": "сигнала не видно", "inverted": "⚠ ранжирует ХУЖЕ случайного"}[q["level"]]
        s = (f"ml_proba против своей цели (пик ≥2% за 5 баров), {q['days']} дн.: AUC {q['auc_mean']:.3f} "
             f"[{q['auc_lo95']:.3f}; {q['auc_hi95']:.3f}] — {txt}")
        if q.get("target_base_rate") is not None:
            s += f"; база {100 * q['target_base_rate']:.0f}%"
        lines.append(s)
        ic = q.get("ic_ret10")
        if ic:
            lines.append(f"Справочно: IC с доходностью 10 баров {ic['mean']:+.3f} [{ic['lo95']:+.3f}; {ic['hi95']:+.3f}] "
                         f"(это не цель модели)")
    else:
        lines.append(f"ml_proba против своей цели: мало данных ({q.get('days', 0)} дн.)")
    feats = [f for f in rep.get("shifted") or [] if f not in REGIME_INDICATORS]
    if feats:
        parts = [f"{f} {rep['input_drift'][f]:.2f} ({shift_streak(seq, f)} дн.)" for f in feats]
        lines.append("Сдвиг рынка во входах, PSI ≥ 0.25: " + ", ".join(parts)
                     + (" — ⚠ держится ≥3 дн., повод переобучить" if any(a.startswith("input_drift") for a in rep.get("alerts", [])) else ""))
    else:
        lines.append("Входы моделей: распределения стабильны (PSI < 0.25)")
    if manifests:
        ch = [m["model"] for m in manifests if m.get("changed")]
        same = [m["model"] for m in manifests if not m.get("changed") and not m.get("error")]
        lines.append("Ночное переобучение: " + (f"обновлены {', '.join(ch)}" if ch else "файлы моделей не изменились")
                     + (f"; без изменений {', '.join(same)}" if ch and same else ""))
    return "\n".join(lines)


def load_latest(path: Path = HEALTH_LOG) -> tuple[dict | None, list]:
    if not path.exists():
        return None, []
    hist = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            hist.append(json.loads(line))
        except Exception:
            continue
    return (hist[-1] if hist else None), hist[:-1]


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", help="first day YYYY-MM-DD: one report per day up to today, printed, not logged")
    ap.add_argument("--dataset", default=str(CRITIC))
    a = ap.parse_args(argv)
    import config
    if not getattr(config, "MODEL_HEALTH_ENABLED", True):
        print("[model_health] disabled by MODEL_HEALTH_ENABLED=False")
        return 0
    threshold = float(getattr(config, "ML_PEAK_LABEL_THRESHOLD_PCT", 2.0))
    today = datetime.now(timezone.utc).date()
    first = date.fromisoformat(a.backfill) if a.backfill else today
    rows = load_rows(Path(a.dataset), since=first - timedelta(days=RECENT_DAYS + REFERENCE_DAYS + 1))
    n_peak = attach_peaks(rows)
    winners = winner_days()
    print(f"[model_health] {len(rows)} ML-gate rows, {n_peak} with a resolved peak label, {len(winners)} winner-days",
          flush=True)
    if a.backfill:
        out = []
        d = first
        while d <= today:
            r = finalize(report_for(rows, d, winners, threshold), out)
            q = r["quality"]
            print(f"{r['day']}  AUCown={q.get('auc_mean')} [{q.get('auc_lo95')},{q.get('auc_hi95')}] {q.get('level')}  "
                  f"IC10={(q.get('ic_ret10') or {}).get('mean')}  AUCwin={q.get('winner_day_auc')}  "
                  f"shifted={r['shifted']}  alerts={r['alerts']}", flush=True)
            out.append(r)
            d += timedelta(days=1)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "backfill.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n",
                                                encoding="utf-8")
        n = len(out)
        al = sum(1 for x in out if x["alerts"])
        print(f"[model_health] backfill {n} days: alert on {al} ({100 * al / max(n, 1):.0f}%), "
              f"watch-only {sum(1 for x in out if x['watch'] and not x['alerts'])}")
        return 0
    latest, hist = load_latest()
    history = (hist + [latest]) if latest else []
    history = [h for h in history if h.get("day") != today.isoformat()]          # a rerun replaces today
    rep = finalize(report_for(rows, today, winners, threshold), history)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(HEALTH_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rep, ensure_ascii=False) + "\n")
    print(render_block(rep, history, last_manifests()) or "(no block)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
