"""The ML gate's GOAL-label model, in SHADOW (ml-goal-shadow-1012-spec.md).

The live gate is trained on `peak5_2.0` -- "+2% within the next five bars" -- which
is a volatility question: on 2026-10-11 it scored ICP 0.023 at the start of a
+13% day-long climb. Trained instead on the goal itself (is this candidate an
early row of a top-20 winner-day?), the same features and estimator reached
+7.05 pp more winner-days before the crossing in a walk-forward backtest, with
trades no worse when counted as the bot takes them -- on data already looked at,
and without the gates downstream of ML. A shadow is the confirmation: this
module trains that model nightly and scores live candidates; monitor logs the
score beside the live one; readout ML-GOAL-SHADOW judges it on unseen days.

No decision of the bot reads this score. Rollback: ML_GOAL_SHADOW_ENABLED=False.
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402

LABEL_VERSION = "goal_early_top20_v1"
TF_LEN = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1)}
MIN_AGE = timedelta(days=1)


def model_path() -> Path:
    return HERE / str(getattr(config, "ML_GOAL_SHADOW_MODEL_FILE", "ml_goal_shadow_model.json"))


# ---------------------------------------------------------------- label

def early_winner(sym: str, dec: datetime, win, dl) -> bool:
    """The decision lies in [day open, first +2.5% crossing) of an immutable top-20 winner-day."""
    k = (dec.strftime("%Y-%m-%d"), sym)
    return k in win and k in dl and dl[k][1] is not None and dl[k][0] <= dec < dl[k][1]


def resolved_days(eod, dl) -> set:
    """Days present in both the EOD returns and the crossing deadlines. On any other
    day every candidate would read as a negative, so it does not train."""
    return {d for d, _ in eod} & {d for d, _ in dl}


def eligible(dec: datetime, resolved: set, now: datetime) -> bool:
    return dec.strftime("%Y-%m-%d") in resolved and dec < now - MIN_AGE


# ---------------------------------------------------------------- training

def load_candidates(path: Path, names, M):
    """Bot candidates deduplicated per (sym, tf, ts_signal) -> [(sym, decision_dt)], X."""
    import numpy as np
    seen, meta, X = set(), [], []
    with io.open(path, "rb") as fh:
        for raw in fh:
            try:
                r = json.loads(raw)
            except Exception:
                continue
            tf = str(r.get("tf") or "")
            if tf not in TF_LEN or not r.get("sym") or not r.get("ts_signal"):
                continue
            key = (r["sym"], tf, r["ts_signal"])
            if key in seen:
                continue
            seen.add(key)
            try:
                ts = datetime.fromisoformat(str(r["ts_signal"]).replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            fmap = M.build_feature_dict(r)
            X.append([M._safe_float(fmap.get(n)) for n in names])
            meta.append((r["sym"], ts + TF_LEN[tf]))
    return meta, np.asarray(X, dtype=float)


def fit_payload(X, y, names, meta: Optional[dict] = None) -> dict:
    """Scaler + the live CatBoost family, in the live payload format."""
    import ml_signal_model as M
    sc = M.StandardScaler().fit(X)
    model = M.CatBoostModel(X.shape[1]).fit(sc.transform(X), y)
    return {"model_name": "catboost", "label_version": LABEL_VERSION, "feature_names": list(names),
            "scaler_mean": [float(v) for v in sc.mean_], "scaler_scale": [float(v) for v in sc.scale_],
            "model": model.to_dict(), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(meta or {})}


def save_payload(payload: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def train(now: Optional[datetime] = None, out: Optional[Path] = None) -> dict:
    import numpy as np
    import immutable_labels as IL
    import label_store as LS
    import ml_signal_model as M
    import _compute_early_capture as E
    now = now or datetime.now(timezone.utc)
    names = M.safe_feature_names()
    meta, X = load_candidates(HERE / "critic_dataset.jsonl", names, M)
    win, eod = IL.winners_by_day(top_n=20, watchlist=E.load_watchlist(), rank_before_filter=True)
    dl = LS.intraday_deadlines()
    res = resolved_days(eod, dl)
    idx = [k for k, (_, dec) in enumerate(meta) if eligible(dec, res, now)]
    if len(idx) < 2000:
        raise RuntimeError("only %d eligible rows" % len(idx))
    y = np.array([1.0 if early_winner(meta[k][0], meta[k][1], win, dl) else 0.0 for k in idx])
    days = sorted({meta[k][1].strftime("%Y-%m-%d") for k in idx})
    info = {"n_rows": len(idx), "n_pos": int(y.sum()), "first_day": days[0], "last_day": days[-1]}
    payload = fit_payload(X[idx], y, names, meta=info)
    save_payload(payload, out or model_path())
    return {"candidates": len(meta), **info, "base_rate": round(float(y.mean()), 4),
            "trained_at": payload["trained_at"]}


# ---------------------------------------------------------------- live scoring

_CACHE: Optional[dict] = None
_MTIME: Optional[float] = None
_CHECKED_AT: float = 0.0
_PATH: Optional[Path] = None


def _reset() -> None:
    global _CACHE, _MTIME, _CHECKED_AT, _PATH
    _CACHE, _MTIME, _CHECKED_AT, _PATH = None, None, 0.0, None


def _load(path: Path) -> Optional[dict]:
    """The shadow payload, re-read when the file's mtime changes (checked at most every
    ML_GOAL_SHADOW_RELOAD_CHECK_SEC). Anything unreadable keeps what is loaded."""
    global _CACHE, _MTIME, _CHECKED_AT, _PATH
    now = time.monotonic()
    if (_CACHE is not None and _PATH == path
            and now - _CHECKED_AT < float(getattr(config, "ML_GOAL_SHADOW_RELOAD_CHECK_SEC", 60))):
        return _CACHE
    _CHECKED_AT = now
    try:
        mt = path.stat().st_mtime
    except OSError:
        return _CACHE if _PATH == path else None
    if _CACHE is not None and _PATH == path and mt == _MTIME:
        return _CACHE
    try:
        p = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _CACHE if _PATH == path else None
    if isinstance(p, dict) and "feature_names" in p and "model" in p:
        _CACHE, _MTIME, _PATH = p, mt, path
    return _CACHE if _PATH == path else None


def model_trained_at() -> Optional[str]:
    return (_CACHE or {}).get("trained_at")


def score(*, sym, tf, signal_type, is_bull_day, feat, data, i, btc_vs_ema50, path: Optional[Path] = None):
    """P(early row of a top-20 winner-day) for a live candidate; None = no score (fails open)."""
    if not getattr(config, "ML_GOAL_SHADOW_ENABLED", False):
        return None
    try:
        payload = _load(path or model_path())
        if not payload:
            return None
        import ml_signal_model as M
        rec = M.build_runtime_record(sym=sym, tf=tf, signal_type=signal_type, is_bull_day=is_bull_day,
                                     bar_ts=int(data["t"][i]), feat=feat, data=data, i=i,
                                     btc_vs_ema50=float(btc_vs_ema50))
        return float(M.predict_proba_from_payload(payload, rec))
    except Exception:
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not getattr(config, "ML_GOAL_SHADOW_ENABLED", False):
        print("[ml_goal_shadow] disabled (ML_GOAL_SHADOW_ENABLED=False)")
        return 0
    print("[ml_goal_shadow] " + json.dumps(train()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
