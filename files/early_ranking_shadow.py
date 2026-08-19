"""Name the day's likely winners at 00 UTC, and record it. Shadow only.

Goal 1 (spot the winners early) directly; goal 2 (signal entry early) only if
this evidence holds, because nothing here emits an alert.

The seven live entry modes all wait for ADX / slope / volume / breakout, so by
construction none can fire before the move — measured, move-relative lead is
0.02 and early alerts carry lift 0.72x, worse than chance. This path does not
wait, and writes down what it would have named so the claim can be checked
before anything fires.

Isolation is the point: this module imports nothing that decides. A test asserts
it, because a shadow that can reach a gate is not a shadow.

    pyembed\\python.exe files\\early_ranking_shadow.py            # write today
    pyembed\\python.exe files\\early_ranking_shadow.py --score    # grade history

Spec: docs/specs/features/early-ranking-shadow-spec.md
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SHADOW_LOG = ROOT / ".runtime" / "early_ranking_shadow.jsonl"
SNAPSHOT_HOUR = 0
DEFAULT_K = 10
MIN_DAYS_TO_JUDGE = 20
DRAWS = 200


def build_list(scored: list, k: int) -> list[dict] | None:
    """Top-k by probability, descending.

    Returns **None** for an empty universe rather than an empty list: an empty
    list would later read as "the model named nobody", a claim about the model
    when the truth is that the snapshot was missing (TH-05).
    """
    if not scored:
        return None
    ranked = sorted(scored, key=lambda t: -float(t[1]))[:k]
    return [{"symbol": s, "proba": round(float(p), 6)} for s, p in ranked]


def latest_snapshot(watchlist: set) -> tuple[str | None, list]:
    """(utc_day, [(symbol, features)]) for the most recent 00 UTC snapshot."""
    from top_gainer_model import FEATURE_NAMES
    by_day: dict[str, dict] = defaultdict(dict)
    path = HERE / "top_gainer_dataset.jsonl"
    if not path.exists():
        return None, []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = e.get("symbol")
            if sym not in watchlist:
                continue
            ts = e.get("ts") or 0
            dt = datetime.fromtimestamp(ts / 1000 if ts > 1e11 else ts, timezone.utc)
            if dt.hour != SNAPSHOT_HOUR:
                continue
            feat = e.get("features") or {}
            # Keyed by symbol, LAST wins: the dataset carries two snapshots in
            # hour 00 (the EOD job and the intraday one), so appending blindly
            # put every coin in twice and a "top-10" was really a top-5.
            by_day[dt.strftime("%Y-%m-%d")][sym] = [
                float(feat.get(f, 0.0) or 0.0) for f in FEATURE_NAMES]
    if not by_day:
        return None, []
    day = max(by_day)
    return day, sorted(by_day[day].items())


def write_today(k: int = DEFAULT_K) -> dict[str, Any]:
    watchlist = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
    day, rows = latest_snapshot(watchlist)
    if not rows:
        return {"written": False, "reason": "no 00 UTC snapshot"}

    from top_gainer_model import FEATURE_NAMES, TopGainerModel
    # TopGainerModel() with no path never calls load(), so `_model_payload`
    # stays None and every prediction silently comes from the HEURISTIC
    # fallback. The first live list was built that way — by the fallback, not by
    # the model whose early-hour AUC is the entire reason this path exists.
    model_file = ROOT / "files" / "top_gainer_model.json"
    model = TopGainerModel(model_path=str(model_file))
    blob = getattr(model, "_model_payload", None) or {}
    if not blob.get("tier_models"):
        return {"written": False,
                "reason": f"model not loaded from {model_file.name}; refusing to "
                          f"write a list the heuristic produced"}
    scored = []
    for sym, x in rows:
        feats = dict(zip(FEATURE_NAMES, x))
        feats["symbol"] = sym
        scored.append((sym, model.predict(feats).prob_top20))

    picks = build_list(scored, k)
    if picks is None:
        return {"written": False, "reason": "empty universe"}

    record = {
        "utc_day": day,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k,
        "universe": len(rows),
        "picks": picks,
        # Provenance travels with the list: a list graded against a model that
        # later trained on the same days would flatter itself.
        "model_evaluation_scope": blob.get("evaluation_scope"),
        "model_label_timing": blob.get("label_timing"),
    }
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SHADOW_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"written": True, "utc_day": day, "k": k, "universe": len(rows)}


def score(lists: list, *, winners: set, label_days: set) -> dict[str, Any]:
    """Grade past shadow lists against the immutable labels."""
    caught = picks = available = 0
    scored_days = 0
    without_labels = 0
    per_day = []
    for rec in lists:
        day = rec.get("utc_day")
        if day not in label_days:
            without_labels += 1
            continue
        scored_days += 1
        named = {p["symbol"] for p in rec.get("picks", [])}
        day_winners = {s for d, s in winners if d == day}
        caught += len(named & day_winners)
        picks += len(named)
        available += len(day_winners)
        per_day.append((len(day_winners), len(named), rec.get("universe") or 0))

    out = {
        "days_scored": scored_days,
        "days_without_labels": without_labels,
        "n_picks": picks,
        "n_winners_available": available,
        "winners_caught": caught,
        "coverage_pct": round(100.0 * caught / available, 2) if available else None,
        "precision_pct": round(100.0 * caught / picks, 2) if picks else None,
    }
    if scored_days < MIN_DAYS_TO_JUDGE:
        out["verdict"] = "too early to judge"
        out["reason"] = ("%d scored days < %d; one good day moves this by "
                         "several points" % (scored_days, MIN_DAYS_TO_JUDGE))
        return out

    # What a list of the same size catches by chance, sampled per day so the
    # band reflects the actual universe sizes rather than a pooled average.
    #
    # A day whose `universe` is unknown CANNOT contribute: treating it as zero
    # gives that day a zero chance of a random hit, which drags the band toward
    # [0, 0] and makes any coverage at all look "above control". A flattering
    # control is worse than no control.
    usable = [(w, n, u) for w, n, u in per_day if u and u > 0]
    out["days_without_universe"] = len(per_day) - len(usable)
    if not usable:
        out["control_band"] = None
        out["verdict"] = "control unavailable"
        out["reason"] = "no scored day records its universe size"
        return out

    avail_usable = sum(w for w, _, _ in usable)
    draws = []
    for seed in range(DRAWS):
        rng = random.Random(seed)
        got = 0
        for n_win, n_named, universe in usable:
            p = min(1.0, n_named / universe)
            got += sum(1 for _ in range(n_win) if rng.random() < p)
        draws.append(100.0 * got / avail_usable if avail_usable else 0.0)
    draws.sort()
    out["control_band"] = (round(draws[int(0.025 * DRAWS)], 2),
                           round(draws[int(0.975 * DRAWS) - 1], 2))
    out["verdict"] = ("above control"
                      if (out["coverage_pct"] or 0) > out["control_band"][1]
                      else "inside control")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args(argv)

    if not args.score:
        res = write_today(args.k)
        print(json.dumps(res, ensure_ascii=False))
        return 0 if res.get("written") else 1

    if not SHADOW_LOG.exists():
        print("no shadow log yet")
        return 1
    with io.open(SHADOW_LOG, encoding="utf-8") as fh:
        lists = [json.loads(l) for l in fh if l.strip()]
    import immutable_labels as IL
    watchlist = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
    winners, _ = IL.winners_by_day(top_n=20, watchlist=watchlist,
                                   rank_before_filter=True)
    res = score(lists, winners=winners, label_days={d for d, _ in winners})
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
