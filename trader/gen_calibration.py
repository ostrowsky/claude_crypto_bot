"""Phase B2 calibration -- does the generator reproduce the bot's real entries? (wide-universe-b-1014-spec.md §3)

recall = share of the bot's real entries (bot_events "entry") for which the generator has a candidate of the
same symbol and timeframe within ±tol bars; width = share of candidates the bot actually took. Recall below
RECALL_MIN stops the money test (the generator does not reproduce the bot's rules).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

BAR_MIN = {"15m": 15, "1h": 60}
RECALL_MIN = 0.79   # operator decision 2026-10-14 (was 0.80), wide-universe-b-1014-spec.md §8


def _dt(v):
    d = v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _bar_index(ts, tf):
    """Index of the bar whose close is at or before ts (the bot logs an entry after the close of bar i)."""
    m = BAR_MIN.get(tf, 15)
    return int(_dt(ts).timestamp() // (m * 60))


def match(bot_entries, cands, tol_bars: int = 1) -> dict:
    by = {}
    for c in cands:
        by.setdefault((c["sym"], c["tf"]), set()).add(_bar_index(c["ts"], c["tf"]))
    matched, taken = 0, set()
    for b in bot_entries:
        key = (b["sym"], b["tf"])
        k = _bar_index(b["ts"], b["tf"])
        idx = by.get(key, set())
        hit = [j for j in range(k - tol_bars, k + tol_bars + 1) if j in idx]
        if hit:
            matched += 1
            for j in hit:
                taken.add((key, j))
    n_c = sum(len(v) for v in by.values())
    n_b = len(bot_entries)
    return {"n_bot": n_b, "n_matched": matched, "recall": matched / n_b if n_b else None,
            "n_cands": n_c, "n_taken": len(taken), "width": len(taken) / n_c if n_c else None, "tol_bars": tol_bars}


def ok(rep) -> bool:
    return rep.get("recall") is not None and rep["recall"] >= RECALL_MIN


# ------------------------------------------------------------------ run (watchlist, the bot's live period)
def bot_entries(path, since: str, until: str, syms) -> list:
    """The bot's real entries (event 'entry') of `syms` with since <= day <= until; broken lines are skipped."""
    out = []
    with open(path, "rb") as fh:
        for raw in fh:
            if b'"entry"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") != "entry" or e.get("sym") not in syms:
                continue
            d = str(e.get("ts", ""))[:10]
            if since <= d <= until:
                out.append({"sym": e["sym"], "tf": e.get("tf"), "ts": e["ts"], "mode": e.get("mode") or e.get("signal_mode")})
    return out


def _gen_one(args):
    sym, window_from_ms, start_ms, end_ms = args
    import history_store as HS
    import candidate_gen as CG
    a = HS.load(HS.path_of(sym))
    if a is None or not len(a["t"]):
        return sym, []
    m = a["t"] >= window_from_ms
    a = {k: v[m] for k, v in a.items()}
    out = CG.generate(sym, "15m", a, start_ms=start_ms, end_ms=end_ms)
    out += CG.generate(sym, "1h", HS.agg_1h(a, complete_only=True), start_ms=start_ms, end_ms=end_ms)
    return sym, out


def run(since="2026-03-03", until="2026-10-13", window_from="2025-12-15", processes=3) -> dict:
    import time
    from multiprocessing import Pool
    import history_store as HS
    import trader_config as TC
    ms = lambda s: int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)  # noqa: E731
    wl = json.loads((TC.BOT_FILES / "watchlist.json").read_text(encoding="utf-8"))
    syms = [s for s in wl if HS.path_of(s).exists()]
    t0 = time.time()
    jobs = [(s, ms(window_from), ms(since), ms(until) + 86_400_000) for s in syms]
    cands = []
    with Pool(processes) as pool:
        for k, (sym, ev) in enumerate(pool.imap_unordered(_gen_one, jobs), 1):
            cands += ev
            if k % 10 == 0:
                print("generated %d/%d symbols, %d candidates, %.0f s" % (k, len(syms), len(cands), time.time() - t0), flush=True)
    bot = bot_entries(TC.BOT_EVENTS, since, until, set(syms))
    rep = {"since": since, "until": until, "window_from": window_from, "symbols": len(syms), "watchlist": len(wl),
           "overall": match(bot, cands)}
    for tf in ("15m", "1h"):
        rep["tf_" + tf] = match([b for b in bot if b["tf"] == tf], [c for c in cands if c["tf"] == tf])
    modes = sorted({b.get("mode") for b in bot if b.get("mode")})
    rep["by_bot_mode"] = {m: match([b for b in bot if b.get("mode") == m], cands) for m in modes}
    rep["cands_by_mode"] = {m: sum(1 for c in cands if c["mode"] == m) for m in sorted({c["mode"] for c in cands})}
    rep["gate_ok"] = ok(rep["overall"])
    rep["seconds"] = round(time.time() - t0, 1)
    rep["not_reproduced"] = ["ML gate, ranker, bandit, guards, rotation, cooldowns of the bot",
                             "catch-up entry on a recently discovered coin (live state)",
                             "impulse_speed -> trend while the mode is curtailed (state by day)"]
    out = TC.RUNTIME / "experiments" / "B2_CALIBRATION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    r = run()
    print(json.dumps({k: r[k] for k in ("symbols", "overall", "tf_15m", "tf_1h", "gate_ok", "seconds")}, ensure_ascii=False, indent=1))
