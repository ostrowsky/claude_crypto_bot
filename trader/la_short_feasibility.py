"""H1 step 0 -- was a Binance USD-M perp trading when the perp-listing announcement came out? (la-short-feasibility-1105-spec.md)

No returns are computed: only whether a short instrument existed at t_pub, from the first 15m bar in
runtime/perp_history. Events are those of experiment_la.build_events on the train window.

  (from trader/) ../pyembed/python.exe -u la_short_feasibility.py run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

SPEC = "la-short-feasibility-1105-spec.md"
OUT = TC.RUNTIME / "experiments" / "LA_SHORT_FEASIBILITY.json"
BAR = 15 * 60 * 1000
HOUR = 3_600_000
PERP_TYPES = ("BN_PERP_NEW", "OKX_PERP_NEW", "BYBIT_PERP_NEW")
MAX_UNAVAILABLE = 0.30


def status(t_pub: int, perp_first: int | None):
    """perp_live: the perp's first 15m bar had opened by t_pub - 15 min, so it was trading when the news came out."""
    if perp_first is None:
        return "no_perp", None
    if perp_first <= t_pub - BAR:
        return "perp_live", None
    return "perp_later", (perp_first - t_pub) / HOUR


def summary(rows) -> dict:
    out = {}
    for et in sorted({r["etype"] for r in rows}):
        rs = [r for r in rows if r["etype"] == et]
        n = len(rs)
        c = {k: sum(1 for r in rs if r["status"] == k) for k in ("perp_live", "perp_later", "no_perp")}
        delays = [r["delay_h"] for r in rs if r["status"] == "perp_later" and r["delay_h"] is not None]
        un = (c["perp_later"] + c["no_perp"]) / n if n else None
        out[et] = {"n": n, **c, "unavailable_share": un,
                   "median_delay_h": float(np.median(delays)) if delays else None,
                   "delay_h_p25_p75": [float(np.percentile(delays, 25)), float(np.percentile(delays, 75))] if delays else None,
                   "out": bool(un is not None and un > MAX_UNAVAILABLE)}
    return out


def run() -> dict:
    import announcements as AN
    import experiment_la as LA
    import experiment_xsmom as XE
    import funding_store as FS
    st = AN.AnnouncementStore()
    records = AN.drop_migrated([r for s in ("binance", "okx", "bybit") for r in st.load(s)])
    panel = XE.load_panel()
    first_bar = {}
    for s, j in panel.col.items():
        rows = np.flatnonzero(panel.has_bar[:, j])
        if len(rows):
            first_bar[s] = (int(panel.d0) + int(rows[0])) * LA.DAY
    events = [e for e in LA.build_events(records, first_bar, LA._ms(LA.TRAIN_FROM), LA._ms(LA.TEST_FROM))
              if e["etype"] in PERP_TYPES]
    perp_first = {}
    out_rows = []
    for e in events:
        s = e["sym"]
        if s not in perp_first:
            p = FS.perp_path(s)
            perp_first[s] = None
            if p.exists():
                with np.load(p) as z:
                    if len(z["t"]):
                        perp_first[s] = int(z["t"][0])
        stt, d = status(e["t_pub"], perp_first[s])
        out_rows.append({"etype": e["etype"], "sym": s, "t_pub": e["t_pub"], "status": stt, "delay_h": d})
    res = {"spec": SPEC, "window": [LA.TRAIN_FROM, LA.TEST_FROM], "n_events": len(out_rows),
           "perp_store_files": len(list(FS.PERP_DIR.glob("*_15m.npz"))), "summary": summary(out_rows), "rows": out_rows}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res["summary"], indent=1))
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        run()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
