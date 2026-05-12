"""North-star: EarlyCapture@<label> = coverage * capture_ratio * time_lead_score

Two parallel north-stars (P1.1 2026-05-07):
  EarlyCapture@top20      — historic ground truth (label_top20 from dataset)
  EarlyCapture@sustained  — clean ground truth (label_sustained_uptrend from
                            dataset_v2, see sustained-uptrend-label-spec.md)

Per winner-day:
  coverage_flag    = 1 if entered, else 0
  capture_ratio    = clamp(realized_pnl / eod_return_pct, 0, 1)
  time_lead_score  = 1 - (entry_hour_UTC / 24)   (early UTC = higher)
EarlyCapture = mean(coverage * capture * time_lead) across all winner-days.
"""
from __future__ import annotations
import argparse, json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
NOW = datetime.now(timezone.utc)


def load_winners(dataset_path: Path, label_field: str, cut_dt: datetime):
    """Returns (winners_set, eod_ret_dict)."""
    winners = set()
    eod_ret = {}
    with io.open(dataset_path, encoding="utf-8") as f:
        for ln in f:
            try: e = json.loads(ln)
            except: continue
            ts_ms = e.get("ts");
            if not ts_ms: continue
            dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
            if dt < cut_dt: continue
            sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
            if e.get(label_field) == 1:
                winners.add((d, sym))
            eod_ret[(d, sym)] = e.get("eod_return_pct")
    return winners, eod_ret


def load_entries(cut_dt: datetime):
    first_entry = {}; pnl_pairs = {}; entries = {}
    with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
        for ln in f:
            if '"event"' not in ln: continue
            try: e = json.loads(ln)
            except: continue
            ev = e.get("event","")
            if ev not in ("entry","exit"): continue
            ts = e.get("ts","")
            try: dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
            except: continue
            if dt < cut_dt: continue
            sym = e.get("sym") or e.get("symbol") or ""
            if not sym: continue
            d = dt.strftime("%Y-%m-%d")
            if ev == "entry":
                ep = float(e.get("price") or e.get("entry_price") or 0)
                entries[sym] = (dt, d, ep)
                prev = first_entry.get((d, sym))
                if prev is None or dt < prev[0]:
                    first_entry[(d, sym)] = (dt, ep)
            else:
                ent = entries.pop(sym, None)
                if not ent: continue
                ex_p = float(e.get("exit_price") or e.get("price") or 0)
                if ent[2] <= 0 or ex_p <= 0: continue
                pnl = (ex_p - ent[2]) / ent[2] * 100
                pnl_pairs[(ent[1], sym)] = pnl
    return first_entry, pnl_pairs


def compute_north_star(winners, eod_ret, first_entry, pnl_pairs, label_name: str):
    ec_scores = []; breakdown = []
    for key in winners:
        d, sym = key
        ent = first_entry.get(key)
        coverage = 1.0 if ent else 0.0
        if ent:
            edt, ep = ent
            time_lead = 1.0 - (edt.hour / 24.0)
            pnl = pnl_pairs.get(key)
            eod = eod_ret.get(key)
            if pnl is not None and eod is not None:
                eod_p = float(eod)
                if abs(eod_p) <= 5: eod_p *= 100
                if abs(eod_p) >= 1.0:
                    cap = max(0.0, min(1.0, pnl / eod_p))
                else:
                    cap = 0.0
            else:
                cap = 0.0
        else:
            time_lead = 0.0; cap = 0.0
        score = coverage * cap * time_lead
        ec_scores.append(score)
        breakdown.append({"d": d, "sym": sym, "cov": coverage, "cap": cap,
                          "lead": time_lead, "score": score})
    n = len(ec_scores)
    mean_ec = sum(ec_scores)/max(1, n)
    mean_cov = sum(b["cov"] for b in breakdown)/max(1,n)
    entered = [b for b in breakdown if b["cov"] > 0]
    mean_cap = sum(b["cap"] for b in entered) / max(1, len(entered))
    mean_lead = sum(b["lead"] for b in entered) / max(1, len(entered))
    return {
        "label": label_name, "n": n,
        "early_capture": mean_ec,
        "decomp_coverage": mean_cov,
        "decomp_capture_mean": mean_cap,
        "decomp_time_lead_mean": mean_lead,
        "_breakdown": breakdown,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    cut_dt = NOW - timedelta(days=args.days)

    first_entry, pnl_pairs = load_entries(cut_dt)

    # Top-20 (existing)
    top20, eod_ret = load_winners(ROOT/"files"/"top_gainer_dataset.jsonl",
                                  label_field="label_top20", cut_dt=cut_dt)
    res_top20 = compute_north_star(top20, eod_ret, first_entry, pnl_pairs, "top20")

    # Sustained (P1.1 — try v2 dataset, fall back if absent)
    res_sustained = None
    v2_path = ROOT/"files"/"top_gainer_dataset_v2.jsonl"
    if v2_path.exists():
        sustained, eod_ret_s = load_winners(v2_path,
                                            label_field="label_sustained_uptrend",
                                            cut_dt=cut_dt)
        res_sustained = compute_north_star(sustained, eod_ret_s, first_entry,
                                           pnl_pairs, "sustained")

    # Output
    print(f"=== NORTH-STAR · last {args.days}d ===\n")
    for r in [res_top20, res_sustained]:
        if r is None: continue
        print(f"EarlyCapture@{r['label']:<10}  {r['early_capture']:.3f}  "
              f"(n={r['n']}, cov={r['decomp_coverage']:.2f}, "
              f"cap={r['decomp_capture_mean']:.2f}, "
              f"lead={r['decomp_time_lead_mean']:.2f})")
    if res_sustained is None:
        print("\nEarlyCapture@sustained: dataset_v2 not found — run "
              "files/_backfill_sustained_uptrend.py first")

    # Top-5 winners by score (top20 — existing behaviour)
    bd = sorted(res_top20["_breakdown"], key=lambda x: -x["score"])
    print(f"\nTop-5 top-20 winners (highest EC):")
    for b in bd[:5]:
        print(f"  {b['d']} {b['sym']:<10} score={b['score']:.3f}  "
              f"(cov={b['cov']:.0f}, cap={b['cap']:.2f}, lead={b['lead']:.2f})")

    # METRIC_JSON for daily aggregator (keep top-20 as primary)
    metric = {
        "metric": "NS_EarlyCapture_top20",
        "n": res_top20["n"],
        "early_capture": res_top20["early_capture"],
        "decomp_coverage": res_top20["decomp_coverage"],
        "decomp_capture_mean": res_top20["decomp_capture_mean"],
        "decomp_time_lead_mean": res_top20["decomp_time_lead_mean"],
    }
    if res_sustained:
        metric["sustained_n"] = res_sustained["n"]
        metric["sustained_early_capture"] = res_sustained["early_capture"]
        metric["sustained_coverage"] = res_sustained["decomp_coverage"]
        metric["sustained_capture_mean"] = res_sustained["decomp_capture_mean"]
    print("\nMETRIC_JSON:" + json.dumps(metric))


if __name__ == "__main__":
    main()
