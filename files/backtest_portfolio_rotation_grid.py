"""
backtest_portfolio_rotation_grid.py
===================================

Grid-sweep по двум параметрам (ml_proba_min, score_min) для
определения Парето-оптимальной комбинации.

Запуск:
    python backtest_portfolio_rotation_grid.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


DATASET = Path(__file__).parent / "critic_dataset.jsonl"


def load_rows():
    port = []
    cd = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            d = r.get("decision", {}) or {}
            if d.get("action") != "blocked":
                continue
            L = r.get("labels", {}) or {}
            item = {
                "sym": r.get("sym", ""),
                "tf": r.get("tf", ""),
                "stype": r.get("signal_type", "") or "",
                "score": float(d.get("candidate_score") or 0.0),
                "mlp": (float(d["ml_proba"]) if d.get("ml_proba") is not None else None),
                "r5": (float(L["ret_5"]) if L.get("ret_5") is not None else None),
                "r10": (float(L["ret_10"]) if L.get("ret_10") is not None else None),
            }
            rc = d.get("reason_code")
            rt = d.get("reason", "") or ""
            if rc == "portfolio" and "portfel" not in rt and "портфель полон" in rt:
                port.append(item)
            elif rc == "cooldown":
                cd.append(item)
    return port, cd


def block_stats(rows, label):
    r5 = [r["r5"] for r in rows if r["r5"] is not None]
    if len(r5) < 10:
        return None
    r10 = [r["r10"] for r in rows if r["r10"] is not None]
    wins5 = sum(1 for v in r5 if v > 0)
    avg = sum(r5) / len(r5)
    sd = statistics.pstdev(r5) if len(r5) > 1 else 1.0
    sharpe = (avg / sd * math.sqrt(len(r5))) if sd > 0 else 0.0
    return {
        "label": label,
        "n": len(r5),
        "avg_r5": avg,
        "med_r5": statistics.median(r5),
        "win5": wins5 / len(r5),
        "sd_r5": sd,
        "sharpe": sharpe,
        "sum_r5": sum(r5),
        "avg_r10": (sum(r10) / len(r10)) if r10 else 0.0,
        "win10": (sum(1 for v in r10 if v > 0) / len(r10)) if r10 else 0.0,
    }


def main():
    port, cd = load_rows()
    print(f"Loaded port-blocks={len(port)}  cooldown-blocks={len(cd)}")

    print("\n[A] Baseline: take NOTHING -> 0 PnL (by definition)")
    print("[B] Naive: take ALL portfolio-blocked -> reference")
    base = block_stats(port, "ALL")
    print(f"    n={base['n']}  avg_r5={base['avg_r5']:+.3f}%  win={base['win5']*100:.1f}%  "
          f"sumPnL5={base['sum_r5']:+.1f}%  Sharpe={base['sharpe']:+.2f}")

    # ── Grid sweep ──
    mlp_grid = [None, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    score_grid = [0, 60, 70, 80, 90, 100, 110]

    print("\n[GRID] avg_r5 (n) -- by (ml_proba_min x score_min):")
    print(f"{'score\\mlp':>10s} " + " ".join(f"{('none' if m is None else f'{m:.2f}'):>10s}" for m in mlp_grid))
    grid_avg = {}
    grid_n = {}
    for sc in score_grid:
        row_parts = [f"{sc:>10d}"]
        for mlp in mlp_grid:
            subset = [r for r in port
                      if r["score"] >= sc
                      and (mlp is None or (r["mlp"] is not None and r["mlp"] >= mlp))]
            st = block_stats(subset, f"sc>={sc}, mlp>={mlp}")
            if st is None:
                row_parts.append(f"{'—':>10s}")
                grid_avg[(sc, mlp)] = None
                grid_n[(sc, mlp)] = len(subset)
            else:
                grid_avg[(sc, mlp)] = st["avg_r5"]
                grid_n[(sc, mlp)] = st["n"]
                row_parts.append(f"{st['avg_r5']:+.2f}({st['n']})".rjust(10))
        print(" ".join(row_parts))

    print("\n[GRID] win_rate_5 (%) -- by (ml_proba_min x score_min):")
    print(f"{'score\\mlp':>10s} " + " ".join(f"{('none' if m is None else f'{m:.2f}'):>10s}" for m in mlp_grid))
    for sc in score_grid:
        row_parts = [f"{sc:>10d}"]
        for mlp in mlp_grid:
            subset = [r for r in port
                      if r["score"] >= sc
                      and (mlp is None or (r["mlp"] is not None and r["mlp"] >= mlp))]
            st = block_stats(subset, f"sc>={sc}, mlp>={mlp}")
            if st is None:
                row_parts.append(f"{'—':>10s}")
            else:
                row_parts.append(f"{st['win5']*100:>4.1f}%({st['n']})".rjust(10))
        print(" ".join(row_parts))

    print("\n[GRID] Sharpe-proxy -- by (ml_proba_min x score_min):")
    print(f"{'score\\mlp':>10s} " + " ".join(f"{('none' if m is None else f'{m:.2f}'):>10s}" for m in mlp_grid))
    best = {"sharpe": -1e9}
    for sc in score_grid:
        row_parts = [f"{sc:>10d}"]
        for mlp in mlp_grid:
            subset = [r for r in port
                      if r["score"] >= sc
                      and (mlp is None or (r["mlp"] is not None and r["mlp"] >= mlp))]
            st = block_stats(subset, f"sc>={sc}, mlp>={mlp}")
            if st is None:
                row_parts.append(f"{'—':>10s}")
            else:
                row_parts.append(f"{st['sharpe']:+.2f}({st['n']})".rjust(10))
                if st["sharpe"] > best["sharpe"] and st["n"] >= 40:
                    best = {**st, "sc": sc, "mlp": mlp}
        print(" ".join(row_parts))

    # ── Pareto-optimal picks ──
    print("\n[PARETO] Best avg_r5/n combinations:")
    candidates = []
    for sc in score_grid:
        for mlp in mlp_grid:
            subset = [r for r in port
                      if r["score"] >= sc
                      and (mlp is None or (r["mlp"] is not None and r["mlp"] >= mlp))]
            st = block_stats(subset, f"sc>={sc}, mlp>={mlp}")
            if st is None:
                continue
            if st["n"] < 40:
                continue
            candidates.append({"sc": sc, "mlp": mlp, **st})

    # top-5 by sharpe, and top-5 by avg_r5
    print("\n  TOP-5 by Sharpe (n>=40):")
    for c in sorted(candidates, key=lambda x: x["sharpe"], reverse=True)[:5]:
        print(f"    score>={c['sc']}, mlp>={c['mlp']}:  "
              f"n={c['n']:>3d}  avg_r5={c['avg_r5']:+.3f}%  "
              f"win5={c['win5']*100:>4.1f}%  Sharpe={c['sharpe']:+.2f}  "
              f"sumPnL5={c['sum_r5']:+.1f}%")

    print("\n  TOP-5 by avg_r5 (n>=40):")
    for c in sorted(candidates, key=lambda x: x["avg_r5"], reverse=True)[:5]:
        print(f"    score>={c['sc']}, mlp>={c['mlp']}:  "
              f"n={c['n']:>3d}  avg_r5={c['avg_r5']:+.3f}%  "
              f"win5={c['win5']*100:>4.1f}%  Sharpe={c['sharpe']:+.2f}  "
              f"sumPnL5={c['sum_r5']:+.1f}%")

    print("\n  TOP-5 by sumPnL5 (n>=40):")
    for c in sorted(candidates, key=lambda x: x["sum_r5"], reverse=True)[:5]:
        print(f"    score>={c['sc']}, mlp>={c['mlp']}:  "
              f"n={c['n']:>3d}  avg_r5={c['avg_r5']:+.3f}%  "
              f"win5={c['win5']*100:>4.1f}%  Sharpe={c['sharpe']:+.2f}  "
              f"sumPnL5={c['sum_r5']:+.1f}%")

    # ── Combined with COOLDOWN ──
    print("\n[COOLDOWN] только cooldown-blocked:")
    cd_st = block_stats(cd, "cooldown")
    if cd_st:
        print(f"  n={cd_st['n']}  avg_r5={cd_st['avg_r5']:+.3f}%  "
              f"win={cd_st['win5']*100:.1f}%  sumPnL5={cd_st['sum_r5']:+.1f}%  "
              f"Sharpe={cd_st['sharpe']:+.2f}")

    # ── Best rotation filter + cooldown (union) ──
    if best.get("n", 0):
        best_subset = [r for r in port if r["score"] >= best["sc"] and (best["mlp"] is None or (r["mlp"] is not None and r["mlp"] >= best["mlp"]))]
        union_rows = best_subset + cd
        un_st = block_stats(union_rows, "best_rot + cooldown")
        if un_st:
            print(f"\n[UNION best_rot + cooldown-approved]:")
            print(f"  Best rot: score>={best['sc']}, mlp>={best['mlp']}")
            print(f"  n={un_st['n']}  avg_r5={un_st['avg_r5']:+.3f}%  "
                  f"win={un_st['win5']*100:.1f}%  sumPnL5={un_st['sum_r5']:+.1f}%  "
                  f"avg_r10={un_st['avg_r10']:+.3f}%  win10={un_st['win10']*100:.1f}%")


if __name__ == "__main__":
    main()
