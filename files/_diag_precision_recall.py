"""
Bot precision / recall vs top-20 daily gainers.

Precision: of all bot entries on day D, how many coins ended in daily top-20?
Recall:    of all top-20 daily gainers (in watchlist), how many did bot enter at least once?

Sources:
  - bot_events.jsonl : entry events (sym, ts, mode, tf)
  - top_gainer_dataset.jsonl : daily snapshots with label_top20 per coin per day
    (fallback: critic_dataset.jsonl labels.label_5 for bot-signal rows)

Also reports:
  - Per-day breakdown: entries, precision, recall, top20_in_watchlist
  - Per-mode breakdown (last 14 days)
  - Model AUC + bandit UCB sep from learning_progress.jsonl
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent
RUNTIME = FILES.parent / ".runtime"

# ── 1. Load entries from bot_events.jsonl ────────────────────────────────
entries_by_date: dict[str, list[dict]] = defaultdict(list)  # date -> [entry events]
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "entry": continue
        ts = e.get("ts", "")
        date = ts[:10] if ts else ""
        if not date: continue
        entries_by_date[date].append({
            "sym": e.get("sym", ""),
            "mode": e.get("mode", ""),
            "tf": e.get("tf", ""),
            "ts": ts,
        })

# ── 2. Load daily top-20 labels from top_gainer_dataset.jsonl ────────────
# Each row: sym, date (or ts), label_top20 (bool/int)
top20_by_date: dict[str, set[str]] = defaultdict(set)  # date -> {sym}

print("Loading top_gainer_dataset (streaming)...", flush=True)
loaded_tg = 0
with io.open(FILES / "top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: row = json.loads(ln)
        except: continue
        label = row.get("label_top20")
        if label is None: label = row.get("label") or row.get("top20")
        if not label: continue       # skip label=0 or None
        sym = row.get("symbol") or row.get("sym", "")
        ts = row.get("ts") or row.get("date") or row.get("snapshot_ts")
        if ts is None: continue
        if isinstance(ts, (int, float)):
            # Unix ms → seconds
            try: date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
            except: continue
        else:
            date = str(ts)[:10]
        if sym and date:
            top20_by_date[date].add(sym)
            loaded_tg += 1
print(f"  top_gainer_dataset: {loaded_tg} top-20 label rows across {len(top20_by_date)} dates")

# Fallback: also collect from critic_dataset label_5 (ENTERED rows only)
# label_5 typically: 1 if ret_5 > threshold, not necessarily top20 — skip for recall
# Instead use learning_progress n_top20_in_watchlist as denominator when TG data missing

# ── 3. Load learning progress ─────────────────────────────────────────────
lp_by_date: dict[str, dict] = {}
lp_path = RUNTIME / "learning_progress.jsonl"
if lp_path.exists():
    with io.open(lp_path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            try: row = json.loads(ln)
            except: continue
            ts = row.get("ts", "")
            # learning runs ~00:30 UTC — date is previous day
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                prev_day = (dt - timedelta(hours=2)).date().isoformat()
            except: continue
            lp_by_date[prev_day] = row

# ── 4. Compute daily precision / recall ──────────────────────────────────
all_dates = sorted(set(entries_by_date.keys()) | set(top20_by_date.keys()))
# Keep only dates with at least some entries OR top20 data
all_dates = [d for d in all_dates if entries_by_date.get(d) or top20_by_date.get(d)]

print(f"\n=== DAILY PRECISION / RECALL (last 30 days with data) ===")
print(f"{'Date':>10}  {'Entries':>7}  {'Prec%':>6}  {'Recall%':>7}  "
      f"{'TP':>3}  {'top20_wl':>8}  {'model_AUC':>9}  {'UCB_sep':>7}")
print("-" * 80)

daily_rows = []
for date in sorted(all_dates)[-30:]:
    ents = entries_by_date.get(date, [])
    top20 = top20_by_date.get(date, set())
    lp = lp_by_date.get(date, {})

    entered_syms = {e["sym"] for e in ents}
    tp_syms = entered_syms & top20
    tp = len(tp_syms)

    n_entries = len(set(entered_syms))           # unique syms entered
    n_top20_wl = len(top20)

    # fallback: use n_top20_in_watchlist from learning progress
    if n_top20_wl == 0 and lp:
        n_top20_wl = lp.get("n_top20_in_watchlist", 0)

    prec  = tp / n_entries * 100 if n_entries else None
    rec   = tp / n_top20_wl * 100 if n_top20_wl else None

    auc   = lp.get("model_auc_top20") if lp else None
    ucb   = lp.get("bandit_ucb_separation") if lp else None

    prec_s = f"{prec:>6.1f}" if prec is not None else f"{'—':>6}"
    rec_s  = f"{rec:>7.1f}" if rec is not None else f"{'—':>7}"
    auc_s  = f"{auc:>9.4f}" if auc is not None else f"{'—':>9}"
    ucb_s  = f"{ucb:>7.4f}" if ucb is not None else f"{'—':>7}"

    print(f"{date:>10}  {n_entries:>7}  {prec_s}  {rec_s}  {tp:>3}  {n_top20_wl:>8}  {auc_s}  {ucb_s}")
    daily_rows.append({"date": date, "entries": n_entries, "tp": tp,
                       "top20_wl": n_top20_wl, "prec": prec, "rec": rec})

# ── 5. Aggregate last 7 / 14 / 30 days ───────────────────────────────────
print()
for window, label in [(7, "last 7d"), (14, "last 14d"), (30, "last 30d")]:
    window_rows = daily_rows[-window:]
    tot_entries = sum(r["entries"] for r in window_rows)
    tot_tp      = sum(r["tp"] for r in window_rows)
    tot_top20   = sum(r["top20_wl"] for r in window_rows)
    prec_agg = tot_tp / tot_entries * 100 if tot_entries else None
    rec_agg  = tot_tp / tot_top20  * 100 if tot_top20  else None
    p_s = f"{prec_agg:.1f}%" if prec_agg is not None else "—"
    r_s = f"{rec_agg:.1f}%"  if rec_agg  is not None else "—"
    print(f"  {label:10s}: entries={tot_entries}  TP={tot_tp}  top20_wl={tot_top20}"
          f"  precision={p_s}  recall={r_s}")

# ── 6. Per-mode breakdown (last 14 days) ─────────────────────────────────
print(f"\n=== PER-MODE breakdown (last 14 days) ===")
cutoff = sorted(all_dates)[-14] if len(all_dates) >= 14 else "0000"

mode_tp:      dict[str, int] = defaultdict(int)
mode_entries: dict[str, set] = defaultdict(set)  # mode -> set of (date,sym)

for date in all_dates:
    if date < cutoff: continue
    ents = entries_by_date.get(date, [])
    top20 = top20_by_date.get(date, set())
    for e in ents:
        key = (e["mode"], e["tf"])
        mode_entries[key].add((date, e["sym"]))
        if e["sym"] in top20:
            mode_tp[key] += 1

for key in sorted(mode_entries.keys()):
    n = len(mode_entries[key])
    tp = mode_tp.get(key, 0)
    prec = tp / n * 100 if n else 0
    print(f"  {key[0]:>16s} {key[1]:>4s}  entries={n:>4d}  TP={tp:>3d}  precision={prec:.1f}%")

# ── 7. Learning progress timeline ────────────────────────────────────────
print(f"\n=== LEARNING PROGRESS (bandit + model) ===")
print(f"{'Date':>10}  {'Recall@20':>9}  {'UCB_sep':>7}  {'AUC_top20':>9}  {'Updates':>8}  {'n_signal':>8}")
lp_sorted = sorted(lp_by_date.items())
for date, row in lp_sorted[-14:]:
    rec20 = row.get("bandit_recall_top20", 0) * 100
    ucb   = row.get("bandit_ucb_separation", 0)
    auc   = row.get("model_auc_top20") or 0
    upd   = row.get("bandit_total_updates", 0)
    nsig  = row.get("bandit_n_signal", 0)
    print(f"  {date:>10}  {rec20:>8.1f}%  {ucb:>7.4f}  {auc:>9.4f}  {upd:>8,}  {nsig:>8,}")
