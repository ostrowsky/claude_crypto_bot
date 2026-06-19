"""Test the top transferable idea from the freqtrade-futures review: do
FUTURES ALT-DATA features (funding rate, open-interest change, long/short ratio)
— which our bot does NOT use — discriminate winners from losers where our
price/indicator features failed (multivariate OOS AUC ~0.50)?

If price-only OOS AUC ~0.50 but price+altdata lifts it meaningfully (>0.55),
alt-data is a real new signal worth ingesting. If not, even alt-data is at the
frontier and we stop chasing capture.

Method (read-only): recent take entries from critic_dataset (alt-data only
covers ~last ~20d), join Binance funding/OI/LS at entry time, train logistic
(price-only vs +altdata) with temporal split, report OOS AUC + univariate
discrimination. ASCII-only.  pyembed\python.exe files\_backtest_altdata_signal.py
"""
from __future__ import annotations
import json, sys, time, urllib.request
from datetime import datetime, timezone, timedelta
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DAYS = 18
CUTd = (datetime.now(timezone.utc) - timedelta(days=DAYS)).strftime("%Y-%m-%d")
PFEATS = ["close_vs_ema50", "close_vs_ema200", "ema50_vs_ema200", "slope", "rsi",
          "adx", "vol_x", "macd_hist_norm", "atr_pct", "daily_range",
          "btc_vs_ema50", "btc_momentum_4h"]


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def _get(u):
    req = urllib.request.Request(u, headers={"User-Agent": "rev"})
    return json.load(urllib.request.urlopen(req, timeout=20))


# ---- load recent entries ----
rows = []
syms = set()
for ln in open("critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    try: e = json.loads(ln)
    except: continue
    d = str(e.get("ts_signal", ""))[:10]
    if d < CUTd: continue
    if str((e.get("decision", {}) or {}).get("action", "")) != "take": continue
    r5 = _f((e.get("labels", {}) or {}).get("ret_5"))
    if r5 is None: continue
    f = e.get("f", {}) or {}
    x = [_f(f.get(k)) for k in PFEATS]
    if any(v is None for v in x): continue
    try: ts = int(datetime.fromisoformat(str(e.get("ts_signal","")).replace("Z","+00:00")).timestamp()*1000)
    except: continue
    sym = e.get("sym")
    rows.append({"sym": sym, "ts": ts, "day": d, "x": x, "r5": r5})
    syms.add(sym)
print(f"recent take entries (last {DAYS}d): {len(rows)}  symbols: {len(syms)}")

# ---- fetch alt-data per symbol (cached) ----
fund, oi, ls = {}, {}, {}
ok = 0
for i, s in enumerate(sorted(syms)):
    try:
        fund[s] = sorted((int(z["fundingTime"]), float(z["fundingRate"]))
                         for z in _get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={s}&limit=200"))
        oi[s] = sorted((int(z["timestamp"]), float(z["sumOpenInterest"]))
                       for z in _get(f"https://fapi.binance.com/futures/data/openInterestHist?symbol={s}&period=1h&limit=500"))
        ls[s] = sorted((int(z["timestamp"]), float(z["longShortRatio"]))
                       for z in _get(f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={s}&period=1h&limit=500"))
        ok += 1
    except Exception:
        fund[s] = oi[s] = ls[s] = []
    time.sleep(0.06)
print(f"alt-data fetched for {ok}/{len(syms)} symbols")


def _last_before(arr, ts):
    v = None
    for t, x in arr:
        if t <= ts: v = x
        else: break
    return v

def _oi_chg(arr, ts, hrs=4):
    now = _last_before(arr, ts); prev = _last_before(arr, ts - hrs*3600*1000)
    if now is None or prev is None or prev <= 0: return None
    return (now - prev) / prev * 100

# ---- build alt features per entry ----
keep = []
for r in rows:
    fr = _last_before(fund.get(r["sym"], []), r["ts"])
    oic = _oi_chg(oi.get(r["sym"], []), r["ts"], 4)
    lsr = _last_before(ls.get(r["sym"], []), r["ts"])
    if fr is None or oic is None or lsr is None: continue
    r["alt"] = [fr*100.0, oic, lsr]   # funding%(per-interval), OI 4h chg %, L/S ratio
    keep.append(r)
print(f"entries with full alt-data: {len(keep)}")
if len(keep) < 60:
    print("not enough joined data"); sys.exit(0)

# ---- univariate discrimination ----
print("\nUnivariate: avg ret_5 by alt-feature tercile (low/mid/high)")
names = ["funding%", "OI_4h_chg%", "longshort"]
arr = np.array([r["alt"] for r in keep], float)
y5 = np.array([r["r5"] for r in keep])
for j, nm in enumerate(names):
    col = arr[:, j]; q1, q2 = np.percentile(col, [33, 66])
    lo = y5[col <= q1].mean(); mid = y5[(col > q1) & (col <= q2)].mean(); hi = y5[col > q2].mean()
    print(f"  {nm:<11} low={lo:+.3f}  mid={mid:+.3f}  high={hi:+.3f}   (n/3~{len(keep)//3})")

# ---- logistic OOS: price-only vs price+alt ----
keep.sort(key=lambda r: r["day"])
days = sorted({r["day"] for r in keep})
nt = max(1, int(round(len(days)*0.3)))
testd = set(days[-nt:])
tr = [r for r in keep if r["day"] not in testd]; te = [r for r in keep if r["day"] in testd]

def fit_auc(use_alt):
    def vec(r): return r["x"] + (r["alt"] if use_alt else [])
    Xtr = np.array([vec(r) for r in tr], float); Xte = np.array([vec(r) for r in te], float)
    ytr = np.array([1.0 if r["r5"]>0 else 0.0 for r in tr]); yte = np.array([1.0 if r["r5"]>0 else 0.0 for r in te])
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr=np.hstack([Xtr,np.ones((len(Xtr),1))]); Xte=np.hstack([Xte,np.ones((len(Xte),1))])
    w=np.zeros(Xtr.shape[1]); lam=1.0
    for _ in range(3000):
        p=1/(1+np.exp(-Xtr@w)); g=Xtr.T@(p-ytr)/len(ytr)+lam*np.r_[w[:-1],0]/len(ytr); w-=0.1*g
    def auc(s,yv):
        pos=s[yv==1]; neg=s[yv==0]
        if not len(pos) or not len(neg): return float("nan")
        o=np.argsort(s); rk=np.empty_like(o,float); rk[o]=np.arange(1,len(s)+1)
        return (rk[yv==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    return auc(Xtr@w,ytr), auc(Xte@w,yte)

a_tr0, a_te0 = fit_auc(False)
a_tr1, a_te1 = fit_auc(True)
print(f"\ntrain={len(tr)} test={len(te)}  (target ret_5>0)")
print(f"  price-only   : AUC train={a_tr0:.3f}  OOS={a_te0:.3f}")
print(f"  price+altdata: AUC train={a_tr1:.3f}  OOS={a_te1:.3f}")
print(f"\nVERDICT: if OOS rises from ~{a_te0:.2f} to >0.55 with alt-data, funding/OI/")
print("L-S is a real new signal worth ingesting. If ~flat, alt-data is at frontier too.")
