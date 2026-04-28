# ML signal-model blind-spot recovery

- **Slug:** `ml-signal-blindspot-recovery`
- **Status:** draft
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/ml_signal_model.py`, `_validate_p5_ml_blindspots.py`,
  `docs/reports/2026-04-28-improvements-validated.md` §P5

---

## 1. Problem

ML signal-model имеет систематические blind-spots: даёт `proba < 0.10`
для coin’ов, которые в тот же день оказываются в EOD top-20 gainers.

Backtest 30 d (`_validate_p5_ml_blindspots.py`):
- **34 top-20 winner-days** проходят через ML-zone block (~24 % всех
  winner-days).
- **19 из 34** — extreme blind-spot (`proba < 0.10`).
- Recurring offenders: `TRUUSDT` (3), `BLURUSDT` (3), `MDTUSDT` (2),
  `ORDIUSDT` (2), `AUDIOUSDT` (2), `QIUSDT` (2), `COMPUSDT` (2),
  `APEUSDT` (2).

AUC top20 = 0.989 обманчив — модель права на 99 % случаев, но
систематически промахивается на определённом подтипе сетапа.

## 2. Success metric

**Primary:** доля top-20 winner-days, заблокированных ML-zone, падает
с 24 % до **≤ 10 %** на out-of-sample тесте.

**Secondary:**
- Общий AUC top20 не падает ниже 0.97 (текущее 0.989).
- Recall@top20 на overall не падает (текущее 100 %).
- На subset blind-spot syms (TRU/BLUR/MDT/ORDI/AUDIO) `recall ≥ 50 %`.

## 3. Scope

### In scope
- Audit blind-spot выборки: feature-level diff vs «правильно
  отработанные» top-20 winners.
- Oversampling в training set с весом ×3 на blind-spot examples.
- Retrain `ml_signal_model.json` через `daily_learning.py` с обновлённым
  весом.
- Shadow A/B (использовать существующую `ml_candidate_ranker_shadow_report.json`
  инфраструктуру).

### Out of scope
- Замена CatBoost на другой алгоритм.
- Изменение signal pipeline / гард-каскада.
- Расширение feature space (отдельной спекой при необходимости).

## 4. Behaviour / design

### Audit phase

Скрипт `files/_audit_ml_blindspots.py`:
1. Прочитать критик-датасет, отфильтровать `top20_label=1 AND ml_proba<0.10`.
2. Для каждого case: dump полный feature vector.
3. Сравнить с distribution фичей у НЕ-blind-spot top-20 winners
   (где ml_proba > 0.5).
4. Найти features, где blind-spot выборка систематически отличается
   (разрыв средних > 1 σ).

Hypothesis: blind-spot syms имеют, например, `low_volume_ratio`,
`stale_orderbook_depth`, или другую low-frequency сигнатуру, которая
плохо представлена в training set.

### Mitigation phase

Два варианта (выбрать после audit):

**A) Sample weighting** (предпочтительно):
- В `train_ml_signal_model`: `sample_weight = 3.0 if blind_spot_match
  else 1.0`.
- Blind-spot match = (top20_label=1) AND (ml_proba_old < 0.10).

**B) Synthetic oversampling**:
- Дублировать blind-spot rows ×3 в training set.

Выбор А — менее инвазивен, не нарушает class balance overall.

### Retrain & shadow

- Запустить retrain в `daily_learning.py` с новым весом.
- 7 d shadow A/B: записывать решения новой модели в
  `ml_signal_shadow_report.json`, но не использовать live.
- Метрики: AUC overall, AUC blind-spot subset, false-positive rate.

### Promotion

После 7 d shadow + audit метрик → swap `ml_signal_model.json`.
Старый артефакт сохранить как `ml_signal_model_pre_blindspot.json`
для быстрого rollback.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `ML_SIGNAL_BLINDSPOT_WEIGHT` | 3.0 | Weight multiplier при retrain |
| `ML_SIGNAL_BLINDSPOT_PROBA_THRESHOLD` | 0.10 | Что считать blind-spot |
| `ML_SIGNAL_SHADOW_ENABLED` | True | Logging shadow predictions |

**Rollback:** swap `ml_signal_model.json` обратно с
`ml_signal_model_pre_blindspot.json`. Никаких schema-changes.

## 6. Risks

- **Over-fit на blind-spot syms.** Mitigation: `ML_SIGNAL_BLINDSPOT_WEIGHT`
  ≤ 5; больше — модель начинает переоценивать low-volume сигналы
  (false positives на не-winners).
- **AUC overall просядет.** Mitigation: shadow A/B; promotion только
  если AUC ≥ 0.97.
- **Concept drift на recurring syms.** TRU/BLUR могут перестать быть
  blind-spot после следующих дней. Mitigation: weight пересчитывается
  каждый daily retrain.

## 7. Verification

- [ ] **Audit script** committed and run on 30 d data; output
      summary blind-spot feature signature.
- [ ] **Retrain** с weight=3.0; запись новой модели как
      `ml_signal_model_v2.json`.
- [ ] **Shadow report** 7 d:
      `ml_signal_shadow_report.json` сравнивает старую и новую модель
      на live data.
- [ ] **Acceptance**:
  - blind-spot block rate (24 % → ≤ 10 %) на out-of-sample;
  - AUC overall ≥ 0.97;
  - false-positive rate (proba > 0.5 на не-winners) не вырос > 20 %
    относительно текущего.
- [ ] **Promote** swap models, keep old as backup.

## 8. Follow-ups

- Если weight=3.0 не достаточно → попробовать =5.0 или synthetic
  oversampling (вариант B).
- Periodic audit: скрипт `_audit_ml_blindspots.py` запускать каждые
  30 d, обновлять weight automatically (если blind-spot rate растёт).
- Расширение features: если audit покажет конкретный gap (vol, spread,
  market_cap) — отдельная спека на feature engineering.
