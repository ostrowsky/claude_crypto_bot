# ML candidate ranker (CatBoost)

- **Slug:** `ml-candidate-ranker`
- **Status:** shipped (retroactive)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/ml_candidate_ranker.py`, `files/ml_candidate_ranker.json`,
  `files/ml_candidate_ranker_report.json`

---

## 1. Problem

После ML signal-gating нужно ещё ранжировать кандидатов: дать **EV-оценку**
ожидаемого return / drawdown / capture-ratio, чтобы (а) выкидывать заведомо
плохие сетапы (`ranker_hard_veto`) и (б) кормить bandit как фичу.

## 2. Success metric

- AUC quality-классификатора стабилен (drift-аларм при падении > 0.05).
- `ranker_hard_veto` на blocked-выборке → avg_r5 < take avg_r5 (т.е. вето
  отсекает реально плохое, а не top-gainer’ов).

## 3. Scope

### In scope
- CatBoost-модель (multi-head: quality_proba, EV, expected_return,
  expected_drawdown, top_gainer_prob, capture_ratio_pred).
- Hard-veto гард `ml_candidate_ranker_hard_veto` в `trend_scout_rules.py`.

### Out of scope
- Signal classifier (`ml_signal_model.py`) — отдельная спека.
- Top-gainer model — отдельная спека.

## 4. Behaviour / design

### Inference

`monitor.py` на каждый кандидат собирает фичи (slope, adx, rsi, vol_x,
ml_proba, daily_range, macd_hist, …) и зовёт `ml_candidate_ranker.predict`.
Результат пишется в `pos.*`:

```
ranker_quality_proba
ranker_final_score
ranker_ev
ranker_expected_return
ranker_expected_drawdown
ranker_top_gainer_prob
ranker_capture_ratio_pred
```

### Hard veto

Гард в `trend_scout_rules.py::ranker_hard_veto`. Блокирует, если:

- 15m: `final_score < ML_CANDIDATE_RANKER_HARD_VETO_15M_FINAL_MAX (-2.50)`.
- 1h: `final_score < threshold` **И** `top_gainer_prob <
  ML_CANDIDATE_RANKER_HARD_VETO_1H_TOP_GAINER_MAX (0.25)`.

(double-condition на 1h — после 2026-04-13, чтобы не вырезать TG-кандидатов).

### Training

Ежедневно `daily_learning.py` вызывает retrain поверх `ml_dataset.jsonl`
(featurised, ~103 MB). Артефакты: `ml_candidate_ranker.json` (model blob),
`ml_candidate_ranker_report.json` (metrics), shadow-вариант для A/B.

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `ML_CANDIDATE_RANKER_HARD_VETO_ENABLED` | True | Off → ranker используется только как фича |
| `ML_CANDIDATE_RANKER_HARD_VETO_15M_FINAL_MAX` | −2.50 | 15m порог |
| `ML_CANDIDATE_RANKER_HARD_VETO_1H_TOP_GAINER_MAX` | 0.25 | 1h дополнительный порог по TG |

Откат: `ML_CANDIDATE_RANKER_HARD_VETO_ENABLED=False`.

## 6. Risks

- **Over-blocking**: до 2026-04-06 вето было `-0.75` и резало 100 % TG.
  Любой сдвиг порога — только после Pareto sweep
  (`files/analyze_blocked_gates.py`).
- **Drift модели** при пропусках EOD-обучения.

## 7. Verification

- Pareto sweep по reason_code: `ranker_hard_veto` не должен иметь
  `avg_r5 > take avg_r5`. На 2026-04 ещё over-blocking
  (`+0.128 %` blocked vs `−0.016 %` take, n=787) — to-fix candidate.

## 8. Follow-ups

- Tightening 1h-веточки на основе TG-prob: рассмотреть пары
  (final_score, top_gainer_prob) в Pareto.
