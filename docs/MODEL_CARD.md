# Model Card - Return Risk Ensemble

## Model details

- **Type:** Soft-voting classifier (RandomForest + XGBoost + GradientBoosting)
- **Task:** Binary return-abuse risk scoring (`abuse_label`)
- **Features:** Behavioral, account, order, and categorical encodings listed in `src/data_prep.FEATURE_COLS`
- **Explainability:** SHAP TreeExplainer on the RF member (importance fallback)

## Data

- **Primary:** E-commerce Return Abuse Detection (synthetic)
- **Supplement:** Returns management / sustainability CSV (category+reason proxies for SLA and return cost)
- **Split:** Chronological (time-based), not random

## Intended use

Decision **support** for return-operations analysts. Outputs are recommendations only.

## Out of scope

- Autonomous blocking, charging, or account punishment
- Claim of production-equivalent accuracy on live merchant traffic

## Metrics honesty

Metrics in `data/processed/metrics.json` are computed on a synthetic held-out chronological slice. They **must not** be interpreted as production performance.

Direct synthetic proxy flags (`multiple_accounts_flag`, `refund_to_different_account`, `address_change_before_delivery`) are **excluded from the ML feature matrix** because they can nearly encode the label. Those fields remain available to the pattern / shared-signal tools as operational evidence, not as model inputs.

Suspiciously perfect precision/recall on related public fraud datasets often indicates leakage (for example balance-before/after identity features). This project avoids those domains, uses a time split, and documents remaining synthetic-data limits openly.

Even after dropping direct proxy flags, this synthetic return-abuse set remains highly separable on behavioral aggregates (e.g. return_rate_pct). Treat near-perfect held-out numbers as a **dataset artifact**, not a production claim.

## Feature-Ablation Stress Test

Reproducible evaluation only (`scripts/feature_ablation.py`). Same soft-voting ensemble builder and chronological split as `train_ensemble`; **does not** overwrite the production model bundle.

RandomForest importance on the full feature set (sample_n=20000) ranks:

1. `return_rate_pct` (~0.388)
2. `customer_support_contacts` (~0.219)
3. `total_returns_lifetime` (~0.145)

| Metric | Full feature set (`metrics.json`) | Without `return_rate_pct` | Without top-3 importance features\* |
|--------|-----------------------------------|---------------------------|-------------------------------------|
| Precision | 1.000 | 1.000 | 0.987 |
| Recall | 0.998 | 0.999 | 0.933 |
| F1 | 0.999 | 1.000 | 0.959 |
| ROC-AUC | 1.000 | 1.000 | 0.994 |

\*Top-3 ablated: `return_rate_pct`, `customer_support_contacts`, `total_returns_lifetime`. Ablation test slice: n_test=4000 (sample_n=20000, test_ratio=0.2). Full numbers: `data/processed/ablation_metrics.json`.

Removing only `return_rate_pct` barely changes held-out scores because correlated behavioral aggregates still carry the same synthetic signal. Removing the top-three importance features drops recall from ~0.998 to ~0.933 and F1 from ~0.999 to ~0.959, confirming the near-perfect ceiling is driven by a small cluster of strong synthetic behavioral signals rather than leakage or overfitting across the full remaining feature set.

Re-run:

```bash
set PYTHONPATH=.
python scripts/feature_ablation.py --sample-n 20000
```

## Cost reporting

Alongside classification metrics, training reports:

- FP friction cost (₹)
- Fraud loss prevented (₹)
- Net Protected Value and x-times ROI

## Limitations

- Agent iteration cap (default 4) bounds cost/latency at the expense of exhaustive investigation.
- Shared-signal graph uses coarse proxies (device / payment / country), not true device fingerprints.
- Demo scenarios and dashboard are tagged **DEMO DATA**.
