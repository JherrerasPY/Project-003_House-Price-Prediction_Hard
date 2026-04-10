# House Price Prediction — Ensemble & Boosting Models

Tabular regression project comparing Decision Tree, Random Forest, and XGBoost on a small real-estate dataset. The emphasis is on rigorous pipeline design, honest model evaluation, and diagnosing why tuning succeeded in some cases and backfired in others.

---

## Dataset

- **Source:** `Housing.csv` — 545 rows, 12 features
- **Target:** `price` (log-transformed during training, reversed for evaluation)
- **Split:** 70/30 stratified, `random_state=1`; 5-fold cross-validation for model selection

**Features:**

- Continuous: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
- Binary (yes/no): `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
- Ordinal: `furnishingstatus` (unfurnished → semi-furnished → furnished)

---

## Pipeline Architecture

All models share the same preprocessing pipeline to prevent data leakage:

```text
ColumnTransformer
├── Numeric  → RobustScaler
└── Categorical → OneHotEncoder(drop='first')
        ↓
    Model
```

Built with `sklearn.Pipeline` — fitting and transforming are always scoped to training data only.

---

## Results

### Baseline (no tuning)

| Model | Test R² | Test RMSE | Test MAE |
| --- | --- | --- | --- |
| Decision Tree (`max_depth=4`) | 0.4817 | 1,452,486 | 803,524 |
| Random Forest (100 trees) | 0.6090 | 1,261,554 | 731,567 |
| **XGBoost** | **0.6866** | **1,129,522** | **408,123** |

### After `GridSearchCV` (cv=5)

| Model | Test R² | vs. Baseline | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| Decision Tree | 0.4584 | −0.023 | 1,484,811 | — |
| **Random Forest** | **0.6620** | **+0.053** | **1,172,964** | **546,897** |
| XGBoost | 0.6534 | −0.033 | 1,187,776 | 532,200 |

**Best overall:** XGBoost baseline — R² = 0.6866

---

## Why Tuning Backfired (and When It Didn't)

A key finding is that exhaustive `GridSearchCV` on a 381-sample training set is a high-variance estimator of the true best hyperparameters — each CV fold contains only ~76 samples.

**Decision Tree** — `max_depth=None` was selected, allowing the tree to memorize fold-specific noise. Performance degraded on held-out test data.

**XGBoost** — the search reduced `learning_rate` from 0.1 → 0.05 without proportionally increasing `n_estimators` (remained at 100). A slower learning rate requires more boosting rounds to converge; the selected config left the model under-boosted relative to the default.

**Random Forest** — ensemble averaging over 300 trees absorbed CV selection noise better than a single tree could. Increasing `n_estimators` to 300 and setting `max_depth=8` improved both bias and variance simultaneously, yielding a genuine +5.3 pp gain.

**Best parameters found:**

| Model | Key Params |
| --- | --- |
| Decision Tree | `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=4` |
| Random Forest | `n_estimators=300`, `max_depth=8`, `min_samples_split=5` |
| XGBoost | `learning_rate=0.05`, `n_estimators=100`, `max_depth=4`, `reg_lambda=2.0`, `colsample_bytree=0.7` |

---

## Feature Importance

From the Random Forest's post-transform feature names:

- `area` — ~56% importance (dominant predictor)
- `bathrooms` — ~25% importance
- All remaining features combined — ~19%

This 80/20 split reveals the ceiling: the dataset carries limited information. No amount of model complexity recovers signal that is not present in the features.

---

## Identified Limitations

1. **No location data** — neighbourhood, school district, and proximity to transit are the top real-estate price drivers. Absent here entirely.
2. **Small dataset** — 381 training samples is insufficient for tree-based methods to generalize reliably; it also makes `GridSearchCV` prone to overfitting the search process.
3. **Ordinal feature treated as nominal** — `furnishingstatus` is one-hot encoded, discarding its natural order. `OrdinalEncoder` would be more appropriate.
4. **No interaction features** — `area × bathrooms`, composite `amenity_score`, and `area_per_bedroom` are natural candidates that were not explored.

---

## Potential Next Steps

| Tier | Action |
| --- | --- |
| Feature engineering | `amenity_score`, `area_per_bedroom`, `area × bathrooms`, OrdinalEncoder for `furnishingstatus` |
| Tuning strategy | `RandomizedSearchCV` or `Optuna` instead of exhaustive grid; tune XGBoost `learning_rate` + `n_estimators` jointly |
| Data | Collect ~1,000+ rows; add external features (location, census data) |
| Stacking | Blend RF and XGBoost predictions as a meta-learner input |

---

## Stack

Python 3 · pandas 3.0 · numpy 2.3 · scikit-learn · xgboost · seaborn 0.13
