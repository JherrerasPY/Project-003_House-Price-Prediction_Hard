# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Solo ML project predicting house prices from tabular data. All work lives in [main.ipynb](main.ipynb).
Dataset: `../Datasets/Housing.csv` (545 rows, 12 features — not committed to git).

**Goal:** Implement and compare three models — Decision Tree (done), Random Forest, XGBoost — using a
shared preprocessing pipeline and consistent evaluation. Beat the current baseline of R² = 0.545.

## Running the Notebook

```bash
jupyter notebook main.ipynb   # or: jupyter lab
pip install xgboost           # required for XGBoost cells
```

Run all cells top-to-bottom. No tests, scripts, or separate modules.

## Architecture

Linear sklearn `Pipeline` workflow:

1. **Data loading** — `pd.read_csv("../Datasets/Housing.csv")`
2. **Target transform** — `y = np.log1p(price)`; always reverse with `np.expm1()` before evaluation
3. **Train/test split** — 70/30, `random_state=1`; use `cross_val_score(cv=5)` for model comparison
4. **Preprocessing** (`ColumnTransformer`):
   - Numeric: `RobustScaler`
   - Categorical: `OneHotEncoder(drop='first')` — `furnishingstatus` is ordinal (unfurnished < semi < furnished)
5. **Models** — each in its own named `Pipeline`: `dtree_pipeline`, `rf_pipeline`, `xgb_pipeline`
6. **Evaluation** — `evaluate_model(y_true, y_pred, name)` reports R², RMSE, MAE on real-scale prices

## Model Roadmap

| Step | Model | Status |
| --- | --- | --- |
| 1 | `DecisionTreeRegressor(max_depth=4)` | ✅ Done — R² 0.545 test |
| 2 | `RandomForestRegressor` | ⬜ Next |
| 3 | `XGBRegressor` | ⬜ After RF |

XGBoost param grid keys must be prefixed: `xgbregressor__max_depth`, `xgbregressor__learning_rate`, etc.

## Key Facts

- Binary yes/no columns (6): `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
- `furnishingstatus`: 3-level ordinal — currently OHE'd but `OrdinalEncoder` is more appropriate
- Fitted sub-objects: `pipeline.named_steps['preprocessor']`, `pipeline.named_steps['model']`
- Feature importance: `preprocessor.get_feature_names_out()` gives post-transform column names
- Stack: Python 3, pandas 3.0, numpy 2.3, seaborn 0.13, scikit-learn (unpinned), xgboost

## Plan Before Code

For any non-trivial addition (new model, feature engineering block, CV loop), propose the approach
and wait for approval before writing cells.
