# Churn Prediction MLOps Pipeline

An end-to-end ML workflow: generate data -> train and compare models -> quality gate -> versioned model registry -> REST serving -> drift monitoring. The point is the **engineering around the model**, not the model.

> Honest scope
> - The data is **synthetic** (generated from a made-up rule plus noise, see `churn/data.py`). The scores below say nothing about real customers. Swap `make_data()` for a real CSV to use it for real.
> - The "registry" is a folder of versioned artifacts, not MLflow. No experiment tracker, scheduler or cloud deployment.
> - Docker files were **not built or run** (Docker unavailable). Training, tests and the API were run locally on Python 3.10.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # 6 tests
python -m churn.train           # trains, evaluates, registers into models/
uvicorn churn.serve:app         # then POST /predict, GET /model, GET /health
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
 -d '{"tenure_months":2,"monthly_charge":110,"support_calls":6,"contract":"monthly","late_payments":3}'
```

## Pipeline
```
make_data -> train/test split (stratified) -> 5-fold CV on train for 2 candidates -> pick best
   -> fit -> evaluate once on held-out test -> quality gate (AUC >= 0.70)
   -> models/<version>/{model.joblib, metadata.json}, models/latest.txt -> FastAPI loads latest
drift.py: compare new data against training histograms with PSI
```

## Results (one real run, seed 0, 4000 synthetic rows)
| | CV ROC-AUC (train split) |
|---|---|
| Logistic regression | 0.787 |
| Gradient boosting | 0.767 |

Chosen: logistic regression. Held-out test: ROC-AUC 0.760, precision 0.606, recall 0.388, F1 0.474 (threshold 0.5).
Logistic regression beating boosting is expected here: the synthetic data was generated from a logistic rule. Recall is low at 0.5; a lower threshold would find more churners at the cost of precision, which is a business decision.
Drift check on a fresh sample from the same generator: all PSI < 0.02 (stable). When one feature is shifted by +60 in the tests, it is flagged "major".

## How it works / key decisions
1. **sklearn Pipeline** bundles scaling + one-hot encoding + model, so training and serving use identical preprocessing (prevents train/serve skew).
2. **No leakage**: the best model is picked with cross-validation on the training split only; the test set is used once at the end.
3. **Quality gate**: a model below the AUC threshold raises an error and is *not* registered (`test_quality_gate_blocks_bad_model`).
4. **Versioning**: each run writes an immutable folder with the model and metadata (metrics, features, seed, training histograms). `latest.txt` points to the live one, so rollback = edit one line.
5. **Serving & drift**: Pydantic validates inputs (bad contract or negative values -> 422). PSI = sum((actual% - expected%) * ln(actual%/expected%)) per feature; <0.1 stable, 0.1 to 0.25 moderate, >0.25 major, which is when to retrain.

## Limitations / next steps
Real data, experiment tracking (MLflow), scheduled retraining, drift endpoint, model calibration and threshold tuning, monitoring of prediction distribution and delayed labels.

## Interview questions to prepare
What is train/serve skew? Why use CV for selection and a separate test set? What does PSI measure and what do you do when it is high? Why is accuracy a bad metric for churn? How would you roll back a bad model?

## CI
A GitHub Actions workflow is in `ci/github-actions-ci.yml`. Move it to `.github/workflows/ci.yml` to enable it. It is not enabled because the push token had no `workflow` scope, and it has not been run on GitHub.
