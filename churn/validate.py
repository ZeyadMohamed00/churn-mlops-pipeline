"""Stronger validation than a single 80/20 split.

Run: python -m churn.validate   (writes reports/validation.json)

- 5-fold stratified cross-validation on the training split: mean and standard deviation per model.
- Bootstrap 95% confidence interval for ROC-AUC on the held-out test split (1,000 resamples).
- Threshold sweep on the held-out split for the served model, plus the threshold that maximises
  expected value under an explicit, adjustable cost assumption (a missed churner costs
  COST_MISSED, a retention offer to someone who would have stayed costs COST_OFFER).
The threshold is chosen on the training split's cross-validated predictions, then scored once on the
held-out split, so the test data does not pick it.
"""

import json
import os

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .data import FEATURES, ROOT, load, split
from .train import CANDIDATES, make_pipeline

COST_MISSED = float(os.getenv("COST_MISSED", "300"))  # lost customer lifetime value (assumption)
COST_OFFER = float(os.getenv("COST_OFFER", "30"))  # discount or call for a customer who stays anyway


def expected_cost(y, p, thr):
    yhat = p >= thr
    missed = int(((y == 1) & ~yhat).sum())
    wasted = int(((y == 0) & yhat).sum())
    return missed * COST_MISSED + wasted * COST_OFFER


def bootstrap_auc_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def main():
    tr, te = split(load())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {"cv_folds": 5, "models": {}}
    for name, mk in CANDIDATES.items():
        pipe = make_pipeline(mk())
        res = cross_validate(pipe, tr[FEATURES], tr["y"], cv=cv, scoring=["roc_auc", "f1", "precision", "recall"])
        pipe.fit(tr[FEATURES], tr["y"])
        p_te = pipe.predict_proba(te[FEATURES])[:, 1]
        lo, hi = bootstrap_auc_ci(te["y"], p_te)
        out["models"][name] = {
            **{m: {"mean": round(float(res[f"test_{m}"].mean()), 4), "std": round(float(res[f"test_{m}"].std()), 4)} for m in ("roc_auc", "f1", "precision", "recall")},
            "test_roc_auc": round(float(roc_auc_score(te["y"], p_te)), 4),
            "test_roc_auc_ci95": [lo, hi],
        }
    served = make_pipeline(CANDIDATES["logistic_regression"]())
    p_cv = cross_val_predict(served, tr[FEATURES], tr["y"], cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    best = min(grid, key=lambda t: expected_cost(tr["y"].to_numpy(), p_cv, t))
    served.fit(tr[FEATURES], tr["y"])
    p_te = served.predict_proba(te[FEATURES])[:, 1]
    y_te = te["y"].to_numpy()
    sweep = []
    for t in grid:
        yhat = (p_te >= t).astype(int)
        sweep.append({"threshold": float(t), "precision": round(float(precision_score(y_te, yhat, zero_division=0)), 3),
                      "recall": round(float(recall_score(y_te, yhat)), 3), "f1": round(float(f1_score(y_te, yhat)), 3),
                      "expected_cost": expected_cost(y_te, p_te, t)})
    out["threshold"] = {"assumed_cost_missed_churner": COST_MISSED, "assumed_cost_unneeded_offer": COST_OFFER,
                        "chosen_on_train_cv": float(best), "test_cost_at_chosen": expected_cost(y_te, p_te, best),
                        "test_cost_at_0.5": expected_cost(y_te, p_te, 0.5), "sweep_on_test": sweep}
    path = os.path.join(ROOT, "reports", "validation.json")
    json.dump(out, open(path, "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
