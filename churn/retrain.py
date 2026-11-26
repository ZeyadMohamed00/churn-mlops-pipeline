"""Drift-triggered retraining (demonstration of the mechanism on SIMULATED drift).
1. split the holdout in two halves: A = 'new production batch', B = 'later evaluation batch', both with simulated shift
2. measure drift of A vs the training data with Evidently
3. if share of drifted columns > threshold: retrain on train + A and compare old vs new model on B
Labels of simulated rows are the original labels, so the metric change shows the mechanism, not a real-world gain.
Run: python -m churn.retrain"""
import json, os
import joblib, pandas as pd
from .data import load, split, simulate_shift, FEATURES, ROOT
from .drift import drift_share, DRIFT_SHARE_THRESHOLD
from .train import main as train_main, metrics


def run():
    tr, te = split(load())
    A, B = te.iloc[: len(te) // 2].reset_index(drop=True), te.iloc[len(te) // 2:].reset_index(drop=True)
    A, B = simulate_shift(A, 1), simulate_shift(B, 2)
    d = drift_share(tr, A)
    out = {"drift": d, "threshold_share": DRIFT_SHARE_THRESHOLD, "retrained": False}
    old_dir = os.path.join(ROOT, "models", open(os.path.join(ROOT, "models", "latest.txt")).read().strip())
    old = joblib.load(os.path.join(old_dir, "model.joblib"))
    out["old_model_on_B"] = {k: round(v, 4) for k, v in metrics(old, B[FEATURES], B["y"]).items()}
    if d["share_drifted"] > DRIFT_SHARE_THRESHOLD:
        latest = os.path.join(ROOT, "models", "latest.txt"); prev = open(latest).read()
        meta, fitted = train_main(pd.concat([tr, A], ignore_index=True), B, tag="retrain-after-drift")
        if not os.getenv("PROMOTE"):  # a candidate is NOT served until promoted explicitly (PROMOTE=1)
            open(latest, "w").write(prev)
        out.update(promoted=bool(os.getenv("PROMOTE")), retrained=True, new_model_version=meta["version"], previous_model_version=os.path.basename(old_dir),
                   new_model_on_B=meta["candidates"]["logistic_regression"])
    json.dump(out, open(os.path.join(ROOT, "reports", "retrain.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
