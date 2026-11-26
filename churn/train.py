"""Train + compare models with MLflow tracking; export the served model (numpy-only inference) and metrics.
Run: python -m churn.train"""
import json, os, time, hashlib
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import load, split, NUM, CAT, FEATURES, ROOT

MODELS = os.path.join(ROOT, "models")


def make_pipeline(clf):
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])


CANDIDATES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, C=1.0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
}


def metrics(pipe, X, y, threshold=0.5):
    p = pipe.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)
    return {"precision": precision_score(y, yhat), "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p)}


def latency_ms(fn, n=500):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3)}


def export_numpy(pipe: Pipeline, path: str):
    """Export a fitted logistic-regression pipeline as plain JSON so serving needs only numpy."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    sc = pre.named_transformers_["num"]; oh = pre.named_transformers_["cat"]
    cats = {}
    for name, cs, drop in zip(CAT, oh.categories_, oh.drop_idx_):
        cats[name] = {"values": [str(c) for c in cs], "drop_idx": None if drop is None else int(drop)}
    json.dump({"num": NUM, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "cat": CAT, "categories": cats,
               "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}, open(path, "w"))


def main(train_df=None, test_df=None, tag="initial"):
    df = load()
    tr, te = (train_df, test_df) if train_df is not None else split(df)
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(ROOT, "mlflow.db"))
    mlflow.set_experiment("churn")
    results, fitted = {}, {}
    for name, mk in CANDIDATES.items():
        with mlflow.start_run(run_name=f"{tag}-{name}"):
            pipe = make_pipeline(mk()); t0 = time.time()
            pipe.fit(tr[FEATURES], tr["y"]); fit_s = time.time() - t0
            m = metrics(pipe, te[FEATURES], te["y"])
            mlflow.log_params({"model": name, "train_rows": len(tr), "test_rows": len(te), "data_tag": tag, **{f"p_{k}": v for k, v in pipe.named_steps["clf"].get_params().items() if isinstance(v, (int, float, str))}})
            mlflow.log_metrics({**m, "fit_seconds": fit_s})
            row = te[FEATURES].iloc[[0]]
            lat = latency_ms(lambda: pipe.predict_proba(row))
            mlflow.log_metrics({"sklearn_single_predict_p50_ms": lat["p50_ms"], "sklearn_single_predict_p95_ms": lat["p95_ms"]})
            results[name] = {**{k: round(v, 4) for k, v in m.items()}, "fit_seconds": round(fit_s, 2), "sklearn_latency": lat}
            fitted[name] = pipe
    # The served model is logistic regression: competitive here, interpretable, and exportable to numpy-only inference.
    version = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.md5(str(results).encode()).hexdigest()[:6]
    vdir = os.path.join(MODELS, version); os.makedirs(vdir)
    joblib.dump(fitted["logistic_regression"], os.path.join(vdir, "model.joblib"))
    export_numpy(fitted["logistic_regression"], os.path.join(vdir, "model.json"))
    meta = {"version": version, "served_model": "logistic_regression", "data_tag": tag, "train_rows": len(tr), "test_rows": len(te),
            "threshold": 0.5, "candidates": results, "churn_rate_train": round(float(tr["y"].mean()), 4)}
    json.dump(meta, open(os.path.join(vdir, "metadata.json"), "w"), indent=1)
    open(os.path.join(MODELS, "latest.txt"), "w").write(version)
    return meta, fitted


if __name__ == "__main__":
    meta, _ = main()
    print(json.dumps(meta, indent=1))
