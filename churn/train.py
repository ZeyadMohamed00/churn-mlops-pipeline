"""Train, compare, evaluate and register a model.
python -m churn.train   ->  models/<version>/{model.joblib, metadata.json} and models/latest.txt"""
import json, hashlib, sys, time
from pathlib import Path
import joblib, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import make_data, FEATURES, NUMERIC, CATEGORICAL, TARGET

MODELS_DIR = Path("models")
MIN_AUC = 0.70  # quality gate: a model below this is NOT registered


def build(estimator) -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", estimator)])


def train(df=None, models_dir: Path = MODELS_DIR, seed: int = 0, min_auc: float = MIN_AUC) -> dict:
    df = make_data(seed=seed) if df is None else df
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    candidates = {
        "logreg": build(LogisticRegression(max_iter=1000)),
        "gboost": build(GradientBoostingClassifier(random_state=seed)),
    }
    cv = {n: float(np.mean(cross_val_score(m, X_tr, y_tr, cv=5, scoring="roc_auc"))) for n, m in candidates.items()}
    best = max(cv, key=cv.get)  # choose on CV of the TRAIN split only; test set stays untouched
    model = candidates[best].fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"roc_auc": float(roc_auc_score(y_te, proba)), "precision": float(precision_score(y_te, pred)),
               "recall": float(recall_score(y_te, pred)), "f1": float(f1_score(y_te, pred))}
    meta = {"model": best, "cv_auc": cv, "test_metrics": metrics, "n_train": len(X_tr), "n_test": len(X_te),
            "features": FEATURES, "data": "synthetic", "seed": seed,
            "reference_stats": {c: {"mean": float(X_tr[c].mean()), "std": float(X_tr[c].std())} for c in NUMERIC},
            "reference_hist": {c: np.histogram(X_tr[c], bins=10)[0].tolist() for c in NUMERIC},
            "reference_edges": {c: np.histogram(X_tr[c], bins=10)[1].tolist() for c in NUMERIC}}
    if metrics["roc_auc"] < min_auc:
        raise RuntimeError(f"quality gate failed: AUC {metrics['roc_auc']:.3f} < {min_auc}")
    version = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(json.dumps(metrics).encode()).hexdigest()[:6]
    out = models_dir / version
    out.mkdir(parents=True)
    joblib.dump(model, out / "model.joblib")
    meta["version"] = version
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (models_dir / "latest.txt").write_text(version)
    return meta


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: m[k] for k in ("version", "model", "cv_auc", "test_metrics")}, indent=2))
