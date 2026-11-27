import json, os
import joblib, numpy as np, pytest
from fastapi.testclient import TestClient
from churn import serve
from churn.data import load, split, simulate_shift, FEATURES, ROOT
from churn.serve import NumpyModel, current_dir

D = current_dir()


def test_data_loads_and_is_clean():
    df = load()
    assert len(df) == 7043 and df["TotalCharges"].isna().sum() == 0 and 0.2 < df["y"].mean() < 0.3


def test_split_is_stratified_and_reproducible():
    a, b = split(load()), split(load())
    assert a[1]["customerID"].tolist() == b[1]["customerID"].tolist()
    assert abs(a[0]["y"].mean() - a[1]["y"].mean()) < 0.01


def test_numpy_export_matches_sklearn_on_whole_test_set():
    """The served numpy model must give the same probabilities as the trained sklearn pipeline."""
    _, te = split(load())
    sk = joblib.load(os.path.join(D, "model.joblib")).predict_proba(te[FEATURES])[:, 1]
    nm = NumpyModel(os.path.join(D, "model.json"))
    mine = np.array([nm.predict_proba(r) for r in te[FEATURES].to_dict("records")])
    assert np.abs(sk - mine).max() < 1e-9


def test_reported_metrics_reproduce():
    meta = json.load(open(os.path.join(D, "metadata.json")))
    _, te = split(load())
    nm = NumpyModel(os.path.join(D, "model.json"))
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(te["y"], [nm.predict_proba(r) for r in te[FEATURES].to_dict("records")])
    assert auc == pytest.approx(meta["candidates"]["logistic_regression"]["roc_auc"], abs=1e-3)


c = TestClient(serve.app)
HIGH = dict(tenure=2, MonthlyCharges=95.0, TotalCharges=190.0, SeniorCitizen=0)
LOW = dict(tenure=60, MonthlyCharges=30.0, TotalCharges=1800.0, SeniorCitizen=0, Contract="Two year", InternetService="DSL", PaymentMethod="Bank transfer (automatic)", PaperlessBilling="No")


def test_api_predict_orders_risk_sensibly():
    hi, lo = c.post("/predict", json=HIGH).json(), c.post("/predict", json=LOW).json()
    assert hi["churn_probability"] > 0.5 > lo["churn_probability"] and hi["will_churn"] and not lo["will_churn"]


def test_api_validation_and_unknown_category_tolerated():
    assert c.post("/predict", json={**HIGH, "tenure": -1}).status_code == 422
    assert c.post("/predict", json={**HIGH, "Contract": "Three year"}).status_code == 200


def test_api_model_stats_health():
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/model").json()["served_model"] == "logistic_regression"
    c.post("/predict", json=HIGH)
    assert c.get("/stats").json()["inference_p95_ms"] is not None


def test_simulated_shift_changes_distributions():
    _, te = split(load())
    s = simulate_shift(te)
    assert s["MonthlyCharges"].mean() > te["MonthlyCharges"].mean() * 1.25 and s["tenure"].mean() < te["tenure"].mean()


def test_drift_detector_flags_shift_not_holdout():
    from churn.drift import drift_share
    tr, te = split(load())
    assert drift_share(tr, te)["share_drifted"] == 0.0
    assert drift_share(tr, simulate_shift(te))["drifted_columns"] >= 3


def test_bootstrap_ci_brackets_the_point_estimate():
    import numpy as np

    from churn.validate import bootstrap_auc_ci, expected_cost

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    lo, hi = bootstrap_auc_ci(y, p, n=200)
    from sklearn.metrics import roc_auc_score

    assert lo < roc_auc_score(y, p) < hi
    # flagging everyone costs only unneeded offers; flagging nobody costs only missed churners
    assert expected_cost(y, p, 0.0) == int((y == 0).sum()) * 30.0
    assert expected_cost(y, p, 1.01) == int((y == 1).sum()) * 300.0
