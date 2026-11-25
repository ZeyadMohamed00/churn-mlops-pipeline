import numpy as np, pytest
from fastapi.testclient import TestClient
from churn.data import make_data, TARGET
from churn.train import train
from churn.drift import psi, drift_report


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    return d, train(models_dir=d)


def test_data_is_reproducible_and_balanced_enough():
    a, b = make_data(500, seed=1), make_data(500, seed=1)
    assert a.equals(b)
    assert 0.15 < a[TARGET].mean() < 0.85


def test_training_registers_versioned_artifacts(trained):
    d, meta = trained
    assert (d / meta["version"] / "model.joblib").exists()
    assert (d / "latest.txt").read_text() == meta["version"]
    assert meta["test_metrics"]["roc_auc"] > 0.7


def test_quality_gate_blocks_bad_model(tmp_path):
    with pytest.raises(RuntimeError):
        train(models_dir=tmp_path, min_auc=0.99)
    assert not (tmp_path / "latest.txt").exists()


def test_psi_zero_for_same_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    counts, edges = np.histogram(ref, bins=10)
    assert psi(counts, edges, rng.normal(0, 1, 5000)) < 0.05
    assert psi(counts, edges, rng.normal(2, 1, 5000)) > 0.25


def test_drift_report_flags_shifted_feature(trained):
    _, meta = trained
    df = make_data(2000, seed=5)
    assert drift_report(meta, df)["monthly_charge"]["level"] == "stable"
    df["monthly_charge"] += 60
    assert drift_report(meta, df)["monthly_charge"]["level"] == "major"


def test_api(trained, monkeypatch):
    d, meta = trained
    import churn.serve as serve
    monkeypatch.setattr(serve, "MODELS_DIR", d)
    with TestClient(serve.app) as c:
        assert c.get("/health").json()["model_loaded"] is True
        risky = {"tenure_months": 2, "monthly_charge": 110, "support_calls": 6, "contract": "monthly", "late_payments": 3}
        safe = {"tenure_months": 60, "monthly_charge": 40, "support_calls": 0, "contract": "two_year", "late_payments": 0}
        pr, ps = c.post("/predict", json=risky).json(), c.post("/predict", json=safe).json()
        assert pr["churn_probability"] > ps["churn_probability"]
        assert pr["model_version"] == meta["version"]
        assert c.post("/predict", json={**safe, "contract": "weekly"}).status_code == 422
        assert c.post("/predict", json={**safe, "tenure_months": -1}).status_code == 422
