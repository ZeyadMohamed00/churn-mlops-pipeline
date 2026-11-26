import uuid

from fastapi.testclient import TestClient

from churn.serve import app

c = TestClient(app)
CUSTOMER = {"tenure": 3, "MonthlyCharges": 85.0, "TotalCharges": 255.0, "SeniorCitizen": 0}


def new_user():
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    r = c.post("/auth/register", json={"email": email, "password": "correct-horse-1", "name": "Ann"})
    assert r.status_code == 201, r.text
    return email, {"Authorization": "Bearer " + r.json()["access_token"]}


def test_register_login_me_and_errors():
    email, h = new_user()
    assert c.get("/auth/me", headers=h).json()["email"] == email
    assert c.post("/auth/login", json={"email": email, "password": "correct-horse-1"}).status_code == 200
    assert c.post("/auth/login", json={"email": email, "password": "wrong-password"}).status_code == 401
    assert c.post("/auth/register", json={"email": email, "password": "correct-horse-1"}).status_code == 409
    assert c.get("/history", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_predictions_are_saved_per_account_and_guests_are_not_saved():
    _, a = new_user()
    _, b = new_user()
    assert c.post("/predict", json=CUSTOMER, headers=a).status_code == 200
    assert c.post("/predict", json=CUSTOMER).status_code == 200  # guest: works, not stored
    ha, hb = c.get("/history", headers=a).json(), c.get("/history", headers=b).json()
    assert len(ha) == 1 and ha[0]["features"]["tenure"] == 3 and hb == []
    assert c.get("/history").status_code in (400, 401)
    assert c.delete("/history", headers=a).status_code == 204
    assert c.get("/history", headers=a).json() == []
