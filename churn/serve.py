import json, os
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from .data import FEATURES

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
app = FastAPI(title="Churn model service")
_state: dict = {}


def load():
    version = (MODELS_DIR / "latest.txt").read_text().strip()
    d = MODELS_DIR / version
    _state["model"] = joblib.load(d / "model.joblib")
    _state["meta"] = json.loads((d / "metadata.json").read_text())


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charge: float = Field(ge=0, le=1000)
    support_calls: int = Field(ge=0, le=100)
    contract: Literal["monthly", "yearly", "two_year"]
    late_payments: int = Field(ge=0, le=100)


@app.on_event("startup")
def startup():
    if (MODELS_DIR / "latest.txt").exists():
        load()


@app.get("/health")
def health():
    return {"model_loaded": "model" in _state, "version": _state.get("meta", {}).get("version")}


@app.post("/predict")
def predict(c: Customer):
    if "model" not in _state:
        raise HTTPException(503, "No model loaded. Run `python -m churn.train` first.")
    p = float(_state["model"].predict_proba(pd.DataFrame([c.model_dump()])[FEATURES])[0, 1])
    return {"churn_probability": round(p, 4), "high_risk": p >= 0.5, "model_version": _state["meta"]["version"]}


@app.get("/model")
def model_info():
    if "meta" not in _state:
        raise HTTPException(503, "No model loaded")
    m = _state["meta"]
    return {k: m[k] for k in ("version", "model", "cv_auc", "test_metrics", "data")}
