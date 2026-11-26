"""FastAPI service. Inference is numpy-only (model.json exported at training time), so the image/function stays small."""
import json, os, time
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import accounts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


def current_dir():
    return os.path.join(MODELS, open(os.path.join(MODELS, "latest.txt")).read().strip())


class NumpyModel:
    def __init__(self, path):
        m = json.load(open(path))
        self.m = m
        self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
        self.coef, self.b = np.array(m["coef"]), m["intercept"]

    def predict_proba(self, row: dict) -> float:
        m = self.m
        x = (np.array([float(row[c]) for c in m["num"]]) - self.mean) / self.scale
        parts = [x]
        for c in m["cat"]:
            spec = m["categories"][c]; v = str(row[c])
            onehot = np.zeros(len(spec["values"]))
            if v in spec["values"]:
                onehot[spec["values"].index(v)] = 1.0  # unknown category -> all zeros (same as handle_unknown="ignore")
            if spec["drop_idx"] is not None:
                onehot = np.delete(onehot, spec["drop_idx"])
            parts.append(onehot)
        z = float(np.concatenate(parts) @ self.coef + self.b)
        return float(1.0 / (1.0 + np.exp(-z)))


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=100)
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0, le=100000)
    SeniorCitizen: int = Field(ge=0, le=1)
    gender: str = "Female"; Partner: str = "No"; Dependents: str = "No"; PhoneService: str = "Yes"
    MultipleLines: str = "No"; InternetService: str = "Fiber optic"; OnlineSecurity: str = "No"; OnlineBackup: str = "No"
    DeviceProtection: str = "No"; TechSupport: str = "No"; StreamingTV: str = "No"; StreamingMovies: str = "No"
    Contract: str = "Month-to-month"; PaperlessBilling: str = "Yes"; PaymentMethod: str = "Electronic check"


def create_app():
    accounts.init()
    d = current_dir()
    model = NumpyModel(os.path.join(d, "model.json"))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    app = FastAPI(title="Churn prediction API", version="2.0.0")
    app.include_router(accounts.router)
    app.include_router(accounts.history_router)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    stats = {"requests": 0, "latencies_ms": []}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": meta["version"]}

    @app.get("/model")
    def model_info():
        return meta

    @app.get("/reports")
    def reports():
        p = os.path.join(ROOT, "reports", "summary.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    @app.post("/predict")
    def predict(c: Customer, authorization: str = Header(default=""), x_client_id: str = Header(default="anonymous-client")):
        owner = accounts.identity(authorization, x_client_id)
        t = time.perf_counter()
        p = model.predict_proba(c.model_dump())
        ms = (time.perf_counter() - t) * 1000
        stats["requests"] += 1; stats["latencies_ms"] = (stats["latencies_ms"] + [ms])[-1000:]
        accounts.save_prediction(owner, c.model_dump(), round(p, 4), bool(p >= meta["threshold"]), meta["version"])
        return {"churn_probability": round(p, 4), "will_churn": bool(p >= meta["threshold"]), "threshold": meta["threshold"],
                "model_version": meta["version"], "inference_ms": round(ms, 3)}

    @app.get("/stats")
    def get_stats():
        l = stats["latencies_ms"]
        return {"requests_since_start": stats["requests"],
                "inference_p50_ms": round(float(np.percentile(l, 50)), 3) if l else None,
                "inference_p95_ms": round(float(np.percentile(l, 95)), 3) if l else None}

    return app


app = create_app()
