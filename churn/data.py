"""SYNTHETIC telecom-style churn data. It is generated from a made-up rule with noise, so model scores
say nothing about real-world churn. It exists so the pipeline can run end to end with no download."""
import numpy as np
import pandas as pd

FEATURES = ["tenure_months", "monthly_charge", "support_calls", "contract", "late_payments"]
NUMERIC = ["tenure_months", "monthly_charge", "support_calls", "late_payments"]
CATEGORICAL = ["contract"]
TARGET = "churned"
CONTRACTS = ["monthly", "yearly", "two_year"]


def make_data(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charge": rng.normal(65, 20, n).clip(15, 150).round(2),
        "support_calls": rng.poisson(1.5, n),
        "contract": rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20]),
        "late_payments": rng.poisson(0.7, n),
    })
    logit = (-0.8 - 0.04 * df.tenure_months + 0.02 * (df.monthly_charge - 65)
             + 0.35 * df.support_calls + 0.4 * df.late_payments
             + df.contract.map({"monthly": 0.9, "yearly": -0.2, "two_year": -0.9}))
    p = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df
