import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "telco_churn.csv")  # IBM Telco Customer Churn sample dataset (Apache-2.0), 7,043 rows
NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
       "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM + CAT


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")  # 11 blank values (tenure 0)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["y"] = (df["Churn"] == "Yes").astype(int)
    return df


def split(df: pd.DataFrame, seed: int = 42):
    tr, te = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["y"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def simulate_shift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """SIMULATED drift for demonstrating monitoring: prices up 30%, newer customers, more month-to-month contracts.
    This is synthetic; it is not observed production data."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["MonthlyCharges"] = (d["MonthlyCharges"] * 1.3).round(2)
    d["tenure"] = (d["tenure"] * 0.6).round().astype(int)
    d["TotalCharges"] = (d["tenure"] * d["MonthlyCharges"]).round(2)
    flip = rng.random(len(d)) < 0.35
    d.loc[flip, "Contract"] = "Month-to-month"
    return d
