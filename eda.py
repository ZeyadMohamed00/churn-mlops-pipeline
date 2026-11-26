"""Exploratory data analysis: writes eda/eda_report.md and charts in eda/."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from churn.data import load, NUM, CAT, ROOT

df = load(); out = os.path.join(ROOT, "eda"); os.makedirs(out, exist_ok=True)
lines = ["# EDA: IBM Telco Customer Churn", "", f"- Rows: {len(df)}, columns: {df.shape[1]-3} features + target", f"- Churn rate: {df.y.mean():.1%} (imbalanced: accuracy alone is misleading)",
         f"- Missing: TotalCharges blank for {(df.tenure==0).sum()} rows with tenure 0 (set to 0)", "", "## Numeric summary", "", df[NUM].describe().round(2).to_markdown(), "", "## Churn rate by category", ""]
for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling"]:
    t = df.groupby(c)["y"].agg(["mean", "count"]).round(3).rename(columns={"mean": "churn_rate", "count": "n"})
    lines += [f"### {c}", "", t.to_markdown(), ""]
lines += ["## Correlation of numeric features with churn", "", df[NUM + ["y"]].corr()["y"].drop("y").round(3).to_markdown(), ""]
open(os.path.join(out, "eda_report.md"), "w").write("\n".join(lines))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
df.groupby("Contract")["y"].mean().plot.bar(ax=ax[0], title="Churn rate by contract", rot=0, color="#4c78a8")
for v, col in [(0, "#4c78a8"), (1, "#e45756")]:
    ax[1].hist(df[df.y == v]["tenure"], bins=24, alpha=.6, label=["stayed", "churned"][v], color=col)
ax[1].set_title("Tenure (months)"); ax[1].legend()
for v, col in [(0, "#4c78a8"), (1, "#e45756")]:
    ax[2].hist(df[df.y == v]["MonthlyCharges"], bins=24, alpha=.6, label=["stayed", "churned"][v], color=col)
ax[2].set_title("Monthly charges")
plt.tight_layout(); plt.savefig(os.path.join(out, "eda.png"), dpi=110)
print(open(os.path.join(out, "eda_report.md")).read()[:900])
