"""Data-drift monitoring with Evidently: reference = training data, current = new data.
Run: python -m churn.drift   (writes reports/drift_*.html and reports/summary.json)"""
import json, os
from evidently import Report
from evidently.presets import DataDriftPreset
from .data import load, split, simulate_shift, FEATURES, ROOT

DRIFT_SHARE_THRESHOLD = 0.2  # my choice: retrain when more than 20% of columns drift (Evidently default drift_share is 0.5)


def drift_share(reference, current, html_path=None):
    snap = Report([DataDriftPreset()]).run(current[FEATURES], reference[FEATURES])
    if html_path:
        snap.save_html(html_path)
    d = snap.dict()
    share, count = None, None
    for m in d["metrics"]:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            share, count = m["value"]["share"], m["value"]["count"]
    return {"share_drifted": round(float(share), 3), "drifted_columns": int(count), "n_columns": len(FEATURES)}


def main():
    tr, te = split(load())
    out = os.path.join(ROOT, "reports"); os.makedirs(out, exist_ok=True)
    clean = drift_share(tr, te, os.path.join(out, "drift_holdout.html"))
    shifted = drift_share(tr, simulate_shift(te), os.path.join(out, "drift_simulated_shift.html"))
    summary = {"threshold_share": DRIFT_SHARE_THRESHOLD,
               "holdout_vs_train": {**clean, "retrain_triggered": clean["share_drifted"] > DRIFT_SHARE_THRESHOLD},
               "simulated_shift_vs_train": {**shifted, "retrain_triggered": shifted["share_drifted"] > DRIFT_SHARE_THRESHOLD,
                                            "note": "SIMULATED: prices +30%, shorter tenure, more month-to-month. Not real production data."}}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
