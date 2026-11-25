"""Population Stability Index per numeric feature: PSI = sum((a-e) * ln(a/e)) over bins.
Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
import numpy as np


def psi(expected_counts, edges, new_values, eps: float = 1e-4) -> float:
    e = np.array(expected_counts, dtype=float)
    e = e / e.sum()
    edges = np.array(edges, dtype=float).copy()
    edges[0], edges[-1] = -np.inf, np.inf  # values outside the training range fall in the end bins
    a = np.histogram(new_values, bins=edges)[0].astype(float)
    a = a / max(a.sum(), 1)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(meta: dict, df) -> dict:
    out = {}
    for c, counts in meta["reference_hist"].items():
        v = psi(counts, meta["reference_edges"][c], df[c].to_numpy())
        out[c] = {"psi": round(v, 4), "level": "major" if v > 0.25 else "moderate" if v > 0.1 else "stable"}
    return out
