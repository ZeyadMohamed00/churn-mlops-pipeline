import { useEffect, useState } from "react";
import { BASE, getJson, predict, type Customer, type Prediction } from "./api";

const HIGH: Customer = { tenure: 2, MonthlyCharges: 95, TotalCharges: 190, SeniorCitizen: 0, InternetService: "Fiber optic", Contract: "Month-to-month", PaymentMethod: "Electronic check", PaperlessBilling: "Yes", TechSupport: "No" };
const LOW: Customer = { tenure: 60, MonthlyCharges: 30, TotalCharges: 1800, SeniorCitizen: 0, InternetService: "DSL", Contract: "Two year", PaymentMethod: "Bank transfer (automatic)", PaperlessBilling: "No", TechSupport: "Yes" };

type Cand = { precision: number; recall: number; f1: number; roc_auc: number; sklearn_latency: { p50_ms: number; p95_ms: number } };
type Meta = { version: string; train_rows: number; test_rows: number; served_model: string; candidates: Record<string, Cand> };
type Drift = { share_drifted: number; drifted_columns: number; n_columns: number; retrain_triggered: boolean; note?: string };
type Summary = { threshold_share: number; holdout_vs_train: Drift; simulated_shift_vs_train: Drift };

function Predict() {
  const [c, setC] = useState<Customer>(HIGH);
  const [p, setP] = useState<Prediction | null>(null);
  const [err, setErr] = useState("");
  const set = <K extends keyof Customer>(k: K, v: Customer[K]) => setC({ ...c, [k]: v });
  const num = (k: "tenure" | "MonthlyCharges" | "TotalCharges") => (
    <label>{k}<input aria-label={k} type="number" value={c[k]} onChange={(e) => set(k, Number(e.target.value))} /></label>
  );
  const run = async () => { setErr(""); try { setP(await predict(c)); } catch (e) { setErr((e as Error).message); } };
  return (
    <section>
      <p className="muted">Example customers (synthetic inputs). Other fields use defaults.</p>
      <button onClick={() => { setC(HIGH); setP(null); }}>Load high-risk example</button>
      <button onClick={() => { setC(LOW); setP(null); }}>Load low-risk example</button>
      <div className="form">
        {num("tenure")}{num("MonthlyCharges")}{num("TotalCharges")}
        <label>Contract<select aria-label="Contract" value={c.Contract} onChange={(e) => set("Contract", e.target.value)}>
          <option>Month-to-month</option><option>One year</option><option>Two year</option></select></label>
        <label>InternetService<select aria-label="InternetService" value={c.InternetService} onChange={(e) => set("InternetService", e.target.value)}>
          <option>DSL</option><option>Fiber optic</option><option>No</option></select></label>
        <label>TechSupport<select aria-label="TechSupport" value={c.TechSupport} onChange={(e) => set("TechSupport", e.target.value)}>
          <option>Yes</option><option>No</option><option>No internet service</option></select></label>
      </div>
      <button onClick={run}>Predict churn</button>
      {err && <p role="alert" className="err">{err}</p>}
      {p && <div className={"result " + (p.will_churn ? "bad" : "good")}>
        <strong>{(p.churn_probability * 100).toFixed(1)}% churn probability</strong> ({p.will_churn ? "likely to churn" : "likely to stay"} at threshold {p.threshold})
        <div className="muted">model {p.model_version}, inference {p.inference_ms} ms</div></div>}
    </section>
  );
}

function Model() {
  const [m, setM] = useState<Meta | null>(null);
  useEffect(() => { getJson<Meta>("model.json").then(setM).catch(() => undefined); }, []);
  if (!m) return <p>Loading...</p>;
  return (
    <section>
      <p>Held-out test set: {m.test_rows} customers (trained on {m.train_rows}). Served model: <strong>{m.served_model}</strong> (version {m.version}).</p>
      <table><thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Predict latency p50/p95 (sklearn, ms)</th></tr></thead>
        <tbody>{Object.entries(m.candidates).map(([k, v]) => (
          <tr key={k}><td>{k}</td><td>{v.precision}</td><td>{v.recall}</td><td>{v.f1}</td><td>{v.roc_auc}</td><td>{v.sklearn_latency.p50_ms} / {v.sklearn_latency.p95_ms}</td></tr>))}</tbody></table>
      <p className="muted">Threshold 0.5. Churn is about 26.5% of customers, so recall matters more than accuracy.</p>
      <img src={BASE + "eda.png"} alt="EDA charts" style={{ maxWidth: "100%" }} />
    </section>
  );
}

function Monitoring() {
  const [s, setS] = useState<Summary | null>(null);
  useEffect(() => { getJson<Summary>("summary.json").then(setS).catch(() => undefined); }, []);
  if (!s) return <p>Loading...</p>;
  const row = (name: string, d: Drift, href: string) => (
    <tr><td>{name}</td><td>{d.drifted_columns} / {d.n_columns} ({(d.share_drifted * 100).toFixed(1)}%)</td><td>{d.retrain_triggered ? "YES" : "no"}</td><td><a href={BASE + href} target="_blank" rel="noreferrer">Evidently report</a></td></tr>
  );
  return (
    <section>
      <p>Evidently data-drift check against the training data. Retrain trigger: more than {(s.threshold_share * 100).toFixed(0)}% of columns drifted.</p>
      <table><thead><tr><th>Batch</th><th>Drifted columns</th><th>Retrain triggered</th><th>Report</th></tr></thead><tbody>
        {row("Holdout vs training (no shift expected)", s.holdout_vs_train, "drift_holdout.html")}
        {row("SIMULATED shift vs training", s.simulated_shift_vs_train, "drift_simulated_shift.html")}</tbody></table>
      <p className="muted">{s.simulated_shift_vs_train.note}</p>
    </section>
  );
}

export function App() {
  const [tab, setTab] = useState<"predict" | "model" | "monitor">("predict");
  return (
    <div className="wrap">
      <h1>Customer churn: MLOps pipeline demo</h1>
      <p className="muted">Public IBM Telco churn dataset (7,043 customers). Predictions come from a logistic-regression model served by a FastAPI function.</p>
      <nav>{(["predict", "model", "monitor"] as const).map((t) => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{{ predict: "Predict", model: "Model & metrics", monitor: "Drift monitoring" }[t]}</button>)}</nav>
      {tab === "predict" ? <Predict /> : tab === "model" ? <Model /> : <Monitoring />}
    </div>
  );
}
