import { useEffect, useState } from "react";
import { BASE, FIELDS_DEFAULT, getJson, predict, type Customer, type Prediction } from "./api";
import { Badge, BarChart, Empty, Shell, Skeleton, Stat, ToastProvider, useToast } from "./ui";

const HIGH: Customer = { tenure: 2, MonthlyCharges: 95, TotalCharges: 190, SeniorCitizen: 0, InternetService: "Fiber optic", Contract: "Month-to-month", PaymentMethod: "Electronic check", PaperlessBilling: "Yes", TechSupport: "No" };
const LOW: Customer = { tenure: 60, MonthlyCharges: 30, TotalCharges: 1800, SeniorCitizen: 0, InternetService: "DSL", Contract: "Two year", PaymentMethod: "Bank transfer (automatic)", PaperlessBilling: "No", TechSupport: "Yes" };

type Cand = { precision: number; recall: number; f1: number; roc_auc: number; fit_seconds?: number; sklearn_latency: { p50_ms: number; p95_ms: number } };
type Meta = { version: string; train_rows: number; test_rows: number; served_model: string; threshold: number; churn_rate_train?: number; candidates: Record<string, Cand> };
type Drift = { share_drifted: number; drifted_columns: number; n_columns: number; retrain_triggered: boolean; note?: string };
type Summary = { threshold_share: number; holdout_vs_train: Drift; simulated_shift_vs_train: Drift };
type Retrain = { drift: { share_drifted: number; drifted_columns: number; n_columns: number }; threshold_share: number; retrained: boolean; promoted: boolean; new_model_version: string; previous_model_version: string; old_model_on_B: Record<string, number>; new_model_on_B: Record<string, number> };
type Page = "predict" | "batch" | "model" | "drift" | "retrain" | "pipeline";

export function App() { return <ToastProvider><Main /></ToastProvider>; }

const risk = (p: number) => (p >= 0.5 ? { label: "High risk", kind: "bad" as const } : p >= 0.3 ? { label: "Medium risk", kind: "warn" as const } : { label: "Low risk", kind: "ok" as const });

function Main() {
  const [page, setPage] = useState<Page>("predict");
  return (
    <Shell brand="Churn Lab" tagline="End-to-end MLOps demo" page={page} onNav={(p) => setPage(p as Page)}
      nav={[{ id: "predict", label: "Predict", icon: "◎" }, { id: "batch", label: "Batch scoring", icon: "▤" }, { id: "model", label: "Model comparison", icon: "▦" }, { id: "drift", label: "Drift monitoring", icon: "≈" }, { id: "retrain", label: "Retraining", icon: "↻" }, { id: "pipeline", label: "Pipeline", icon: "⇢" }]}>
      {page === "predict" && <PredictPage />}
      {page === "batch" && <BatchPage />}
      {page === "model" && <ModelPage />}
      {page === "drift" && <DriftPage />}
      {page === "retrain" && <RetrainPage />}
      {page === "pipeline" && <PipelinePage />}
    </Shell>
  );
}

function PredictPage() {
  const toast = useToast();
  const [c, setC] = useState<Customer>(HIGH);
  const [p, setP] = useState<Prediction | null>(null);
  const [whatif, setWhatif] = useState<{ label: string; prob: number }[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const set = <K extends keyof Customer>(k: K, v: Customer[K]) => { setC({ ...c, [k]: v }); };
  const num = (k: "tenure" | "MonthlyCharges" | "TotalCharges", min: number, max: number, step = 1) => (
    <label className="f">{k} <b style={{ color: "var(--text)" }}>{c[k]}</b>
      <input aria-label={k} type="number" min={min} max={max} step={step} value={c[k]} onChange={(e) => set(k, Number(e.target.value))} />
      <input type="range" aria-hidden tabIndex={-1} min={min} max={max} step={step} value={c[k]} onChange={(e) => set(k, Number(e.target.value))} />
    </label>
  );
  const run = async () => {
    setErr(""); setBusy(true); setWhatif([]);
    try {
      const base = await predict(c); setP(base);
      const variants: [string, Partial<Customer>][] = [];
      if (c.Contract === "Month-to-month") { variants.push(["Switch to a one-year contract", { Contract: "One year" }]); variants.push(["Switch to a two-year contract", { Contract: "Two year" }]); }
      if (c.TechSupport === "No") variants.push(["Add tech support", { TechSupport: "Yes" }]);
      if (c.PaymentMethod === "Electronic check") variants.push(["Pay by bank transfer", { PaymentMethod: "Bank transfer (automatic)" }]);
      const outs = await Promise.all(variants.map(async ([label, v]) => ({ label, prob: (await predict({ ...c, ...v })).churn_probability })));
      setWhatif(outs.sort((a, b) => a.prob - b.prob));
    } catch (e) { setErr((e as Error).message); toast("Prediction failed", "bad"); } finally { setBusy(false); }
  };
  return (
    <>
      <div className="page-h"><div><h1>Predict customer churn</h1><p>Logistic regression trained on the public IBM Telco dataset (7,043 customers). Inputs below are synthetic.</p></div>
        <div className="sp"><button className="btn" onClick={() => { setC(HIGH); setP(null); setWhatif([]); }}>High-risk example</button><button className="btn" onClick={() => { setC(LOW); setP(null); setWhatif([]); }}>Low-risk example</button></div></div>
      <div className="grid g-main">
        <div className="card"><h3>Customer profile</h3>
          <div className="grid g3">{num("tenure", 0, 72)}{num("MonthlyCharges", 18, 120, 0.5)}{num("TotalCharges", 0, 9000, 10)}</div>
          <div className="grid g3" style={{ marginTop: 14 }}>
            <label className="f">Contract<select aria-label="Contract" value={c.Contract} onChange={(e) => set("Contract", e.target.value)}><option>Month-to-month</option><option>One year</option><option>Two year</option></select></label>
            <label className="f">InternetService<select aria-label="InternetService" value={c.InternetService} onChange={(e) => set("InternetService", e.target.value)}><option>DSL</option><option>Fiber optic</option><option>No</option></select></label>
            <label className="f">TechSupport<select aria-label="TechSupport" value={c.TechSupport} onChange={(e) => set("TechSupport", e.target.value)}><option>Yes</option><option>No</option><option>No internet service</option></select></label>
            <label className="f">PaymentMethod<select aria-label="PaymentMethod" value={c.PaymentMethod} onChange={(e) => set("PaymentMethod", e.target.value)}><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></label>
            <label className="f">PaperlessBilling<select aria-label="PaperlessBilling" value={c.PaperlessBilling} onChange={(e) => set("PaperlessBilling", e.target.value)}><option>Yes</option><option>No</option></select></label>
            <label className="f">SeniorCitizen<select aria-label="SeniorCitizen" value={c.SeniorCitizen} onChange={(e) => set("SeniorCitizen", Number(e.target.value))}><option value={0}>No</option><option value={1}>Yes</option></select></label>
          </div>
          <div className="row" style={{ marginTop: 16 }}><button className="btn primary" onClick={run} disabled={busy}>{busy ? "Scoring..." : "Predict churn"}</button></div>
          {err && <p role="alert" className="err">{err}</p>}
        </div>
        <div className="col">
          <div className="card">{!p ? <Empty icon="◎" title="No prediction yet" hint="Fill the profile and press Predict churn." /> : (
            <div className="col"><span className="muted small">CHURN PROBABILITY</span><div className="big">{(p.churn_probability * 100).toFixed(1)}%</div>
              <div><Badge kind={risk(p.churn_probability).kind}>{risk(p.churn_probability).label}</Badge> <span className="muted small">{p.will_churn ? "likely to churn" : "likely to stay"} at threshold {p.threshold}</span></div>
              <div className="gauge" aria-hidden><i style={{ left: `calc(${Math.min(100, p.churn_probability * 100)}% - 2px)` }} /></div>
              <div className="row small muted" style={{ justifyContent: "space-between" }}><span>0%</span><span>50%</span><span>100%</span></div>
              <span className="muted small">model {p.model_version}, inference {p.inference_ms} ms</span></div>)}</div>
          {p && whatif.length > 0 && <div className="card"><h3>What-if analysis</h3><p className="muted small" style={{ marginBottom: 8 }}>Same customer, one change at a time (each is a real call to the model).</p>
            <div className="list">{whatif.map((w) => { const d = (w.prob - p.churn_probability) * 100; return <div key={w.label} className="row"><span>{w.label}</span><span className="sp"><Badge kind={d < 0 ? "ok" : "bad"}>{d < 0 ? "" : "+"}{d.toFixed(1)} pts</Badge> <b>{(w.prob * 100).toFixed(1)}%</b></span></div>; })}</div></div>}
        </div>
      </div>
    </>
  );
}

const CSV_SAMPLE = "tenure,MonthlyCharges,TotalCharges,Contract,InternetService,TechSupport\n2,95,190,Month-to-month,Fiber optic,No\n60,30,1800,Two year,DSL,Yes\n24,70,1680,One year,Fiber optic,Yes";
function BatchPage() {
  const [text, setText] = useState(CSV_SAMPLE);
  const [rows, setRows] = useState<{ row: Customer; prob: number }[]>([]);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState("");
  const parse = (): Customer[] => {
    const [h, ...lines] = text.trim().split(/\r?\n/); const cols = h.split(",").map((x) => x.trim());
    return lines.filter(Boolean).slice(0, 25).map((l) => { const v = l.split(","); const o: Record<string, string | number> = { ...FIELDS_DEFAULT }; cols.forEach((k, i) => { const x = (v[i] ?? "").trim(); o[k] = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"].includes(k) ? Number(x) : x; }); return o as unknown as Customer; });
  };
  const run = async () => { setErr(""); setBusy(true); try { const cs = parse(); setRows(await Promise.all(cs.map(async (row) => ({ row, prob: (await predict(row)).churn_probability })))); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); } };
  const download = () => { const csv = "tenure,MonthlyCharges,Contract,churn_probability\n" + rows.map((r) => `${r.row.tenure},${r.row.MonthlyCharges},${r.row.Contract},${r.prob}`).join("\n"); const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "churn_scores.csv"; a.click(); };
  return (
    <>
      <div className="page-h"><div><h1>Batch scoring</h1><p>Paste CSV rows (up to 25). Missing columns use defaults. Each row is scored by the live API.</p></div></div>
      <div className="card col"><textarea aria-label="CSV" style={{ minHeight: 140 }} className="mono" value={text} onChange={(e) => setText(e.target.value)} /><div className="row"><button className="btn primary" onClick={run} disabled={busy}>{busy ? "Scoring..." : "Score rows"}</button>{rows.length > 0 && <button className="btn" onClick={download}>Download CSV</button>}</div>{err && <p role="alert" className="err">{err}</p>}</div>
      {rows.length > 0 && <div className="card tw"><table className="t"><thead><tr><th>Tenure</th><th>Monthly</th><th>Contract</th><th>Churn probability</th><th>Risk</th></tr></thead><tbody>{rows.map((r, i) => <tr key={i}><td>{r.row.tenure}</td><td>${r.row.MonthlyCharges}</td><td>{r.row.Contract}</td><td><b>{(r.prob * 100).toFixed(1)}%</b></td><td><Badge kind={risk(r.prob).kind}>{risk(r.prob).label}</Badge></td></tr>)}</tbody></table></div>}
    </>
  );
}

function useJson<T>(name: string) { const [d, setD] = useState<T | null>(null); useEffect(() => { getJson<T>(name).then(setD).catch(() => undefined); }, [name]); return d; }

function ModelPage() {
  const m = useJson<Meta>("model.json");
  if (!m) return <Skeleton lines={5} />;
  const entries = Object.entries(m.candidates);
  return (
    <>
      <div className="page-h"><div><h1>Model comparison</h1><p>Held-out test set: {m.test_rows} customers (trained on {m.train_rows}). Served model: <b>{m.served_model}</b>, version {m.version}.</p></div></div>
      <div className="grid g4">{["precision", "recall", "f1", "roc_auc"].map((k) => <Stat key={k} label={k.replace("_", "-")} value={m.candidates[m.served_model]?.[k as keyof Cand] as number} hint={m.served_model} />)}</div>
      <div className="grid g2">
        <div className="card"><h3>ROC-AUC by model</h3><BarChart max={1} data={entries.map(([k, v]) => ({ label: k, value: v.roc_auc }))} /><div className="row small muted" style={{ justifyContent: "space-around" }}>{entries.map(([k]) => <span key={k}>{k}</span>)}</div></div>
        <div className="card"><h3>Recall by model</h3><BarChart max={1} data={entries.map(([k, v]) => ({ label: k, value: v.recall }))} color="var(--accent-2)" /><div className="row small muted" style={{ justifyContent: "space-around" }}>{entries.map(([k]) => <span key={k}>{k}</span>)}</div></div>
      </div>
      <div className="card tw"><table className="t"><thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Predict p50 / p95 (ms)</th></tr></thead><tbody>
        {entries.map(([k, v]) => <tr key={k}><td><b>{k}</b> {k === m.served_model && <Badge kind="acc">served</Badge>}</td><td>{v.precision}</td><td>{v.recall}</td><td>{v.f1}</td><td>{v.roc_auc}</td><td>{v.sklearn_latency.p50_ms} / {v.sklearn_latency.p95_ms}</td></tr>)}</tbody></table>
        <p className="muted small" style={{ marginTop: 8 }}>Threshold {m.threshold}. About 26.5% of customers churn, so recall matters more than accuracy. The live API serves logistic regression via exported numpy weights (free serverless limits), verified equal to the sklearn pipeline.</p></div>
    </>
  );
}

function DriftPage() {
  const s = useJson<Summary>("summary.json");
  if (!s) return <Skeleton lines={4} />;
  const card = (title: string, d: Drift, href: string) => (
    <div className="card col"><h3>{title}</h3><div className="big">{(d.share_drifted * 100).toFixed(1)}%</div><span className="muted small">{d.drifted_columns} of {d.n_columns} columns drifted</span>
      <div><Badge kind={d.retrain_triggered ? "bad" : "ok"}>{d.retrain_triggered ? "Retrain triggered" : "No retrain needed"}</Badge></div>
      {d.note && <p className="muted small">{d.note}</p>}<a href={BASE + href} target="_blank" rel="noreferrer">Open Evidently report</a></div>);
  return (<>
    <div className="page-h"><div><h1>Drift monitoring</h1><p>Evidently compares new data against the training data. Retrain when more than {(s.threshold_share * 100).toFixed(0)}% of columns drift.</p></div></div>
    <div className="grid g2">{card("Holdout vs training (no shift expected)", s.holdout_vs_train, "drift_holdout.html")}{card("SIMULATED shift vs training", s.simulated_shift_vs_train, "drift_simulated_shift.html")}</div></>);
}

function RetrainPage() {
  const r = useJson<Retrain>("retrain.json");
  if (!r) return <Skeleton lines={4} />;
  const keys = ["precision", "recall", "f1", "roc_auc"];
  return (<>
    <div className="page-h"><div><h1>Retraining</h1><p>When drift crosses the threshold, the pipeline trains a candidate on the new data and promotes it only if it beats the current model.</p></div></div>
    <div className="grid g3"><Stat label="Drift share" value={`${(r.drift.share_drifted * 100).toFixed(1)}%`} hint={`threshold ${(r.threshold_share * 100).toFixed(0)}%`} /><Stat label="Retrained" value={r.retrained ? "Yes" : "No"} /><Stat label="Promoted" value={r.promoted ? "Yes" : "No"} hint="Candidate did not clear the gate" /></div>
    <div className="card tw"><h3>Old vs new model on the shifted data</h3><table className="t"><thead><tr><th>Metric</th><th>Current ({r.previous_model_version.slice(-6)})</th><th>Candidate ({r.new_model_version.slice(-6)})</th></tr></thead><tbody>{keys.map((k) => <tr key={k}><td>{k}</td><td>{r.old_model_on_B[k]}</td><td>{r.new_model_on_B[k]}</td></tr>)}</tbody></table>
      <p className="muted small" style={{ marginTop: 8 }}>The gate is deliberately strict, so a marginal gain keeps the current model in service. This is a demo of the mechanism, not a claim about production gains.</p></div></>);
}

function PipelinePage() {
  const steps = [["1 Data", "Public IBM Telco churn CSV, cleaned."], ["2 EDA", "Distributions and churn by segment."], ["3 Train", "Logistic regression, random forest, gradient boosting."], ["4 Evaluate", "Precision, recall, F1, ROC-AUC on a held-out set."], ["5 Track", "MLflow runs and a versioned model registry."], ["6 Serve", "FastAPI /predict with latency tracking."], ["7 Monitor", "Evidently drift reports."], ["8 Retrain", "Triggered by drift, gated by metrics."]];
  return (<>
    <div className="page-h"><div><h1>The pipeline</h1><p>From raw data to a monitored, retrainable service.</p></div></div>
    <div className="pipe">{steps.map(([t, d]) => <div className="st" key={t}><b>{t}</b><p className="muted small">{d}</p></div>)}</div>
    <div className="card"><h3>Exploratory data analysis</h3><img src={BASE + "eda.png"} alt="EDA charts" style={{ maxWidth: "100%", borderRadius: 8 }} /></div></>);
}
