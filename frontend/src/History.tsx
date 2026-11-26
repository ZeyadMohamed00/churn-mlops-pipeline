import { useEffect, useState } from "react";
import { API_BASE } from "./api";
import { authHeaders, getToken } from "./auth";

type Row = { id: number; features: Record<string, number | string>; churn_probability: number; will_churn: boolean; model_version: string; created_at: string | null };

/** Saved predictions for the signed-in account (stored in PostgreSQL). Hidden for guests. */
export function History() {
  const [rows, setRows] = useState<Row[]>([]);
  const [err, setErr] = useState("");
  const load = async () => {
    try {
      const res = await fetch(API_BASE + "/history", { headers: authHeaders() });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      setRows(await res.json());
      setErr("");
    } catch (e) { setErr((e as Error).message); }
  };
  useEffect(() => { if (getToken()) void load(); }, []);
  if (!getToken()) return null;
  const clear = async () => { await fetch(API_BASE + "/history", { method: "DELETE", headers: authHeaders() }); void load(); };
  return (
    <section aria-label="Saved predictions" style={{ maxWidth: 960, margin: "24px auto", padding: "0 16px" }}>
      <h2>My saved predictions</h2>
      <p style={{ opacity: 0.75, fontSize: 13 }}>Every prediction you run while logged in is saved to your account.</p>
      <button onClick={load}>Refresh</button> <button onClick={clear}>Clear history</button>
      {err && <p role="alert">{err}</p>}
      {rows.length === 0 ? <p>No saved predictions yet. Run one above.</p> : (
        <table style={{ width: "100%", fontSize: 13, marginTop: 8 }}>
          <thead><tr><th align="left">When</th><th align="left">Tenure</th><th align="left">Monthly</th><th align="left">Contract</th><th align="left">Churn risk</th></tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.id}><td>{r.created_at?.slice(0, 16).replace("T", " ")}</td><td>{String(r.features.tenure)}</td><td>{String(r.features.MonthlyCharges)}</td><td>{String(r.features.Contract)}</td><td>{(r.churn_probability * 100).toFixed(1)}% {r.will_churn ? "(likely to churn)" : ""}</td></tr>
          ))}</tbody>
        </table>
      )}
    </section>
  );
}
