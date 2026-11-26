export const API_BASE: string = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "/api" : "");
export const BASE = import.meta.env.BASE_URL;

export type Customer = {
  tenure: number; MonthlyCharges: number; TotalCharges: number; SeniorCitizen: number;
  InternetService: string; Contract: string; PaymentMethod: string; PaperlessBilling: string; TechSupport: string;
};
export type Prediction = { churn_probability: number; will_churn: boolean; threshold: number; model_version: string; inference_ms: number };

export async function predict(c: Customer): Promise<Prediction> {
  const res = await fetch(API_BASE + "/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(c) });
  if (!res.ok) throw new Error((await res.text()).slice(0, 200));
  return res.json();
}

export async function getJson<T>(name: string): Promise<T> {
  const res = await fetch(BASE + name);
  if (!res.ok) throw new Error(`Could not load ${name}`);
  return res.json();
}

export const FIELDS_DEFAULT: Customer = { tenure: 12, MonthlyCharges: 70, TotalCharges: 840, SeniorCitizen: 0, InternetService: "Fiber optic", Contract: "Month-to-month", PaymentMethod: "Electronic check", PaperlessBilling: "Yes", TechSupport: "No" };
