import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { App } from "./App";

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const ok = (b: unknown) => new Response(JSON.stringify(b), { status: 200 });
    if (url.endsWith("/predict")) return ok({ churn_probability: 0.73, will_churn: true, threshold: 0.5, model_version: "v1", inference_ms: 0.1 });
    if (url.endsWith("model.json")) return ok({ version: "v1", train_rows: 10, test_rows: 5, served_model: "logistic_regression", candidates: { logistic_regression: { precision: 0.65, recall: 0.56, f1: 0.6, roc_auc: 0.84, sklearn_latency: { p50_ms: 2, p95_ms: 3 } } } });
    if (url.endsWith("summary.json")) return ok({ threshold_share: 0.2, holdout_vs_train: { share_drifted: 0, drifted_columns: 0, n_columns: 19, retrain_triggered: false },
      simulated_shift_vs_train: { share_drifted: 0.21, drifted_columns: 4, n_columns: 19, retrain_triggered: true, note: "SIMULATED" } });
    return ok({});
  }));
});

test("predict shows probability", async () => {
  render(<App />);
  await userEvent.click(screen.getByText("Predict churn"));
  await waitFor(() => expect(screen.getByText(/73.0% churn probability/)).toBeInTheDocument());
});

test("model and monitoring tabs render metrics", async () => {
  render(<App />);
  await userEvent.click(screen.getByText("Model & metrics"));
  await screen.findByText("0.84");
  await userEvent.click(screen.getByText("Drift monitoring"));
  await screen.findByText(/4 \/ 19/);
  expect(screen.getByText("YES")).toBeInTheDocument();
});
