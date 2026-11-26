import { createRoot } from "react-dom/client";
import { App } from "./App";
import { AuthGate } from "./auth";
import { History } from "./History";
import "./styles.css";
createRoot(document.getElementById("root")!).render(<AuthGate title="Churn Predictor"><App /><History /></AuthGate>);
