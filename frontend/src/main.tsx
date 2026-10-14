import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { clearServiceWorkers } from "./pwa";
import "./styles.css";

clearServiceWorkers();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
