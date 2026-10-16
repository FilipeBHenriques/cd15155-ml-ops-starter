"""Send simulated production traffic and log window metrics to MLflow."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-monitoring")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "50"))
SLEEP_MS = int(os.getenv("SLEEP_MS", "100"))
def predict(text: str) -> dict:
    response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
    response.raise_for_status()
    return response.json()
def log_window(window: list[dict], window_idx: int) -> None:
    if not window: return
    n = len(window)
    counts = {s: sum(1 for p in window if p["sentiment"].lower() == s) for s in ("positive", "negative", "neutral")}
    metrics = {f"pct_{s}": counts[s] * 100.0 / n for s in counts}
    metrics["avg_confidence"] = sum(float(p["confidence"]) for p in window) / n
    metrics["avg_latency_ms"] = sum(float(p["latency_ms"]) for p in window) / n
    mlflow.log_metrics(metrics, step=window_idx)
    print(f"Logged monitoring window {window_idx}: {metrics}")
def main():
    df = pd.read_csv("data/stream.csv")
    if "text" not in df or df.empty: raise ValueError("data/stream.csv must contain at least one text row")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    window, window_idx = [], 0
    with mlflow.start_run(run_name="production-monitoring-stream"):
        for text in df["text"].dropna().astype(str):
            if not text.strip(): continue
            result = predict(text)
            window.append(result)
            if len(window) >= WINDOW_SIZE:
                log_window(window, window_idx); window_idx += 1; window = []
            time.sleep(SLEEP_MS / 1000)
        if window: log_window(window, window_idx)
if __name__ == "__main__": main()
