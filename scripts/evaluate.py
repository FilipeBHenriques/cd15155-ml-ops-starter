"""Evaluate FinBERT and log/register it in MLflow."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.transformers
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

from app.utils import patch_model_config

load_dotenv(override=True)

def load_test_data(test_path):
    df = pd.read_csv(test_path)
    return df["text"].astype(str).tolist(), df["label"].astype(str).tolist()
def build_classifier(model_id):
    return pipeline("text-classification", model=model_id, tokenizer=model_id, truncation=True, max_length=512, device=-1)
def run_inference(classifier, texts):
    results = classifier(texts, batch_size=32, truncation=True)
    return [str(r["label"]).lower() for r in results]
def compute_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true,y_pred), "f1_weighted": f1_score(y_true,y_pred,average="weighted",zero_division=0), "precision_weighted": precision_score(y_true,y_pred,average="weighted",zero_division=0), "recall_weighted": recall_score(y_true,y_pred,average="weighted",zero_division=0)}
def main():
    model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    model_name = os.getenv("MODEL_NAME", "finbert")
    print(f"Using MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    texts, y_true = load_test_data(os.getenv("TEST_DATA_PATH", "data/test.csv"))
    patch_model_config(model_id)
    classifier = build_classifier(model_id)
    y_pred = run_inference(classifier, texts)
    metrics = compute_metrics(y_true, y_pred)
    print("Evaluation metrics:", metrics)
    print(classification_report(y_true, y_pred, zero_division=0))
    with mlflow.start_run(run_name="finbert-evaluation") as run:
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("test_samples", len(texts))
        mlflow.log_metrics({k: float(v) for k,v in metrics.items()})
        # Log the actual HuggingFace pipeline as an MLflow Transformers model and register it.
        mlflow.transformers.log_model(transformers_model={"model": classifier.model, "tokenizer": classifier.tokenizer}, artifact_path="model", registered_model_name=model_name, input_example=[texts[0]] if texts else None)
        print(f"MLflow run: {run.info.run_id}; registered model: {model_name}")
if __name__ == "__main__": main()
