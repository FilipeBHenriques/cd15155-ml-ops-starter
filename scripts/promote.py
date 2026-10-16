"""Promote the latest registered model when latest evaluation F1 meets threshold."""
import os

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv(override=True)
def load_params():
    with open("params.yaml", encoding="utf-8") as f: return yaml.safe_load(f)["promote"]
def get_latest_f1(client: MlflowClient, experiment_name: str):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None: raise RuntimeError(f"MLflow experiment not found: {experiment_name}")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs or "f1_weighted" not in runs[0].data.metrics: raise RuntimeError("No evaluation run with f1_weighted metric was found")
    return runs[0].info.run_id, float(runs[0].data.metrics["f1_weighted"])
def promote(client: MlflowClient, model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions: raise RuntimeError(f"No registered versions found for model {model_name}")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(model_name, "production", latest.version)
    print(f"Promoted {model_name} version {latest.version} to alias 'production'")
def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    threshold = float(load_params()["f1_threshold"])
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    model_name = os.getenv("MODEL_NAME", "finbert")
    run_id, f1 = get_latest_f1(client, experiment)
    print(f"Latest evaluation run {run_id}: f1_weighted={f1:.4f}; threshold={threshold:.4f}")
    if f1 >= threshold: promote(client, model_name)
    else: print(f"Model not promoted: F1 {f1:.4f} does not meet threshold {threshold:.4f}.")
if __name__ == "__main__": main()
