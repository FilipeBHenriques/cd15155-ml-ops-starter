"""Model loading utilities shared by the API and evaluation scripts."""
import json
import os

from transformers import pipeline


def patch_model_config(model_id: str) -> None:
    """Normalize the upstream model's malformed id2label/label2id config if needed."""
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        expected = {"0": "negative", "1": "neutral", "2": "positive"}
        if config.get("id2label") != expected or not all(isinstance(v, int) for v in config.get("label2id", {}).values()):
            config["id2label"] = expected
            config["label2id"] = {"negative": 0, "neutral": 1, "positive": 2}
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
    except Exception as exc:  # noqa: BLE001 - best-effort patch; pipeline load will surface real errors
        # Do not hide model loading errors; the pipeline will report invalid config if relevant.
        print(f"Warning: could not patch model config for {model_id}: {exc}")

def load_classifier():
    """Load from HuggingFace or an MLflow registered-model alias."""
    model_source = os.getenv("MODEL_SOURCE", "mlflow")
    if model_source == "huggingface":
        hf_model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
        print(f"Loading model from HuggingFace: {hf_model_id}")
        patch_model_config(hf_model_id)
        return pipeline("text-classification", model=hf_model_id, tokenizer=hf_model_id, device="cpu", truncation=True)
    import mlflow
    import mlflow.transformers
    model_name = os.getenv("MODEL_NAME", "finbert")
    model_alias = os.getenv("MODEL_STAGE", "production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"Loading model from MLflow: models:/{model_name}@{model_alias}")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.transformers.load_model(f"models:/{model_name}@{model_alias}")
