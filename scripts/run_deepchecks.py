"""Fail CI when text-property or prediction distributions drift too far."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from app.utils import load_classifier

load_dotenv(override=True)
def load_params():
    with open("params.yaml", encoding="utf-8") as f: return yaml.safe_load(f)["deepchecks"]
def run_predictions(classifier, texts):
    outputs=[]
    for start in range(0, len(texts), 32):
        batch=classifier(texts[start:start+32], batch_size=32, truncation=True)
        outputs.extend(str(x["label"]).lower() for x in batch)
    return outputs
def main():
    params=load_params()
    stream_df=pd.read_csv("data/stream.csv"); test_df=pd.read_csv("data/test.csv")
    stream_texts=stream_df["text"].dropna().astype(str).tolist(); test_texts=test_df["text"].dropna().astype(str).tolist()
    if not stream_texts or not test_texts: raise ValueError("Both stream.csv and test.csv must contain text rows")
    # Use Deepchecks NLP's TextData and built-in properties; the sentiment property is
    # represented by sentiment polarity scores derived consistently from model predictions.
    try:
        from deepchecks.nlp import TextData
        reference = TextData(raw_text=test_texts, task_type="text_classification")
        current = TextData(raw_text=stream_texts, task_type="text_classification")
        reference.calculate_builtin_properties(include_properties=["Sentiment"])
        current.calculate_builtin_properties(include_properties=["Sentiment"])
        from deepchecks.nlp.checks import PropertyDrift
        check = PropertyDrift(properties=["Sentiment"])
        result = check.run(reference, current)
        display = result.value
        score = None
        if isinstance(display, dict):
            item = display.get("Sentiment", display)
            if isinstance(item, dict): score = item.get("Drift score", item.get("drift_score", item.get("score")))
        if score is None:
            # API versions differ; extract the documented result table when available.
            table = getattr(result, "display", None)
            if callable(table):
                shown = table()
                if isinstance(shown, pd.DataFrame) and "Drift score" in shown.columns:
                    row = shown[shown.iloc[:, 0].astype(str).str.contains("Sentiment", case=False)]
                    if not row.empty: score = float(row["Drift score"].iloc[0])
        if score is None: raise RuntimeError(f"Could not extract Sentiment drift score from Deepchecks result: {display}")
        print(f"Sentiment property drift: {float(score):.4f}")
        if float(score) > params["property_drift_threshold"]: raise SystemExit("Property drift threshold exceeded")
    except ImportError as exc:
        raise RuntimeError("Deepchecks NLP is required; install requirements.txt") from exc
    classifier=load_classifier()
    ref_pred=run_predictions(classifier, test_texts); cur_pred=run_predictions(classifier, stream_texts)
    # Distribution drift is the total-variation distance over predicted sentiment labels (0..1).
    labels=sorted(set(ref_pred) | set(cur_pred))
    p=np.array([ref_pred.count(label)/len(ref_pred) for label in labels]); q=np.array([cur_pred.count(label)/len(cur_pred) for label in labels])
    prediction_drift=float(0.5*np.abs(p-q).sum())
    print(f"Prediction drift: {prediction_drift:.4f}")
    if prediction_drift > params["prediction_drift_threshold"]: raise SystemExit("Prediction drift threshold exceeded")
if __name__ == "__main__": main()
