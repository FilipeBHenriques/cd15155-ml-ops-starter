"""FinBERT sentiment prediction API with Prometheus instrumentation."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from utils import load_classifier

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
_logger = logging.getLogger("finbert-api")

def log(level: str, message: str, **kwargs) -> None:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": level.upper(), "message": message, **kwargs}
    getattr(_logger, level.lower(), _logger.info)(json.dumps(entry, default=str))

PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total predictions by sentiment", ["sentiment"])
PREDICTION_LATENCY = Histogram("prediction_latency_ms", "Prediction latency in milliseconds", buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, float("inf")))
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors")
load_dotenv(override=True)
classifiers = {}

class PredictRequest(BaseModel):
    text: str
class PredictBatchRequest(BaseModel):
    texts: list[str]
class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log("INFO", "Loading model")
        classifiers["sentiment"] = load_classifier()
        log("INFO", "Model loaded successfully")
    except Exception as exc:  # noqa: BLE001 - keep API alive on any load failure
        classifiers.pop("sentiment", None)
        log("ERROR", "Model loading failed", error=str(exc))
        # Keep the API alive so health checks can report 503 and orchestration can restart it.
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")

app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

def run_predictions(texts: list[str]) -> list[PredictionResult]:
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise HTTPException(status_code=422, detail="Text must not be empty")
    try:
        for text in texts:
            if len(text) > 2000:
                log("WARNING", "Input exceeds 2000 characters; model may truncate input", text_length=len(text))
        start = time.perf_counter()
        outputs = classifiers["sentiment"](texts if len(texts) > 1 else texts[0], batch_size=32, truncation=True)
        if isinstance(outputs, dict):
            outputs = [outputs]
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_item_ms = elapsed_ms / max(1, len(texts))
        results = []
        for text, output in zip(texts, outputs):
            sentiment = str(output["label"]).lower()
            confidence = float(output["score"])
            PREDICTION_REQUESTS.labels(sentiment=sentiment).inc()
            PREDICTION_LATENCY.observe(per_item_ms)
            result = PredictionResult(text=text, sentiment=sentiment, confidence=confidence, latency_ms=round(per_item_ms, 3))
            results.append(result)
            log("INFO", "Prediction completed", sentiment=sentiment, confidence=confidence, latency_ms=result.latency_ms)
        return results
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", error=str(exc))
        raise

# Keep singular helper available for callers/tests expecting run_prediction.
def run_prediction(text: str):
    result = run_predictions([text])[0]
    return {"label": result.sentiment, "score": result.confidence}, result.latency_ms

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty")
    return run_predictions([request.text])[0]

@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    if not request.texts:
        raise HTTPException(status_code=422, detail="Texts list must not be empty")
    if any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=422, detail="Texts must not be empty")
    return run_predictions(request.texts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
