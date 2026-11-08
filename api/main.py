import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.monitoring_logger import log_inference
from api.preprocessing import clean_prediction_data
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BreastCancerInput,
    ErrorDetail,
    ErrorResponse,
    PredictionResponse,
)

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load()
    yield


app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description="Predicts benign or malignant diagnosis from breast-mass measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in error["loc"] if part != "body"),
            message=error["msg"],
        ).model_dump()
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            status="error",
            message="Request validation failed",
            details=details,
        ).model_dump(),
    )


@app.get("/")
def root():
    return {"message": "Breast Cancer Diagnosis API is running", "status": "success"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if model_loader.is_ready() else "not_ready",
        "model_loaded": model_loader.model is not None,
    }


@app.get("/metadata")
def metadata():
    return {
        "service": "Breast Cancer Diagnosis API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "model_uri": model_loader.model_uri,
        "model_loaded": model_loader.is_ready(),
        "feature_count": len(model_loader.feature_names),
    }


def predict_one(sample: BreastCancerInput) -> PredictionResponse:
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        features = clean_prediction_data(
            sample.features, model_loader.feature_names
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    started_at = time.perf_counter()
    prediction = int(model_loader.model.predict(features)[0])
    malignant_probability = float(model_loader.model.predict_proba(features)[0][1])
    latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
    log_inference(
        input_data={"feature_count": len(sample.features)},
        prediction=prediction,
        malignant_probability=malignant_probability,
        latency_ms=latency_ms,
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
    )
    return PredictionResponse(
        prediction=prediction,
        diagnosis="M" if prediction == 1 else "B",
        malignant_probability=malignant_probability,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(sample: BreastCancerInput):
    return predict_one(sample)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    predictions = [predict_one(sample) for sample in request.samples]
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
