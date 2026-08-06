"""app.py — Stage 3: FastAPI inference service.

Implement /health (liveness + loaded model info) and POST /predict (multipart image upload
→ {label, prob_defect, confidence}). Load the model once at startup; log every prediction
to artifacts/predictions.log.   Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io, json, time
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone

import torch, torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

import config
from src import data_prep
from src.model import load_model

_state = {"model": None, "tf": None, "meta": {}}


def _load():
    # TODO 3: if config.MODEL_PATH exists, load model + eval transforms + model_meta.json.
    """
    Load the production model, inference transform and metadata.
    
    """

    # Reset the state so repeated TestClient sessions remain predictable.
    _state.update(
        {
            "model": None,
            "tf": None,
            "meta": {},
        }
    )

    # Metadata is useful for /health but is not required for inference.
    if config.MODEL_META_PATH.exists():
        _state["meta"] = json.loads(
            config.MODEL_META_PATH.read_text(
                encoding="utf-8"
            )
        )

    # Allow the API to start without a model. In that state, /health
    # reports model_loaded=False and /predict returns HTTP 503.
    if not config.MODEL_PATH.exists():
        return

    model = load_model(
        path=config.MODEL_PATH,
        freeze=True,
    )

    model = model.to(config.DEVICE)
    model.eval()

    _state["model"] = model

    # Validation, test and inference all use the same deterministic
    # preprocessing pipeline.
    _state["tf"] = data_prep.get_transforms(
        train=False
    )

    # Create the predictions log file if it doesn't exist yet.
    config.PREDICTIONS_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    #raise NotImplementedError



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage resources used for the application lifetime.
    """

    _load() # Load the model and inference transforms at startup.

    yield # Allow the application to run.

    # Release references when the service shuts down.
    _state.update(
        {
            "model": None,
            "tf": None,
            "meta": {},
        }
    )

app = FastAPI(title="Casting Defect Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    # TODO 3: return status + model_loaded + classes + positive_class + test_metrics.
    """
    Return service readiness and production-model information.
    
    """

    # Determine whether the model and inference transforms are loaded.
    model_loaded = (
        _state["model"] is not None
        and _state["tf"] is not None
    )

    # Extract metadata from the model_meta.json file, if it exists.
    metadata = _state["meta"]

    # Return a dictionary with the service status and model information.
    return {
        "status": (
            "healthy"
            if model_loaded
            else "unavailable"
        ),
        "model_loaded": model_loaded,
        "classes": config.CLASSES,
        "positive_class": config.POSITIVE_CLASS,
        "dataset_version": metadata.get(
            "dataset_version"
        ),
        "mlflow_run_id": metadata.get(
            "mlflow_run_id"
        ),
        "registered_model": metadata.get(
            "registered_model"
        ),
        "registered_model_version": metadata.get(
            "registered_model_version"
        ),
        "production_alias": metadata.get(
            "production_alias"
        ),
        "test_metrics": metadata.get(
            "test_metrics",
            {},
        ),
    }
    raise NotImplementedError


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    # TODO 3: validate model loaded (503 else); read image (400 if invalid); preprocess;
    #         softmax; return {label, is_defective, prob_defect, confidence}; log prediction.
    """
    Classify one uploaded casting image.
    
    """

    # Determine whether the model and inference transforms are loaded.
    model = _state["model"]
    transform = _state["tf"]

    # If the model or transform is not loaded, raise an HTTP 503 Service Unavailable error.
    if model is None or transform is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    # Measure the time taken for the prediction process.
    start_time = time.perf_counter()

    # Read the uploaded image file as bytes. If the file is empty, raise an HTTP 400 Bad Request error.
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        # load() forces Pillow to decode the image now so corrupt image
        # data is detected before inference.
        with Image.open(
            io.BytesIO(image_bytes)
        ) as opened_image:
            opened_image.load()
            image = opened_image.copy()

    # Handle exceptions related to image loading and decoding. If the uploaded file is not a valid image, 
    # raise an HTTP 400 Bad Request error.
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc

    # Preprocess the image using the loaded inference transform and convert it to a tensor suitable for model input.
    input_tensor = (
        transform(image)
        .unsqueeze(0) # Add a batch dimension to the tensor.
        .to(config.DEVICE)
    )

    # Perform inference using the loaded model without tracking gradients, and compute the softmax probabilities for each class.
    with torch.no_grad():
        logits = model(input_tensor)

        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

    # Determine the predicted class index by finding the index of the maximum probability in the softmax output.
    predicted_index = int(
        probabilities.argmax().item()
    )

    # Map the predicted class index to the corresponding class label using the configuration mapping.
    label = config.IDX_TO_CLASS[
        predicted_index
    ]

    # Calculate the probability of the positive class (defective) and the confidence of the prediction based on the softmax probabilities.
    probability_defect = float(
        probabilities[
            config.POSITIVE_IDX
        ].item()
    )

    # Calculate the confidence of the prediction based on the softmax probabilities for the predicted class.
    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )

    # Determine whether the predicted class corresponds to a defective casting based on the configuration mapping.
    is_defective = (
        predicted_index
        == config.POSITIVE_IDX
    )

    # Measure the latency of the prediction process in milliseconds by calculating the time difference from the 
    # start of the prediction to the current time.
    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    # Prepare the response dictionary containing the predicted label, defect status, probability of defect, and confidence of the prediction.
    response = {
        "label": label,
        "is_defective": is_defective,
        "prob_defect": probability_defect,
        "confidence": confidence,
    }

    # Log the prediction details, including the timestamp, filename, content type, prediction results, and latency, 
    # to a JSON Lines file for monitoring and analysis.
    prediction_record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "filename": (
            file.filename
            or "uploaded_image"
        ),
        "content_type": file.content_type,
        **response,
        "latency_ms": round(
            latency_ms,
            3,
        ),
    }

    # JSON Lines keeps each prediction as one independently readable
    # record and will be reused during Stage 4 monitoring.
    with config.PREDICTIONS_LOG.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            json.dumps(
                prediction_record
            )
            + "\n"
        )

    return response

    raise NotImplementedError
