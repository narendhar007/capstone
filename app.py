"""app.py — Stage 3: FastAPI inference service.

Implement /health (liveness + loaded model info) and POST /predict (multipart image upload
→ {label, prob_defect, confidence}). Load the model once at startup; log every prediction
to artifacts/predictions.log.   Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io, json, time
from contextlib import asynccontextmanager
from pathlib import Path

import torch, torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

import config
from src import data_prep
from src.model import load_model

_state = {"model": None, "tf": None, "meta": {}}


def _load():
    # TODO 3: if config.MODEL_PATH exists, load model + eval transforms + model_meta.json.
    raise NotImplementedError


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load(); yield


app = FastAPI(title="Casting Defect Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    # TODO 3: return status + model_loaded + classes + positive_class + test_metrics.
    raise NotImplementedError


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    # TODO 3: validate model loaded (503 else); read image (400 if invalid); preprocess;
    #         softmax; return {label, is_defective, prob_defect, confidence}; log prediction.
    raise NotImplementedError
