"""
server/routes/deepfake.py — image deepfake detection endpoint.

POST /detect-deepfake (multipart/form-data, field: "file")
    → DeepfakeResponse  { verdict, confidence, analysis, face_detected, inference_ms }

Behavior:
  * Validates MIME and file size before any heavy work.
  * Decodes via PIL (HEIC supported when pillow-heif is installed).
  * Runs inference inside a thread executor so the event loop stays free.
  * Returns 503 with a clear detail message when weights are missing.
"""

import asyncio
import json
import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from server.ml.deepfake_inference import get_detector, load_image_from_bytes
from server.schemas import DeepfakeResponse

router = APIRouter()
logger = logging.getLogger("forge.routes.deepfake")

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_PREFIX = "image/"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


def _validate_filename(name: str | None) -> bool:
    if not name:
        return True  # browsers sometimes omit; rely on MIME
    lower = name.lower()
    return any(lower.endswith(ext) for ext in ALLOWED_EXTS)


@router.post("/detect-deepfake", response_model=DeepfakeResponse, tags=["Deepfake"])
async def detect_deepfake(file: UploadFile = File(...)) -> DeepfakeResponse:
    detector = get_detector()
    if detector is None or not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="Deepfake detector unavailable. Ensure checkpoints/deepfake/model.pth exists "
                   "and torchvision/timm/facenet-pytorch are installed.",
        )

    # MIME / extension gate (cheap).
    ctype = (file.content_type or "").lower()
    if not (ctype.startswith(ALLOWED_PREFIX) or _validate_filename(file.filename)):
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {ctype!r}")

    # Read with size cap (avoid pulling huge files into memory).
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_BYTES // (1024*1024)} MB).")

    # Decode.
    try:
        image = load_image_from_bytes(data)
        image.load()  # force-decode now so errors surface as 400, not 500
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    # Run inference off the event loop.
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, detector.predict, image)
    except Exception as e:
        logger.exception("Deepfake inference crashed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return DeepfakeResponse(**result)


@router.get("/detect-deepfake/status")
async def detect_deepfake_status():
    """Lightweight health probe so the frontend can show a 'model ready' badge."""
    detector = get_detector()
    if detector is None:
        return {"ready": False, "reason": "not_initialized"}

    # Read val_accuracy from config.json to display training percentage.
    val_accuracy = None
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "deepfake", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            val_accuracy = cfg.get("val_accuracy")
    except Exception:
        pass

    return {
        "ready": bool(detector.ready),
        "mode": getattr(detector, "mode", "unknown"),
        "reason": "heuristic_fallback" if getattr(detector, "mode", "") == "heuristic" else None,
        "device": detector.device,
        "weights_path": str(detector.weights_path),
        "threshold": detector.threshold,
        "val_accuracy": val_accuracy,
    }
