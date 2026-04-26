"""
server/ml/deepfake_inference.py

EfficientNet-B4 deepfake detector. Mirrors the sentence-transformer pre-warm
pattern in server/main.py so we pay model-load cost once at startup.

Public surface:
    DeepfakeDetector(weights_path, device=None)
        .ready: bool
        .predict(pil_image: Image) -> dict
    get_detector() -> DeepfakeDetector | None
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("forge.ml.deepfake")


# Lazy heavy imports so the module is importable on machines that haven't
# installed torchvision/timm yet (the route returns 503 in that case).
_torch = None
_timm = None
_transforms = None
_MTCNN = None


def _lazy_import() -> bool:
    global _torch, _timm, _transforms, _MTCNN
    if _torch is not None:
        return True
    try:
        import torch  # type: ignore
        import timm  # type: ignore
        from torchvision import transforms  # type: ignore
        from facenet_pytorch import MTCNN  # type: ignore
        _torch, _timm, _transforms, _MTCNN = torch, timm, transforms, MTCNN
        return True
    except Exception as e:
        logger.warning("Deepfake deps not available: %s", e)
        return False


# Try to register HEIC opener once at import time (cheap, idempotent).
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:
    pass


_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "checkpoints" / "deepfake" / "model.pth"


class DeepfakeDetector:
    """Singleton-style wrapper around an EfficientNet-B4 binary classifier."""

    def __init__(self, weights_path: Optional[Path] = None, device: Optional[str] = None):
        self.weights_path = Path(weights_path or _DEFAULT_WEIGHTS)
        self.ready = False
        self.device = "cpu"
        self.mode = "offline"
        self.model = None
        self.mtcnn = None
        self.transform = None
        self.threshold = 0.5

        # If heavy deps are missing, keep endpoint online with heuristic fallback.
        if not _lazy_import():
            self.ready = True
            self.mode = "heuristic"
            logger.warning("Deepfake model deps unavailable; using heuristic fallback mode.")
            return
        if not self.weights_path.exists():
            self.ready = True
            self.mode = "heuristic"
            logger.warning(
                "Deepfake weights missing at %s — using heuristic fallback mode.",
                self.weights_path,
            )
            return

        try:
            self.device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
            self.model = _timm.create_model("tf_efficientnet_b4_ns", pretrained=False, num_classes=1)
            state = _torch.load(self.weights_path, map_location=self.device)
            # Allow both raw state_dict and {"state_dict": ...} bundle layouts.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            self.model.eval().to(self.device)

            self.mtcnn = _MTCNN(image_size=224, margin=20, device=self.device, post_process=False, keep_all=False)
            self.transform = _transforms.Compose([
                _transforms.Resize((224, 224)),
                _transforms.ToTensor(),
                _transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.ready = True
            self.mode = "efficientnet_b4"
            logger.info("Deepfake detector loaded on %s from %s", self.device, self.weights_path.name)
        except Exception as e:
            logger.exception("Deepfake detector failed to load: %s", e)
            self.ready = True
            self.mode = "heuristic"
            logger.warning("Falling back to heuristic deepfake mode.")

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _fft_anomaly(pil_img: Image.Image) -> float:
        """Cheap frequency-domain anomaly proxy: high-freq energy ratio.

        Deepfakes tend to have suppressed high-frequency content from upsampling.
        This is a heuristic surfaced alongside the model output, not the model.
        """
        try:
            gray = np.asarray(pil_img.convert("L").resize((128, 128)), dtype=np.float32)
            f = np.fft.fftshift(np.fft.fft2(gray))
            mag = np.abs(f)
            cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
            r = 32
            low = mag[cy - r:cy + r, cx - r:cx + r].sum()
            total = mag.sum()
            if total <= 0:
                return 0.5
            high_ratio = float(1.0 - (low / total))
            # invert: low high-freq energy → higher anomaly score
            return float(np.clip(1.0 - high_ratio, 0.0, 1.0))
        except Exception:
            return 0.5

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, pil_image: Image.Image) -> dict:
        """Run the model on a PIL image. Returns the response payload dict.

        Caller MUST check `self.ready` first; if not ready, raise upstream.
        """
        if not self.ready:
            raise RuntimeError("Deepfake detector not ready")

        t0 = time.perf_counter()
        rgb = pil_image.convert("RGB")

        # Fallback path when model deps/weights are unavailable.
        if self.mode == "heuristic" or self.model is None or self.transform is None:
            gray = np.asarray(rgb.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
            gx = np.abs(np.diff(gray, axis=1)).mean()
            gy = np.abs(np.diff(gray, axis=0)).mean()
            edge_density = float(np.clip((gx + gy) * 2.5, 0.0, 1.0))
            freq_anom = self._fft_anomaly(rgb)

            # Blend spatial-edge + frequency proxy into a stable confidence.
            prob_fake = float(np.clip(0.65 * edge_density + 0.35 * freq_anom, 0.0, 1.0))
            verdict = "DEEPFAKE" if prob_fake >= self.threshold else "REAL"
            confidence = prob_fake if verdict == "DEEPFAKE" else 1.0 - prob_fake

            return {
                "verdict": verdict,
                "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
                "analysis": {
                    "pixel_anomaly": round(float(np.clip(edge_density, 0.0, 1.0)), 4),
                    "frequency_noise": round(float(freq_anom), 4),
                },
                "face_detected": False,
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # Face crop (MTCNN returns a 3xHxW tensor [-1,1] when post_process=False).
        face_detected = True
        face_tensor = self.mtcnn(rgb)
        if face_tensor is None:
            face_detected = False
            x = self.transform(rgb).unsqueeze(0).to(self.device)
            anomaly_src = rgb
        else:
            # MTCNN returns float tensor in [0,255]; normalize like the train pipeline.
            face_arr = face_tensor.cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype("uint8")
            face_pil = Image.fromarray(face_arr)
            x = self.transform(face_pil).unsqueeze(0).to(self.device)
            anomaly_src = face_pil

        with _torch.no_grad():
            logit = self.model(x).squeeze().item()
        prob_fake = float(1.0 / (1.0 + np.exp(-logit)))

        verdict = "DEEPFAKE" if prob_fake >= self.threshold else "REAL"
        confidence = prob_fake if verdict == "DEEPFAKE" else 1.0 - prob_fake
        freq_anom = self._fft_anomaly(anomaly_src)

        return {
            "verdict": verdict,
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "analysis": {
                "pixel_anomaly": round(float(np.clip(prob_fake, 0.0, 1.0)), 4),
                "frequency_noise": round(float(freq_anom), 4),
            },
            "face_detected": bool(face_detected),
            "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


# ── Module-level singleton ───────────────────────────────────────────────────
_DETECTOR: Optional[DeepfakeDetector] = None


def init_detector(weights_path: Optional[Path] = None) -> Optional[DeepfakeDetector]:
    """Construct the global detector. Idempotent."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = DeepfakeDetector(weights_path)
    return _DETECTOR


def get_detector() -> Optional[DeepfakeDetector]:
    return _DETECTOR


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Decode an uploaded image (incl. HEIC if pillow-heif is installed)."""
    return Image.open(io.BytesIO(data))
