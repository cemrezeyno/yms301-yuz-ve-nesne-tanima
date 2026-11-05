"""Yüz embedding çıkarma ve kişi eşleştirme (OpenCV SFace)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.face_recognition import UNKNOWN_NAME, FaceResult, detect_faces

logger = logging.getLogger(__name__)

SFACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_recognition_sface_2021dec.onnx"
)
# SFace için OpenCV'nin önerdiği kosinüs benzerliği eşiği.
COSINE_THRESHOLD = 0.363

_recognizer: cv2.FaceRecognizerSF | None = None


def get_recognizer() -> cv2.FaceRecognizerSF:
    """SFace modelini yalnızca ilk çağrıda yükler."""
    global _recognizer

    if _recognizer is None:
        if not SFACE_PATH.is_file():
            raise FileNotFoundError(f"SFace modeli bulunamadı: {SFACE_PATH}")
        # Model bellekten verilir: tam yolla açmak bazı yollarda
        # (Türkçe karakter / OneDrive) OpenCV'de hata veriyor.
        model_bytes = np.frombuffer(SFACE_PATH.read_bytes(), dtype=np.uint8)
        _recognizer = cv2.FaceRecognizerSF.create(
            "onnx", model_bytes, np.empty(0, dtype=np.uint8)
        )
        logger.info("SFace modeli yüklendi: %s", SFACE_PATH.name)

    return _recognizer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki embedding vektörü arasındaki kosinüs benzerliği."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_best_match(
    embedding: np.ndarray,
    known: dict[int, np.ndarray],
    threshold: float = COSINE_THRESHOLD,
) -> tuple[int | None, float]:
    """Kayıtlı embedding'ler arasından en benzer kişiyi bulur.

    known: {person_id: embedding}. Eşik altında kalırsa (None, en_iyi_skor).
    """
    best_id: int | None = None
    best_score = -1.0

    for person_id, other in known.items():
        score = cosine_similarity(embedding, other)
        if score > best_score:
            best_id, best_score = person_id, score

    if best_id is None or best_score < threshold:
        return None, max(best_score, 0.0)
    return best_id, best_score