"""Yüz tespiti modülü (OpenCV YuNet)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)
UNKNOWN_NAME = "Bilinmeyen kişi"
TARGET_SIZE = 640
SCORE_THRESHOLD = 0.7
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_detector: cv2.FaceDetectorYN | None = None


@dataclass(frozen=True)
class FaceResult:
    """Tek bir yüzün sonucu. box: (x, y, genişlik, yükseklik)."""

    box: tuple[int, int, int, int]
    name: str
    confidence: float
    person_id: int | None = None

    def box_xyxy(self) -> tuple[int, int, int, int]:
        """Kutuyu (x1, y1, x2, y2) köşe biçiminde döndürür."""
        x, y, w, h = self.box
        return x, y, x + w, y + h

    def to_dict(self) -> dict:
        """database.save_analysis'in beklediği yüz sözlüğünü üretir."""
        return {
            "person_id": self.person_id,
            "confidence": self.confidence,
            "box": list(self.box_xyxy()),
        }


def get_detector() -> cv2.FaceDetectorYN:
    """Yüz tespit modelini yalnızca ilk çağrıda yükler."""
    global _detector

    if _detector is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Yüz modeli bulunamadı: {MODEL_PATH}")
        # Model bellekten verilir: Türkçe karakterli yollarda dosya yolu
        # üzerinden açmak OpenCV'de hata verebilir.
        model_bytes = np.frombuffer(MODEL_PATH.read_bytes(), dtype=np.uint8)
        _detector = cv2.FaceDetectorYN.create(
            "onnx", model_bytes, np.empty(0, dtype=np.uint8), (320, 320),
            score_threshold=SCORE_THRESHOLD,
        )
        logger.info("Yüz tespit modeli yüklendi: %s", MODEL_PATH.name)

    return _detector


def load_image(image_path: str | Path) -> np.ndarray:
    """Görseli diskten okur; geçersiz uzantı veya bozuk dosyada hata verir."""
    path = Path(image_path)

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Desteklenmeyen dosya uzantısı: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(f"Görsel bulunamadı: {path}")

    # np.fromfile, Türkçe karakterli yollarda cv2.imread'den daha güvenlidir.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Görsel okunamadı (bozuk dosya): {path}")

    return image


def detect_faces(image: np.ndarray) -> list[FaceResult]:
    """BGR görseldeki yüzleri bulur. Kutular orijinal boyuta göredir."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        raise ValueError("Geçersiz görsel: dolu bir BGR numpy dizisi bekleniyor.")

    height, width = image.shape[:2]
    scale = min(1.0, TARGET_SIZE / max(height, width))

    small = image
    if scale < 1.0:
        small = cv2.resize(image, (int(width * scale), int(height * scale)))

    small_h, small_w = small.shape[:2]
    detector = get_detector()
    detector.setInputSize((small_w, small_h))
    _, faces = detector.detect(small)

    if faces is None:
        logger.info("Yüz bulunamadı.")
        return []

    results: list[FaceResult] = []
    for face in faces:
        x, y, w, h = (face[:4] / scale).astype(int)
        x, y = max(int(x), 0), max(int(y), 0)
        w, h = min(int(w), width - x), min(int(h), height - y)
        results.append(
            FaceResult(box=(x, y, w, h), name=UNKNOWN_NAME,
                       confidence=float(face[-1]))
        )

    logger.info("%d yüz bulundu.", len(results))
    return results


def analyze(image: np.ndarray | str | Path) -> list[FaceResult]:
    """Dosya yolu veya numpy dizisi alır, FaceResult listesi döndürür."""
    if isinstance(image, (str, Path)):
        image = load_image(image)
    return detect_faces(image)


def draw_faces(image: np.ndarray, results: list[FaceResult]) -> np.ndarray:
    """Yüz kutularını görselin kopyası üzerine çizer."""
    output = image.copy()
    thickness = max(2, max(output.shape[:2]) // 300)
    for result in results:
        x, y, w, h = result.box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), thickness)
    return output