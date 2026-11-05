from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Model dosyasının proje ana klasöründeki konumu
MODEL_PATH = Path(__file__).resolve().parents[2] / "yolo26n.pt"

# Varsayılan güven eşiği: bu değerin altındaki tespitler döndürülmez
DEFAULT_MIN_CONFIDENCE = 0.25

_model = None


def get_model():
    """YOLO modelini gerektiğinde yükler."""
    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"YOLO model dosyası bulunamadı: {MODEL_PATH}"
            )
        _model = YOLO(str(MODEL_PATH))

    return _model


def _check_image(image: Path) -> None:
    """Dosyanın gerçekten açılabilen bir görsel olduğunu kontrol eder.

    Kontrol OpenCV ile yapılır. Pillow kullanılmaz, çünkü Pillow dosyayı
    tanımadığında Ultralytics ek paket (pi-heif) kurmaya çalışabiliyor.
    Dosya önce bayt olarak okunur; böylece Türkçe karakterli
    Windows yollarında da çalışır.
    """
    try:
        data = np.fromfile(str(image), dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"Görsel okunamadı: {image}") from error

    if data.size == 0 or cv2.imdecode(data, cv2.IMREAD_COLOR) is None:
        raise ValueError(f"Görsel okunamadı veya bozuk: {image}")


def detect_objects(
    image_path: str,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[dict]:
    """Görseldeki nesneleri ve özelliklerini döndürür.

    Hatalar:
        ValueError: min_confidence 0-1 aralığında değilse
            veya dosya geçerli bir görsel değilse.
        FileNotFoundError: Görsel ya da model dosyası bulunamazsa.
    """
    # 1) Güven eşiği 0 ile 1 arasında olmalı
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence 0 ile 1 arasında olmalıdır.")

    # 2) Görsel dosyası var mı?
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Görsel bulunamadı: {image}")

    # 3) Dosya gerçekten bir görsel mi?
    _check_image(image)

    # 4) Nesne tespiti
    model = get_model()
    results = model(str(image), conf=min_confidence, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            detections.append({
                "label": result.names[class_id],
                "confidence": float(box.conf[0]),
                "box": [
                    float(value)
                    for value in box.xyxy[0].tolist()
                ],
            })

    return detections
