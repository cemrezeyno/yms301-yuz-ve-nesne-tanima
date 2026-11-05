
from pathlib import Path
from ultralytics import YOLO

# Model dosyasının proje ana klasöründeki konumu
MODEL_PATH = Path(__file__).resolve().parents[2] / "yolo26n.pt"

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


def detect_objects(image_path: str) -> list[dict]:
    """Görseldeki nesneleri ve özelliklerini döndürür."""
    image = Path(image_path)

    if not image.is_file():
        raise FileNotFoundError(f"Görsel bulunamadı: {image}")

    model = get_model()
    results = model(str(image))

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
