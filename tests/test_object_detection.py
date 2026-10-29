
import pytest

from src.core.object_detection import detect_objects


def test_detect_objects_returns_detections():
    """Geçerli bir görselde nesne tespiti yapılabilmeli."""
    results = detect_objects("bus.jpg")

    assert isinstance(results, list)
    assert len(results) > 0

    for item in results:
        assert "label" in item
        assert "confidence" in item
        assert "box" in item

        assert isinstance(item["label"], str)
        assert 0 <= item["confidence"] <= 1
        assert len(item["box"]) == 4


def test_detect_objects_missing_image():
    """Olmayan görsel için FileNotFoundError oluşmalı."""
    with pytest.raises(FileNotFoundError):
        detect_objects("olmayan_gorsel.jpg")
