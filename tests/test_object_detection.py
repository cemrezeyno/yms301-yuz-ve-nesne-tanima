from pathlib import Path

import pytest

from src.core.object_detection import MODEL_PATH, detect_objects

# Test görseli depoda tests/assets klasöründe durur,
# böylece testler her bilgisayarda aynı dosyayla çalışır.
TEST_IMAGE = Path(__file__).parent / "assets" / "bus.jpg"

# Model dosyası GitHub'a yüklenmediği için yoksa bu testler atlanır.
requires_model = pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason="yolo26n.pt proje ana klasöründe bulunamadı",
)


@requires_model
def test_detect_objects_returns_detections():
    """Geçerli bir görselde nesne tespiti yapılabilmeli."""
    results = detect_objects(str(TEST_IMAGE))

    assert isinstance(results, list)
    assert len(results) > 0

    for item in results:
        assert set(item.keys()) == {"label", "confidence", "box"}
        assert isinstance(item["label"], str)
        assert 0 <= item["confidence"] <= 1
        assert len(item["box"]) == 4


@requires_model
def test_detect_objects_finds_bus_and_person():
    """Test görselinde otobüs ve insan bulunmalı."""
    labels = {item["label"] for item in detect_objects(str(TEST_IMAGE))}

    assert "bus" in labels
    assert "person" in labels


@requires_model
def test_box_coordinates_are_valid():
    """Kutularda x1 < x2 ve y1 < y2 olmalı, değerler negatif olmamalı."""
    for item in detect_objects(str(TEST_IMAGE)):
        x1, y1, x2, y2 = item["box"]

        assert x1 < x2
        assert y1 < y2
        assert min(x1, y1, x2, y2) >= 0


@requires_model
def test_higher_confidence_filters_results():
    """Güven eşiği yükselince daha az (veya eşit) sonuç dönmeli."""
    all_results = detect_objects(str(TEST_IMAGE), min_confidence=0.25)
    strict_results = detect_objects(str(TEST_IMAGE), min_confidence=0.85)

    assert len(strict_results) <= len(all_results)
    assert all(item["confidence"] >= 0.85 for item in strict_results)


def test_detect_objects_missing_image():
    """Olmayan görsel için FileNotFoundError oluşmalı."""
    with pytest.raises(FileNotFoundError):
        detect_objects("olmayan_gorsel.jpg")


@pytest.mark.parametrize("bad_value", [-0.1, 1.5])
def test_invalid_confidence_raises(bad_value):
    """0-1 aralığı dışındaki güven eşiği ValueError vermeli."""
    with pytest.raises(ValueError):
        detect_objects(str(TEST_IMAGE), min_confidence=bad_value)


def test_corrupt_image_raises(tmp_path):
    """Görsel olmayan bir dosya ValueError vermeli."""
    fake_image = tmp_path / "bozuk.jpg"
    fake_image.write_text("bu bir resim değil", encoding="utf-8")

    with pytest.raises(ValueError):
        detect_objects(str(fake_image))
