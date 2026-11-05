"""face_recognition modülü için birim testleri."""

import cv2
import numpy as np
import pytest

from src.core import face_recognition as fr


@pytest.fixture
def blank_image() -> np.ndarray:
    """Yüz içermeyen düz gri BGR görsel."""
    return np.full((480, 640, 3), 127, dtype=np.uint8)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fr.analyze(tmp_path / "yok.jpg")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "not_image.txt"
    path.write_text("merhaba")
    with pytest.raises(ValueError):
        fr.analyze(path)


def test_corrupt_image_raises(tmp_path):
    path = tmp_path / "bozuk.jpg"
    path.write_text("bu bir gorsel degil")
    with pytest.raises(ValueError):
        fr.analyze(path)


@pytest.mark.parametrize(
    "bad_input",
    [np.empty((0, 0, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8), None],
)
def test_invalid_array_raises(bad_input):
    with pytest.raises(ValueError):
        fr.detect_faces(bad_input)


def test_blank_image_returns_empty_list(blank_image):
    assert fr.detect_faces(blank_image) == []


def test_analyze_accepts_path_and_array(tmp_path, blank_image):
    path = tmp_path / "bos.png"
    cv2.imwrite(str(path), blank_image)
    assert fr.analyze(path) == []
    assert fr.analyze(blank_image) == []


def test_detector_is_loaded_once():
    assert fr.get_detector() is fr.get_detector()


def test_draw_faces_does_not_modify_original(blank_image):
    result = fr.FaceResult(box=(10, 10, 50, 50), name=fr.UNKNOWN_NAME, confidence=0.9)
    original = blank_image.copy()
    drawn = fr.draw_faces(blank_image, [result])
    assert np.array_equal(blank_image, original)
    assert not np.array_equal(drawn, original)


def test_box_xyxy_converts_from_xywh():
    result = fr.FaceResult(box=(10, 20, 30, 40), name=fr.UNKNOWN_NAME, confidence=0.9)
    assert result.box_xyxy() == (10, 20, 40, 60)


def test_to_dict_matches_database_format():
    result = fr.FaceResult(box=(10, 20, 30, 40), name=fr.UNKNOWN_NAME, confidence=0.9)
    assert result.to_dict() == {
        "person_id": None,
        "confidence": 0.9,
        "box": [10, 20, 40, 60],
    }