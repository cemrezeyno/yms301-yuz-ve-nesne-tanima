"""face_matching modülü için birim testleri."""

import numpy as np

from src.core import face_matching as fm


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(fm.cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(fm.cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    assert fm.cosine_similarity(a, b) == 0.0


def test_find_best_match_returns_closest_person():
    known = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    query = np.array([0.1, 0.9, 0.0], dtype=np.float32)
    person_id, score = fm.find_best_match(query, known)
    assert person_id == 2
    assert score > fm.COSINE_THRESHOLD


def test_find_best_match_below_threshold_is_unknown():
    known = {1: np.array([1.0, 0.0], dtype=np.float32)}
    query = np.array([0.0, 1.0], dtype=np.float32)
    person_id, _ = fm.find_best_match(query, known)
    assert person_id is None


def test_find_best_match_with_no_known_people():
    person_id, score = fm.find_best_match(np.ones(3, dtype=np.float32), {})
    assert person_id is None
    assert score == 0.0


def test_recognizer_is_loaded_once():
    assert fm.get_recognizer() is fm.get_recognizer()

import pytest

from src.core.face_recognition import FaceResult


def test_extract_embedding_requires_landmarks():
    face = FaceResult(box=(0, 0, 10, 10), name="x", confidence=0.9)
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        fm.extract_embedding(image, face)


def test_identify_faces_without_known_people_returns_faces(monkeypatch):
    face = FaceResult(box=(0, 0, 10, 10), name="Bilinmeyen kişi", confidence=0.9)
    monkeypatch.setattr(fm, "detect_faces", lambda image: [face])
    result = fm.identify_faces(np.zeros((50, 50, 3), dtype=np.uint8), [])
    assert result == [face]


def test_identify_faces_matches_known_person(monkeypatch):
    face = FaceResult(
        box=(0, 0, 10, 10), name="Bilinmeyen kişi", confidence=0.9,
        landmarks=tuple(float(i) for i in range(10)),
    )
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(fm, "detect_faces", lambda image: [face])
    monkeypatch.setattr(fm, "extract_embedding", lambda image, f: vec)
    people = [{"person_id": 7, "name": "Ayşe", "embedding": vec}]
    result = fm.identify_faces(np.zeros((50, 50, 3), dtype=np.uint8), people)
    assert result[0].person_id == 7
    assert result[0].name == "Ayşe"


def test_identify_faces_unknown_when_below_threshold(monkeypatch):
    face = FaceResult(
        box=(0, 0, 10, 10), name="Bilinmeyen kişi", confidence=0.9,
        landmarks=tuple(float(i) for i in range(10)),
    )
    monkeypatch.setattr(fm, "detect_faces", lambda image: [face])
    monkeypatch.setattr(
        fm, "extract_embedding",
        lambda image, f: np.array([0.0, 1.0], dtype=np.float32),
    )
    people = [{"person_id": 1, "name": "Ali",
               "embedding": np.array([1.0, 0.0], dtype=np.float32)}]
    result = fm.identify_faces(np.zeros((50, 50, 3), dtype=np.uint8), people)
    assert result[0].person_id is None
    assert result[0].name == "Bilinmeyen kişi"

