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