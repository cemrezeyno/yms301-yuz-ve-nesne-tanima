"""src/data/database.py için birim testleri.

Çalıştırma: proje kök dizininde `pytest -v`
Her test geçici bir veritabanı kullanır; gerçek data/project.db etkilenmez.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.data import database as db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    db.initialize_database(path)
    return path


def _box(x1: float = 10, y1: float = 20, x2: float = 40,
         y2: float = 60) -> list[float]:
    """Nesne modülünün döndürdüğü [x1, y1, x2, y2] formatında kutu."""
    return [x1, y1, x2, y2]


# ---------------------------- Kurulum -------------------------------------

def test_initialize_creates_tables(db_path: Path) -> None:
    with db.get_connection(db_path) as connection:
        tables = {
            row["name"] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"persons", "analyses", "face_results",
            "object_results"} <= tables


def test_initialize_is_idempotent(db_path: Path) -> None:
    db.initialize_database(db_path)  # ikinci kez hata vermemeli


def test_foreign_keys_enabled(db_path: Path) -> None:
    with db.get_connection(db_path) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# ---------------------------- Kişiler -------------------------------------

def test_add_and_get_person(db_path: Path) -> None:
    person_id = db.add_person("  Ayşe  ", db_path=db_path)
    person = db.get_person(person_id, db_path=db_path)
    assert person is not None
    assert person["name"] == "Ayşe"
    assert person["is_active"] == 1


def test_add_person_rejects_empty_name(db_path: Path) -> None:
    with pytest.raises(ValueError):
        db.add_person("   ", db_path=db_path)


def test_get_missing_person_returns_none(db_path: Path) -> None:
    assert db.get_person(999, db_path=db_path) is None


def test_list_persons_filters_inactive(db_path: Path) -> None:
    ali = db.add_person("Ali", db_path=db_path)
    db.add_person("Zeynep", db_path=db_path)
    db.set_person_active(ali, False, db_path=db_path)

    active = db.list_persons(db_path=db_path)
    everyone = db.list_persons(only_active=False, db_path=db_path)

    assert [p["name"] for p in active] == ["Zeynep"]
    assert len(everyone) == 2


def test_delete_person_sets_face_result_to_unknown(db_path: Path) -> None:
    person_id = db.add_person("Mehmet", db_path=db_path)
    analysis_id = db.save_analysis(
        "foto.jpg", "face",
        face_results=[{"person_id": person_id, "confidence": 0.9,
                       "box": _box()}],
        db_path=db_path,
    )
    assert db.delete_person(person_id, db_path=db_path)

    details = db.get_analysis_details(analysis_id, db_path=db_path)
    assert details is not None
    assert details["faces"][0]["person_id"] is None
    assert details["faces"][0]["person_name"] == "Bilinmeyen kişi"


# ---------------------------- Yüz embedding -------------------------------

def _embedding(seed: int = 0, size: int = 128) -> np.ndarray:
    return np.random.default_rng(seed).random(size, dtype=np.float32)


def test_add_person_with_embedding_roundtrip(db_path: Path) -> None:
    embedding = _embedding()
    person_id = db.add_person("Ayşe", face_embedding=embedding,
                              db_path=db_path)
    stored = db.get_face_embedding(person_id, db_path=db_path)
    assert stored is not None
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored, embedding)


def test_embedding_accepts_list_and_2d_row(db_path: Path) -> None:
    first = db.add_person("A", face_embedding=[0.1, 0.2, 0.3],
                          db_path=db_path)
    second = db.add_person("B", face_embedding=np.ones((1, 3)),
                           db_path=db_path)
    assert db.get_face_embedding(first, db_path=db_path).shape == (3,)
    assert db.get_face_embedding(second, db_path=db_path).shape == (3,)


def test_person_without_embedding(db_path: Path) -> None:
    person_id = db.add_person("Ali", db_path=db_path)
    assert db.get_face_embedding(person_id, db_path=db_path) is None
    person = db.get_person(person_id, db_path=db_path)
    assert person is not None and person["has_face_embedding"] == 0


def test_person_queries_do_not_return_raw_embedding(db_path: Path) -> None:
    person_id = db.add_person("Ayşe", face_embedding=_embedding(),
                              db_path=db_path)
    person = db.get_person(person_id, db_path=db_path)
    listed = db.list_persons(db_path=db_path)[0]
    for record in (person, listed):
        assert record is not None
        assert "face_embedding" not in record
        assert record["has_face_embedding"] == 1


def test_set_and_clear_embedding(db_path: Path) -> None:
    person_id = db.add_person("Ali", db_path=db_path)
    assert db.set_face_embedding(person_id, _embedding(1), db_path=db_path)
    np.testing.assert_array_equal(
        db.get_face_embedding(person_id, db_path=db_path), _embedding(1))

    # Aynı kişinin embedding'i güncellenebilir
    assert db.set_face_embedding(person_id, _embedding(2), db_path=db_path)
    np.testing.assert_array_equal(
        db.get_face_embedding(person_id, db_path=db_path), _embedding(2))

    assert db.clear_face_embedding(person_id, db_path=db_path)
    assert db.get_face_embedding(person_id, db_path=db_path) is None
    assert db.get_person(person_id, db_path=db_path) is not None


def test_embedding_functions_on_missing_person(db_path: Path) -> None:
    assert not db.set_face_embedding(999, _embedding(), db_path=db_path)
    assert not db.clear_face_embedding(999, db_path=db_path)
    assert db.get_face_embedding(999, db_path=db_path) is None


@pytest.mark.parametrize("bad", [
    [],                         # boş
    [[1.0, 2.0], [3.0, 4.0]],   # iki boyutlu
    [1.0, float("nan")],        # NaN
    [1.0, float("inf")],        # sonsuz
    ["a", "b"],                 # sayı değil
])
def test_invalid_embedding_rejected(db_path: Path, bad) -> None:
    with pytest.raises(ValueError):
        db.add_person("X", face_embedding=bad, db_path=db_path)
    assert db.list_persons(only_active=False, db_path=db_path) == []


def test_embedding_size_must_match_existing(db_path: Path) -> None:
    db.add_person("A", face_embedding=_embedding(size=128), db_path=db_path)
    other = db.add_person("B", db_path=db_path)
    with pytest.raises(ValueError, match="boyut"):
        db.set_face_embedding(other, _embedding(size=512), db_path=db_path)
    with pytest.raises(ValueError, match="boyut"):
        db.add_person("C", face_embedding=_embedding(size=512),
                      db_path=db_path)


def test_single_person_can_change_embedding_size(db_path: Path) -> None:
    """Tek kayıtlı embedding kendi kişisinde farklı boyutla değişebilir."""
    person_id = db.add_person("A", face_embedding=_embedding(size=128),
                              db_path=db_path)
    assert db.set_face_embedding(person_id, _embedding(size=512),
                                 db_path=db_path)


def test_known_embeddings_only_active_with_embedding(db_path: Path) -> None:
    ayse = db.add_person("Ayşe", face_embedding=_embedding(1),
                         db_path=db_path)
    ali = db.add_person("Ali", face_embedding=_embedding(2), db_path=db_path)
    db.add_person("Embedding'siz", db_path=db_path)
    db.set_person_active(ali, False, db_path=db_path)

    known = db.get_known_face_embeddings(db_path=db_path)
    assert [k["person_id"] for k in known] == [ayse]
    assert known[0]["name"] == "Ayşe"
    np.testing.assert_array_equal(known[0]["embedding"], _embedding(1))


def test_delete_person_removes_embedding(db_path: Path) -> None:
    person_id = db.add_person("Ayşe", face_embedding=_embedding(),
                              db_path=db_path)
    db.delete_person(person_id, db_path=db_path)
    assert db.get_known_face_embeddings(db_path=db_path) == []


def test_initialize_adds_column_to_week2_database(tmp_path: Path) -> None:
    """face_embedding sütunu olmayan eski veritabanı veri kaybetmeden
    güncellenir."""
    old_db = tmp_path / "week2.db"
    connection = sqlite3.connect(old_db)
    connection.executescript(
        db.SCHEMA_SQL.replace(",\n    face_embedding BLOB", "")
    )
    connection.execute("INSERT INTO persons (name) VALUES ('Eski Kayıt')")
    connection.commit()
    connection.close()

    db.initialize_database(old_db)

    persons = db.list_persons(db_path=old_db)
    assert [p["name"] for p in persons] == ["Eski Kayıt"]
    assert db.set_face_embedding(persons[0]["person_id"], _embedding(),
                                 db_path=old_db)


# ---------------------------- Analiz kaydı --------------------------------

def test_save_analysis_with_results(db_path: Path) -> None:
    person_id = db.add_person("Ayşe", db_path=db_path)
    analysis_id = db.save_analysis(
        "sinif.jpg", "both",
        face_results=[
            {"person_id": person_id, "confidence": 0.87, "box": _box()},
            {"person_id": None, "confidence": None,
             "box": _box(50, 20, 90, 60)},
        ],
        object_results=[
            {"label": "chair", "confidence": 0.6, "box": _box()},
            {"label": "laptop", "confidence": 0.95,
             "box": _box(1.6, 2.4, 300.5, 400)},  # ondalıklı değerler
        ],
        db_path=db_path,
    )

    details = db.get_analysis_details(analysis_id, db_path=db_path)
    assert details is not None
    assert details["status"] == "success"
    assert [f["person_name"] for f in details["faces"]] == [
        "Ayşe", "Bilinmeyen kişi"]
    # Nesneler güven skoruna göre azalan sırada gelir
    assert [o["label"] for o in details["objects"]] == [
        "laptop", "chair"]
    laptop = details["objects"][0]
    assert (laptop["x1"], laptop["y1"], laptop["x2"], laptop["y2"]) == (
        2, 2, 300, 400)


def test_save_analysis_without_results(db_path: Path) -> None:
    analysis_id = db.save_analysis("bos.jpg", "object", db_path=db_path)
    details = db.get_analysis_details(analysis_id, db_path=db_path)
    assert details is not None
    assert details["faces"] == [] and details["objects"] == []


@pytest.mark.parametrize("analysis_type", ["", "car", "FACE"])
def test_save_analysis_rejects_invalid_type(
    db_path: Path, analysis_type: str
) -> None:
    with pytest.raises(ValueError):
        db.save_analysis("a.jpg", analysis_type,  # type: ignore[arg-type]
                         db_path=db_path)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_save_analysis_rejects_bad_confidence(
    db_path: Path, confidence: float
) -> None:
    with pytest.raises(ValueError):
        db.save_analysis(
            "a.jpg", "object",
            object_results=[{"label": "cup",
                             "confidence": confidence, "box": _box()}],
            db_path=db_path,
        )


def test_save_analysis_rejects_negative_box(db_path: Path) -> None:
    with pytest.raises(ValueError):
        db.save_analysis(
            "a.jpg", "object",
            object_results=[{"label": "cup", "confidence": 0.5,
                             "box": _box(x1=-5)}],
            db_path=db_path,
        )


def test_save_analysis_rejects_missing_box_key(db_path: Path) -> None:
    with pytest.raises(ValueError):
        db.save_analysis(
            "a.jpg", "object",
            object_results=[{"label": "cup", "confidence": 0.5,
                             "box": [1, 2, 3]}],
            db_path=db_path,
        )


def test_unknown_person_id_rolls_back_everything(db_path: Path) -> None:
    """Var olmayan person_id FK hatası verir ve analiz de kaydedilmez."""
    with pytest.raises(db.DatabaseError):
        db.save_analysis(
            "a.jpg", "face",
            face_results=[{"person_id": 999, "confidence": 0.5,
                           "box": _box()}],
            db_path=db_path,
        )
    assert db.get_analysis_history(db_path=db_path) == []


def test_box_accepts_dict_and_tuple(db_path: Path) -> None:
    analysis_id = db.save_analysis(
        "a.jpg", "object",
        object_results=[
            {"label": "cup", "confidence": 0.5,
             "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
            {"label": "pen", "confidence": 0.4, "box": (5, 6, 7, 8)},
        ],
        db_path=db_path,
    )
    details = db.get_analysis_details(analysis_id, db_path=db_path)
    assert details is not None
    boxes = [(o["x1"], o["y1"], o["x2"], o["y2"])
             for o in details["objects"]]
    assert boxes == [(1, 2, 3, 4), (5, 6, 7, 8)]


@pytest.mark.parametrize("box", [
    [50, 20, 10, 60],   # x2 < x1
    [10, 60, 40, 20],   # y2 < y1
    "10,20,40,60",      # metin
    {"x": 1, "y": 2, "width": 3, "height": 4},  # eski format
])
def test_save_analysis_rejects_invalid_box(db_path: Path, box) -> None:
    with pytest.raises(ValueError):
        db.save_analysis(
            "a.jpg", "object",
            object_results=[{"label": "cup", "confidence": 0.5,
                             "box": box}],
            db_path=db_path,
        )


def test_save_analysis_rejects_empty_label(db_path: Path) -> None:
    with pytest.raises(ValueError):
        db.save_analysis(
            "a.jpg", "object",
            object_results=[{"label": "  ", "confidence": 0.5,
                             "box": _box()}],
            db_path=db_path,
        )


def test_initialize_rejects_old_schema(tmp_path: Path) -> None:
    old_db = tmp_path / "old.db"
    with db.get_connection(old_db) as connection:
        connection.execute(
            "CREATE TABLE object_results (x INTEGER, y INTEGER)")
    with pytest.raises(db.DatabaseError, match="eski"):
        db.initialize_database(old_db)


def test_save_failed_analysis(db_path: Path) -> None:
    analysis_id = db.save_failed_analysis(
        "bozuk.png", "face", "Görsel okunamadı", db_path=db_path
    )
    details = db.get_analysis_details(analysis_id, db_path=db_path)
    assert details is not None
    assert details["status"] == "failed"
    assert details["error_message"] == "Görsel okunamadı"


# ---------------------------- Sorgulama -----------------------------------

def test_history_order_counts_and_filter(db_path: Path) -> None:
    first = db.save_analysis(
        "1.jpg", "object",
        object_results=[{"label": "cup", "confidence": 0.5,
                         "box": _box()}],
        db_path=db_path,
    )
    second = db.save_analysis(
        "2.jpg", "face",
        face_results=[{"person_id": None, "confidence": None,
                       "box": _box()}] * 2,
        db_path=db_path,
    )

    history = db.get_analysis_history(db_path=db_path)
    assert [h["analysis_id"] for h in history] == [second, first]
    assert history[0]["face_count"] == 2
    assert history[1]["object_count"] == 1

    only_objects = db.get_analysis_history(
        analysis_type="object", db_path=db_path)
    assert [h["analysis_id"] for h in only_objects] == [first]

    assert len(db.get_analysis_history(limit=1, db_path=db_path)) == 1


def test_history_rejects_bad_limit(db_path: Path) -> None:
    with pytest.raises(ValueError):
        db.get_analysis_history(limit=0, db_path=db_path)


def test_get_missing_analysis_returns_none(db_path: Path) -> None:
    assert db.get_analysis_details(123, db_path=db_path) is None


def test_delete_analysis_cascades(db_path: Path) -> None:
    analysis_id = db.save_analysis(
        "a.jpg", "object",
        object_results=[{"label": "cup", "confidence": 0.5,
                         "box": _box()}],
        db_path=db_path,
    )
    assert db.delete_analysis(analysis_id, db_path=db_path)
    with db.get_connection(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM object_results").fetchone()[0]
    assert count == 0
    assert not db.delete_analysis(analysis_id, db_path=db_path)
