"""src/data/database.py için birim testleri.

Çalıştırma: proje kök dizininde `pytest -v`
Her test geçici bir veritabanı kullanır; gerçek data/project.db etkilenmez.
"""

from pathlib import Path

import pytest

from src.data import database as db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    db.initialize_database(path)
    return path


def _box(x: int = 10, y: int = 20, width: int = 30, height: int = 40):
    return {"x": x, "y": y, "width": width, "height": height}


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


# ---------------------------- Analiz kaydı --------------------------------

def test_save_analysis_with_results(db_path: Path) -> None:
    person_id = db.add_person("Ayşe", db_path=db_path)
    analysis_id = db.save_analysis(
        "sinif.jpg", "both",
        face_results=[
            {"person_id": person_id, "confidence": 0.87, "box": _box()},
            {"person_id": None, "confidence": None, "box": _box(50)},
        ],
        object_results=[
            {"object_name": "chair", "confidence": 0.6, "box": _box()},
            {"object_name": "laptop", "confidence": 0.95,
             "box": _box(1.6, 2.4, 3.5, 4)},  # float koordinatlar
        ],
        db_path=db_path,
    )

    details = db.get_analysis_details(analysis_id, db_path=db_path)
    assert details is not None
    assert details["status"] == "success"
    assert [f["person_name"] for f in details["faces"]] == [
        "Ayşe", "Bilinmeyen kişi"]
    # Nesneler güven skoruna göre azalan sırada gelir
    assert [o["object_name"] for o in details["objects"]] == [
        "laptop", "chair"]
    assert details["objects"][0]["x"] == 2


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
            object_results=[{"object_name": "cup",
                             "confidence": confidence, "box": _box()}],
            db_path=db_path,
        )


def test_save_analysis_rejects_negative_box(db_path: Path) -> None:
    with pytest.raises(ValueError):
        db.save_analysis(
            "a.jpg", "object",
            object_results=[{"object_name": "cup", "confidence": 0.5,
                             "box": _box(x=-5)}],
            db_path=db_path,
        )


def test_save_analysis_rejects_missing_box_key(db_path: Path) -> None:
    with pytest.raises(ValueError):
        db.save_analysis(
            "a.jpg", "object",
            object_results=[{"object_name": "cup", "confidence": 0.5,
                             "box": {"x": 1, "y": 2}}],
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
        object_results=[{"object_name": "cup", "confidence": 0.5,
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
        object_results=[{"object_name": "cup", "confidence": 0.5,
                         "box": _box()}],
        db_path=db_path,
    )
    assert db.delete_analysis(analysis_id, db_path=db_path)
    with db.get_connection(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM object_results").fetchone()[0]
    assert count == 0
    assert not db.delete_analysis(analysis_id, db_path=db_path)
