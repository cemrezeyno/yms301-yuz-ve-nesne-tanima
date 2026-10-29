"""SQLite veritabanı katmanı.

Kişi, analiz ve analiz sonuçlarının (yüz / nesne) saklanması ve
sorgulanmasından sorumludur. Diğer modüller veritabanına doğrudan SQL
yazmak yerine bu dosyadaki fonksiyonları kullanmalıdır.
"""

import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "project.db"

AnalysisType = Literal["face", "object", "both"]
VALID_ANALYSIS_TYPES: frozenset[str] = frozenset({"face", "object", "both"})


class DatabaseError(Exception):
    """Veritabanı işlemleri sırasında oluşan hatalar için ortak istisna."""


class BoundingBox(TypedDict):
    """Görsel üzerindeki kutunun piksel cinsinden konumu."""

    x: int
    y: int
    width: int
    height: int


class FaceResult(TypedDict):
    """Yüz modülünden beklenen tek bir sonuç.

    person_id None ise kişi "Bilinmeyen kişi" kabul edilir.
    """

    person_id: int | None
    confidence: float | None
    box: BoundingBox


class ObjectResult(TypedDict):
    """Nesne modülünden beklenen tek bir sonuç."""

    object_name: str
    confidence: float
    box: BoundingBox


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS persons (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT NOT NULL,
    analysis_type TEXT NOT NULL
        CHECK (analysis_type IN ('face', 'object', 'both')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'success'
        CHECK (status IN ('success', 'failed')),
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS face_results (
    face_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    person_id INTEGER,
    confidence REAL
        CHECK (confidence IS NULL OR
               (confidence >= 0 AND confidence <= 1)),
    x INTEGER NOT NULL CHECK (x >= 0),
    y INTEGER NOT NULL CHECK (y >= 0),
    width INTEGER NOT NULL CHECK (width >= 0),
    height INTEGER NOT NULL CHECK (height >= 0),
    FOREIGN KEY (analysis_id)
        REFERENCES analyses(analysis_id)
        ON DELETE CASCADE,
    FOREIGN KEY (person_id)
        REFERENCES persons(person_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS object_results (
    object_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    object_name TEXT NOT NULL,
    confidence REAL NOT NULL
        CHECK (confidence >= 0 AND confidence <= 1),
    x INTEGER NOT NULL CHECK (x >= 0),
    y INTEGER NOT NULL CHECK (y >= 0),
    width INTEGER NOT NULL CHECK (width >= 0),
    height INTEGER NOT NULL CHECK (height >= 0),
    FOREIGN KEY (analysis_id)
        REFERENCES analyses(analysis_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_analysis
    ON face_results(analysis_id);

CREATE INDEX IF NOT EXISTS idx_object_analysis
    ON object_results(analysis_id);

CREATE INDEX IF NOT EXISTS idx_analysis_date
    ON analyses(created_at);
"""


# --------------------------------------------------------------------------
# Bağlantı ve kurulum
# --------------------------------------------------------------------------

@contextmanager
def get_connection(
    db_path: Path = DATABASE_PATH,
) -> Iterator[sqlite3.Connection]:
    """Yapılandırılmış bir bağlantı açar; işlem bitince commit edip kapatır.

    Hata olursa rollback yapılır ve hata DatabaseError olarak yükseltilir.
    PRAGMA foreign_keys bağlantıya özel olduğu için her bağlantıda açılır.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        logger.exception("Veritabanı işlemi başarısız oldu")
        raise DatabaseError(str(error)) from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    """Tablolar yoksa oluşturur."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
    logger.info("Veritabanı hazır: %s", db_path)


# --------------------------------------------------------------------------
# Doğrulama yardımcıları
# --------------------------------------------------------------------------

def _last_id(cursor: sqlite3.Cursor) -> int:
    """INSERT sonrası oluşan satırın id değerini döndürür."""
    if cursor.lastrowid is None:
        raise DatabaseError("Eklenen kaydın id değeri alınamadı.")
    return cursor.lastrowid


def _validate_confidence(confidence: float | None, allow_none: bool) -> None:
    if confidence is None:
        if not allow_none:
            raise ValueError("confidence boş olamaz.")
        return
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"confidence 0 ile 1 arasında olmalı, gelen: {confidence}"
        )


def _box_values(box: BoundingBox) -> tuple[int, int, int, int]:
    """Kutuyu doğrular ve (x, y, width, height) tamsayılarına çevirir."""
    try:
        values = tuple(
            int(round(box[key])) for key in ("x", "y", "width", "height")
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Geçersiz kutu verisi: {box}") from error
    if any(value < 0 for value in values):
        raise ValueError(f"Kutu değerleri negatif olamaz: {box}")
    return values  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Kişi işlemleri
# --------------------------------------------------------------------------

def add_person(name: str, db_path: Path = DATABASE_PATH) -> int:
    """Yeni bir kişi ekler ve person_id değerini döndürür."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Kişi adı boş olamaz.")

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO persons (name) VALUES (?)", (clean_name,)
        )
        person_id = _last_id(cursor)
    logger.info("Kişi eklendi: %s (id=%d)", clean_name, person_id)
    return person_id


def get_person(
    person_id: int, db_path: Path = DATABASE_PATH
) -> dict[str, Any] | None:
    """Kişiyi döndürür; bulunamazsa None."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()
    return dict(row) if row else None


def list_persons(
    only_active: bool = True, db_path: Path = DATABASE_PATH
) -> list[dict[str, Any]]:
    """Kişileri ada göre sıralı listeler."""
    query = "SELECT * FROM persons"
    if only_active:
        query += " WHERE is_active = 1"
    query += " ORDER BY name"
    with get_connection(db_path) as connection:
        rows = connection.execute(query).fetchall()
    return [dict(row) for row in rows]


def set_person_active(
    person_id: int, is_active: bool, db_path: Path = DATABASE_PATH
) -> bool:
    """Kişiyi aktif/pasif yapar. Kişi bulunduysa True döner."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE persons SET is_active = ? WHERE person_id = ?",
            (int(is_active), person_id),
        )
    return cursor.rowcount > 0


def delete_person(person_id: int, db_path: Path = DATABASE_PATH) -> bool:
    """Kişiyi tamamen siler (rıza geri çekildiğinde kullanılır).

    Geçmiş yüz sonuçlarındaki person_id otomatik olarak NULL olur.
    """
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM persons WHERE person_id = ?", (person_id,)
        )
    return cursor.rowcount > 0


# --------------------------------------------------------------------------
# Analiz kayıt işlemleri
# --------------------------------------------------------------------------

def save_analysis(
    image_name: str,
    analysis_type: AnalysisType,
    face_results: Sequence[FaceResult] = (),
    object_results: Sequence[ObjectResult] = (),
    db_path: Path = DATABASE_PATH,
) -> int:
    """Başarılı bir analizi ve tüm sonuçlarını tek işlemde kaydeder.

    Sonuçlardan biri hatalıysa hiçbir şey kaydedilmez (transaction).
    analysis_id döndürür.
    """
    if not image_name.strip():
        raise ValueError("Görsel adı boş olamaz.")
    if analysis_type not in VALID_ANALYSIS_TYPES:
        raise ValueError(f"Geçersiz analiz türü: {analysis_type}")

    face_rows = []
    for face in face_results:
        _validate_confidence(face.get("confidence"), allow_none=True)
        face_rows.append(
            (face.get("person_id"), face.get("confidence"),
             *_box_values(face["box"]))
        )

    object_rows = []
    for obj in object_results:
        object_name = str(obj.get("object_name", "")).strip()
        if not object_name:
            raise ValueError("Nesne adı boş olamaz.")
        _validate_confidence(obj.get("confidence"), allow_none=False)
        object_rows.append(
            (object_name, float(obj["confidence"]), *_box_values(obj["box"]))
        )

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO analyses (image_name, analysis_type) VALUES (?, ?)",
            (image_name.strip(), analysis_type),
        )
        analysis_id = _last_id(cursor)

        connection.executemany(
            """
            INSERT INTO face_results
                (analysis_id, person_id, confidence, x, y, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(analysis_id, *row) for row in face_rows],
        )
        connection.executemany(
            """
            INSERT INTO object_results
                (analysis_id, object_name, confidence, x, y, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(analysis_id, *row) for row in object_rows],
        )

    logger.info(
        "Analiz kaydedildi (id=%d): %d yüz, %d nesne",
        analysis_id, len(face_rows), len(object_rows),
    )
    return analysis_id


def save_failed_analysis(
    image_name: str,
    analysis_type: AnalysisType,
    error_message: str,
    db_path: Path = DATABASE_PATH,
) -> int:
    """Başarısız bir analizi hata mesajıyla birlikte kaydeder."""
    if analysis_type not in VALID_ANALYSIS_TYPES:
        raise ValueError(f"Geçersiz analiz türü: {analysis_type}")

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses
                (image_name, analysis_type, status, error_message)
            VALUES (?, ?, 'failed', ?)
            """,
            (image_name.strip() or "bilinmeyen", analysis_type,
             error_message),
        )
        analysis_id = _last_id(cursor)
    logger.warning(
        "Başarısız analiz kaydedildi (id=%d): %s", analysis_id, error_message
    )
    return analysis_id


# --------------------------------------------------------------------------
# Sorgulama işlemleri
# --------------------------------------------------------------------------

def get_analysis_history(
    limit: int = 50,
    analysis_type: AnalysisType | None = None,
    db_path: Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Analiz geçmişini en yeniden eskiye, sonuç sayılarıyla döndürür."""
    if limit <= 0:
        raise ValueError("limit pozitif olmalı.")

    query = """
        SELECT a.*,
               (SELECT COUNT(*) FROM face_results f
                 WHERE f.analysis_id = a.analysis_id) AS face_count,
               (SELECT COUNT(*) FROM object_results o
                 WHERE o.analysis_id = a.analysis_id) AS object_count
        FROM analyses a
    """
    params: list[Any] = []
    if analysis_type is not None:
        if analysis_type not in VALID_ANALYSIS_TYPES:
            raise ValueError(f"Geçersiz analiz türü: {analysis_type}")
        query += " WHERE a.analysis_type = ?"
        params.append(analysis_type)
    query += " ORDER BY a.created_at DESC, a.analysis_id DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_analysis_details(
    analysis_id: int, db_path: Path = DATABASE_PATH
) -> dict[str, Any] | None:
    """Bir analizin tüm yüz ve nesne sonuçlarını döndürür.

    Yüz sonuçlarında person_name alanı bulunur; eşleşme yoksa
    "Bilinmeyen kişi" yazılır.
    """
    with get_connection(db_path) as connection:
        analysis = connection.execute(
            "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
        ).fetchone()
        if analysis is None:
            return None

        faces = connection.execute(
            """
            SELECT f.*, COALESCE(p.name, 'Bilinmeyen kişi') AS person_name
            FROM face_results f
            LEFT JOIN persons p ON p.person_id = f.person_id
            WHERE f.analysis_id = ?
            ORDER BY f.face_result_id
            """,
            (analysis_id,),
        ).fetchall()
        objects = connection.execute(
            """
            SELECT * FROM object_results
            WHERE analysis_id = ?
            ORDER BY confidence DESC
            """,
            (analysis_id,),
        ).fetchall()

    result = dict(analysis)
    result["faces"] = [dict(row) for row in faces]
    result["objects"] = [dict(row) for row in objects]
    return result


def delete_analysis(analysis_id: int, db_path: Path = DATABASE_PATH) -> bool:
    """Analizi ve (CASCADE ile) tüm sonuçlarını siler."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,)
        )
    return cursor.rowcount > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_database()
    print(f"Database initialized: {DATABASE_PATH}")
