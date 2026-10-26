
import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "project.db"


def initialize_database() -> None:
    """Create the database tables if they do not exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        connection.executescript(
            """
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
        )


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DATABASE_PATH}")
