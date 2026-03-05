import os
import sqlite3
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATA_DIR, 'attendance_system.sqlite')

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def initialize_database():
    """Create database tables with CASCADE foreign keys"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_admin_number VARCHAR(10) UNIQUE NOT NULL,
                student_full_name VARCHAR(512) NOT NULL,
                student_course TEXT
            )
        """)

        # Create exams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exams (
                exam_id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_name VARCHAR(512) NOT NULL,
                exam_date DATE NOT NULL,
                exam_module_code VARCHAR(15) NOT NULL,
                exam_description TEXT
            )
        """)

        # Migrate face_embedding to include ON DELETE CASCADE if not already present
        schema = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='face_embedding'"
        ).fetchone()
        if schema is None or "ON DELETE CASCADE" not in (schema["sql"] or ""):
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embedding_new (
                    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_student_id INTEGER,
                    embedding_data TEXT NOT NULL,
                    FOREIGN KEY (embedding_student_id) REFERENCES students(student_id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO face_embedding_new
                SELECT * FROM face_embedding
            """ if schema else "SELECT 1")  # only copy if old table existed
            if schema:
                cursor.execute("DROP TABLE face_embedding")
            cursor.execute("ALTER TABLE face_embedding_new RENAME TO face_embedding")
            cursor.execute("PRAGMA foreign_keys = ON")

        # Migrate attendance to include ON DELETE CASCADE if not already present
        schema = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='attendance'"
        ).fetchone()
        if schema is None or "ON DELETE CASCADE" not in (schema["sql"] or ""):
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_new (
                    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attendance_student_id INTEGER,
                    attendance_exam_id INTEGER,
                    attendance_status BOOLEAN NOT NULL,
                    FOREIGN KEY (attendance_student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                    FOREIGN KEY (attendance_exam_id) REFERENCES exams(exam_id) ON DELETE CASCADE
                )
            """)
            if schema:
                cursor.execute("INSERT OR IGNORE INTO attendance_new SELECT * FROM attendance")
                cursor.execute("DROP TABLE attendance")
            cursor.execute("ALTER TABLE attendance_new RENAME TO attendance")
            cursor.execute("PRAGMA foreign_keys = ON")

        print("Database tables initialized with CASCADE foreign keys")

# Initialize on import
initialize_database()
               