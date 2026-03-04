import sqlite3
from contextlib import contextmanager
import os

# Store the database in a data/ folder at the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)  # Create data/ folder if it doesn't exist
DATABASE_PATH = os.path.join(DATA_DIR, 'attendance_system.sqlite')

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def initialize_database():
    """Create database tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_admin_number VARCHAR(10) UNIQUE NOT NULL,
                student_full_name VARCHAR(512) NOT NULL,
                student_course TEXT
            );
        """)
        
        # Create face embedding table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embedding (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_student_id INTEGER,
                embedding_data TEXT NOT NULL,
                FOREIGN KEY (embedding_student_id) REFERENCES students(student_id)
            );
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
        
        # Create attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                attendance_student_id INTEGER,
                attendance_exam_id INTEGER,
                attendance_status BOOLEAN NOT NULL,
                FOREIGN KEY (attendance_student_id) REFERENCES students(student_id),
                FOREIGN KEY (attendance_exam_id) REFERENCES exams(exam_id)
            )
        """)
        
        print("Database tables initialized")

# Initialize on import
initialize_database()
               