"""
seed_data.py
Populate the database with Group 15 team members and exam.
 
Run from project root:
    python3 seed_data.py
"""
 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from app.database import get_db_connection, DATABASE_PATH
 
print(f"Database at: {DATABASE_PATH}")
 
with get_db_connection() as conn:
    c = conn.cursor()
 
    # -- Students (names must match dataset folder names in encodings.pickle) --
    c.executemany(
        "INSERT OR IGNORE INTO students (student_admin_number, student_full_name, student_course) VALUES (?, ?, ?)",
        [
            ("2301001A", "Royce",   "ICT (SE)"),
            ("2301002B", "Rayner",  "ICT (SE)"),
            ("2301003C", "ye chen", "ICT (SE)"),
        ],
    )
    count = c.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    print(f"Students in DB: {count}")
 
    # -- Print enrolled students --
    for row in c.execute("SELECT student_id, student_admin_number, student_full_name FROM students").fetchall():
        print(f"  #{row['student_id']} | {row['student_admin_number']} | {row['student_full_name']}")
 
    # -- Exam --
    c.executemany(
        "INSERT OR IGNORE INTO exams (exam_name, exam_date, exam_module_code, exam_description) VALUES (?, ?, ?, ?)",
        [
            ("Edge Computing Midterm", "2026-03-25", "INF2009", "Midterm exam for Edge Computing & Analytics"),
        ],
    )
    count = c.execute("SELECT COUNT(*) FROM exams").fetchone()[0]
    print(f"Exams in DB: {count}")
 
    for row in c.execute("SELECT exam_id, exam_name, exam_module_code FROM exams").fetchall():
        print(f"  #{row['exam_id']} | [{row['exam_module_code']}] {row['exam_name']}")
 
print("\nDone! Run 'python3 run_system.py' from app/ to start.")