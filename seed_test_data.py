"""
seed_test_data.py
Temporary script to populate the database with test data for dashboard testing.
Run from project root: python3 seed_test_data.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import get_db_connection, DATABASE_PATH

print(f"Database at: {DATABASE_PATH}")

with get_db_connection() as conn:
    c = conn.cursor()

    # Seed students
    c.executemany(
        "INSERT OR IGNORE INTO students (student_admin_number, student_full_name, student_course) VALUES (?, ?, ?)",
        [
            ("2301001A", "Alice Tan",   "ICT (SE)"),
            ("2301002B", "Bob Lim",     "ICT (SE)"),
            ("2301003C", "Charlie Ng",  "ICT (IoT)"),
            ("2301004D", "Diana Wong",  "ICT (IoT)"),
            ("2301005E", "Ethan Koh",   "ICT (AI)"),
        ],
    )
    count = c.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    print(f"Students in DB: {count}")

    # Seed exams
    c.executemany(
        "INSERT OR IGNORE INTO exams (exam_name, exam_date, exam_module_code, exam_description) VALUES (?, ?, ?, ?)",
        [
            ("Edge Computing Midterm", "2026-03-06", "INF2009", "Midterm exam for Edge Computing"),
            ("IoT Systems Final",      "2026-03-10", "INF2006", "Final exam for IoT Systems"),
        ],
    )
    count = c.execute("SELECT COUNT(*) FROM exams").fetchone()[0]
    print(f"Exams in DB: {count}")

    # Get IDs
    students = {}
    for row in c.execute("SELECT student_id, student_admin_number FROM students").fetchall():
        students[row["student_admin_number"]] = row["student_id"]

    exams = {}
    for row in c.execute("SELECT exam_id, exam_module_code FROM exams").fetchall():
        exams[row["exam_module_code"]] = row["exam_id"]

    # Seed attendance
    c.executemany(
        "INSERT OR IGNORE INTO attendance (attendance_student_id, attendance_exam_id, attendance_status) VALUES (?, ?, ?)",
        [
            (students["2301001A"], exams["INF2009"], 1),  # Alice - PRESENT
            (students["2301002B"], exams["INF2009"], 1),  # Bob - PRESENT
            (students["2301003C"], exams["INF2009"], 0),  # Charlie - INCOMPLETE
            (students["2301004D"], exams["INF2009"], 1),  # Diana - PRESENT
            # Ethan has NO attendance -> will show as ABSENT
        ],
    )
    count = c.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    print(f"Attendance records in DB: {count}")

print("Done! Test data ready.")
