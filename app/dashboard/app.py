"""
dashboard/app.py
Standalone Flask dashboard for the Attendance Verification System.

Run independently:
    cd app/dashboard && python3 app.py
    or: python3 -m app.dashboard.app

Reads directly from: data/attendance_system.sqlite
Serves on: http://0.0.0.0:5000
"""

import os
import sqlite3
from flask import Flask, jsonify, render_template

# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # app/dashboard/
APP_DIR = os.path.dirname(BASE_DIR)                            # app/
PROJECT_DIR = os.path.dirname(APP_DIR)                         # project root
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DATABASE_PATH = os.path.join(DATA_DIR, 'attendance_system.sqlite')

# ─── Flask setup ─────────────────────────────────────────────────
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))


def get_db():
    """Open a read-only connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ─── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the dashboard HTML page."""
    return render_template('dashboard.html')


@app.route('/api/stats')
def api_stats():
    """
    Return summary statistics.
    {
        total_students, total_exams,
        present  (attendance_status = 1),
        incomplete (attendance_status = 0),
        absent (students with no attendance record at all)
    }
    """
    conn = get_db()
    try:
        cur = conn.cursor()

        total_students = cur.execute(
            "SELECT COUNT(*) FROM students"
        ).fetchone()[0]

        total_exams = cur.execute(
            "SELECT COUNT(*) FROM exams"
        ).fetchone()[0]

        present = cur.execute(
            "SELECT COUNT(*) FROM attendance WHERE attendance_status = 1"
        ).fetchone()[0]

        incomplete = cur.execute(
            "SELECT COUNT(*) FROM attendance WHERE attendance_status = 0"
        ).fetchone()[0]

        students_with_attendance = cur.execute(
            "SELECT COUNT(DISTINCT attendance_student_id) FROM attendance"
        ).fetchone()[0]

        absent = total_students - students_with_attendance

        return jsonify({
            "total_students": total_students,
            "total_exams": total_exams,
            "present": present,
            "incomplete": incomplete,
            "absent": absent
        })
    finally:
        conn.close()


@app.route('/api/attendance')
def api_attendance():
    """
    Return all attendance records joined with students and exams.
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                s.student_admin_number,
                s.student_full_name,
                s.student_course,
                e.exam_name,
                e.exam_module_code,
                e.exam_date,
                a.attendance_status
            FROM attendance a
            JOIN students s ON a.attendance_student_id = s.student_id
            JOIN exams e    ON a.attendance_exam_id    = e.exam_id
            ORDER BY e.exam_date DESC, s.student_admin_number ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify(rows)
    finally:
        conn.close()


@app.route('/api/exams')
def api_exams():
    """Return all exams."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT exam_id, exam_name, exam_date, exam_module_code, exam_description
            FROM exams
            ORDER BY exam_date DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify(rows)
    finally:
        conn.close()


@app.route('/api/incomplete')
def api_incomplete():
    """
    Return only attendance records with attendance_status = 0.
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                s.student_admin_number,
                s.student_full_name,
                s.student_course,
                e.exam_name,
                e.exam_module_code,
                e.exam_date,
                a.attendance_status
            FROM attendance a
            JOIN students s ON a.attendance_student_id = s.student_id
            JOIN exams e    ON a.attendance_exam_id    = e.exam_id
            WHERE a.attendance_status = 0
            ORDER BY e.exam_date DESC, s.student_admin_number ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify(rows)
    finally:
        conn.close()


# ─── Entry point ─────────────────────────────────────────────────
if __name__ == '__main__':
    if not os.path.exists(DATABASE_PATH):
        print(f"[WARNING] Database not found at {DATABASE_PATH}")
        print("  Make sure the main attendance system has been run at least once.")
    app.run(host='0.0.0.0', port=5000, debug=True)
