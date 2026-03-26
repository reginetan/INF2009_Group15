"""
clear_data.py
Delete all data from the attendance system database.
Comment out any tables you want to keep.

Usage:
    python clear_data.py
"""

import os
import sys
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATABASE_PATH = os.path.join(DATA_DIR, 'attendance_system.sqlite')
IMAGES_DIR = os.path.join(BASE_DIR, 'app', 'images')

if not os.path.exists(DATABASE_PATH):
    print(f"Database not found at {DATABASE_PATH}")
    sys.exit(1)

import sqlite3

conn = sqlite3.connect(DATABASE_PATH)
conn.execute("PRAGMA foreign_keys = ON")
cursor = conn.cursor()

# Show current counts
for table in ['students', 'exams', 'attendance', 'face_embedding']:
    try:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")
    except sqlite3.OperationalError:
        print(f"  {table}: table does not exist")

print()
confirm = input("Are you sure you want to delete ALL data? (yes/no): ").strip().lower()
if confirm != 'yes':
    print("Aborted.")
    conn.close()
    sys.exit(0)

# Delete in order that respects foreign keys
# attendance depends on students & exams, face_embedding depends on students
cursor.execute("DELETE FROM attendance")
print("Deleted all attendance records.")

cursor.execute("DELETE FROM face_embedding")
print("Deleted all face embeddings.")

cursor.execute("DELETE FROM exams")
print("Deleted all exams.")

cursor.execute("DELETE FROM students")
print("Deleted all students.")

conn.commit()
conn.close()

# Remove uploaded face images
removed = 0
for f in glob.glob(os.path.join(IMAGES_DIR, '*')):
    if os.path.isfile(f):
        os.remove(f)
        removed += 1
if removed:
    print(f"Removed {removed} image file(s) from app/images/.")

print("Done. All data cleared.")
