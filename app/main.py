import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date
from dotenv import load_dotenv
from app.database import get_db_connection

load_dotenv()

# ==================== API KEY SECURITY ====================
API_KEY = os.environ.get("API_KEY", "changeme")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key


# Ensure images directory exists
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="Attendance System API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class StudentCreate(BaseModel):
    student_admin_number: str
    student_full_name: str
    student_course: Optional[str] = None

class StudentUpdate(BaseModel):
    student_admin_number: Optional[str] = None
    student_full_name: Optional[str] = None
    student_course: Optional[str] = None

class ExamCreate(BaseModel):
    exam_name: str
    exam_date: date
    exam_module_code: str
    exam_description: Optional[str] = None

class ExamUpdate(BaseModel):
    exam_name: Optional[str] = None
    exam_date: Optional[date] = None
    exam_module_code: Optional[str] = None
    exam_description: Optional[str] = None

class AttendanceCreate(BaseModel):
    attendance_student_id: int
    attendance_exam_id: int
    attendance_status: bool

class AttendanceUpdate(BaseModel):
    attendance_status: bool

# ==================== STUDENT ENDPOINTS ====================

@app.get("/api/students", dependencies=[Depends(verify_api_key)])
def get_students():
    """Get all students"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students")
        students = [dict(row) for row in cursor.fetchall()]
        return {"students": students}

@app.get("/api/students/{student_id}", dependencies=[Depends(verify_api_key)]
         )
def get_student(student_id: int):
    """Get student by ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        student = cursor.fetchone()

        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        return {"student": dict(student)}

@app.post("/api/students", status_code=201, dependencies=[Depends(verify_api_key)])
def create_student(student: StudentCreate):
    """Create a new student"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO students (student_admin_number, student_full_name, student_course) VALUES (?, ?, ?)",
                (student.student_admin_number, student.student_full_name, student.student_course)
            )
            student_id = cursor.lastrowid
            return {
                "student_id": student_id,
                "student_admin_number": student.student_admin_number,
                "student_full_name": student.student_full_name,
                "student_course": student.student_course
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/students/{student_id}", dependencies=[Depends(verify_api_key)])
def update_student(student_id: int, student: StudentUpdate):
    """Update a student"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        updates = []
        params = []
        if student.student_admin_number is not None:
            updates.append("student_admin_number = ?")
            params.append(student.student_admin_number)
        if student.student_full_name is not None:
            updates.append("student_full_name = ?")
            params.append(student.student_full_name)
        if student.student_course is not None:
            updates.append("student_course = ?")
            params.append(student.student_course)

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        params.append(student_id)
        query = f"UPDATE students SET {', '.join(updates)} WHERE student_id = ?"
        cursor.execute(query, params)

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Student not found")

        return {"message": "Student updated successfully"}

@app.delete("/api/students/{student_id}", dependencies=[Depends(verify_api_key)])
def delete_student(student_id: int):
    """Delete a student and their associated image files"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Fetch image paths before cascade delete removes them
        cursor.execute(
            "SELECT embedding_data FROM face_embedding WHERE embedding_student_id = ?",
            (student_id,)
        )
        image_paths = [row["embedding_data"] for row in cursor.fetchall()]

        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Student not found")

        # Remove image files from disk
        for image_path in image_paths:
            abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)
            if os.path.exists(abs_path):
                os.remove(abs_path)

        return {"message": "Student deleted successfully"}

# ==================== FACE EMBEDDING ENDPOINTS ====================

@app.get("/api/embeddings", dependencies=[Depends(verify_api_key)])
def get_embeddings():
    """Get all face embeddings"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT face_embedding.*, students.student_full_name
            FROM face_embedding
            LEFT JOIN students ON face_embedding.embedding_student_id = students.student_id
        """)
        embeddings = [dict(row) for row in cursor.fetchall()]
        return {"embeddings": embeddings}

@app.get("/api/embeddings/{student_id}", dependencies=[Depends(verify_api_key)])
def get_embedding_by_student(student_id: int):
    """Get face embedding by student ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM face_embedding WHERE embedding_student_id = ?", (student_id,)
        )
        embedding = cursor.fetchone()

        if not embedding:
            raise HTTPException(status_code=404, detail="Face embedding not found for this student")

        return {"embedding": dict(embedding)}

@app.post("/api/embeddings", status_code=201, dependencies=[Depends(verify_api_key)])
async def create_embedding(
    embedding_student_id: int = Form(...),
    image: UploadFile = File(...)
):
    """Upload a student face image and save it to app/images/"""
    # Validate file is an image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save image as {student_id}_{original_filename}
    ext = os.path.splitext(image.filename)[1]
    filename = f"{embedding_student_id}{ext}"
    file_path = os.path.join(IMAGES_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Store relative image path in embedding_data
    relative_path = f"app/images/{filename}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO face_embedding (embedding_student_id, embedding_data) VALUES (?, ?)",
                (embedding_student_id, relative_path)
            )
            embedding_id = cursor.lastrowid
            return {
                "embedding_id": embedding_id,
                "embedding_student_id": embedding_student_id,
                "image_path": relative_path
            }
        except Exception as e:
            # Remove saved file if DB insert fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/embeddings/{embedding_id}", dependencies=[Depends(verify_api_key)])
def delete_embedding(embedding_id: int):
    """Delete a face embedding and its associated image file"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Fetch image path before deleting
        cursor.execute(
            "SELECT embedding_data FROM face_embedding WHERE embedding_id = ?",
            (embedding_id,)
        )
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Face embedding not found")

        image_path = row["embedding_data"]
        cursor.execute("DELETE FROM face_embedding WHERE embedding_id = ?", (embedding_id,))

        # Remove image file from disk
        abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)

        return {"message": "Face embedding deleted successfully"}

# ==================== EXAM ENDPOINTS ====================

@app.get("/api/exams", dependencies=[Depends(verify_api_key)])
def get_exams():
    """Get all exams"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM exams")
        exams = [dict(row) for row in cursor.fetchall()]
        return {"exams": exams}

@app.get("/api/exams/{exam_id}", dependencies=[Depends(verify_api_key)])
def get_exam(exam_id: int):
    """Get exam by ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM exams WHERE exam_id = ?", (exam_id,))
        exam = cursor.fetchone()

        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")

        return {"exam": dict(exam)}

@app.post("/api/exams", status_code=201, dependencies=[Depends(verify_api_key)])
def create_exam(exam: ExamCreate):
    """Create a new exam"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO exams (exam_name, exam_date, exam_module_code, exam_description) VALUES (?, ?, ?, ?)",
                (exam.exam_name, str(exam.exam_date), exam.exam_module_code, exam.exam_description)
            )
            exam_id = cursor.lastrowid
            return {
                "exam_id": exam_id,
                "exam_name": exam.exam_name,
                "exam_date": str(exam.exam_date),
                "exam_module_code": exam.exam_module_code,
                "exam_description": exam.exam_description
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/exams/{exam_id}", dependencies=[Depends(verify_api_key)])
def update_exam(exam_id: int, exam: ExamUpdate):
    """Update an exam"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        updates = []
        params = []
        if exam.exam_name is not None:
            updates.append("exam_name = ?")
            params.append(exam.exam_name)
        if exam.exam_date is not None:
            updates.append("exam_date = ?")
            params.append(str(exam.exam_date))
        if exam.exam_module_code is not None:
            updates.append("exam_module_code = ?")
            params.append(exam.exam_module_code)
        if exam.exam_description is not None:
            updates.append("exam_description = ?")
            params.append(exam.exam_description)

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        params.append(exam_id)
        query = f"UPDATE exams SET {', '.join(updates)} WHERE exam_id = ?"
        cursor.execute(query, params)

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Exam not found")

        return {"message": "Exam updated successfully"}

@app.delete("/api/exams/{exam_id}", dependencies=[Depends(verify_api_key)])
def delete_exam(exam_id: int):
    """Delete an exam"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM exams WHERE exam_id = ?", (exam_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Exam not found")

        return {"message": "Exam deleted successfully"}

# ==================== ATTENDANCE ENDPOINTS ====================

@app.get("/api/attendance", dependencies=[Depends(verify_api_key)])
def get_attendance():
    """Get all attendance records with student and exam details"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT attendance.*,
                students.student_full_name,
                students.student_admin_number,
                exams.exam_name,
                exams.exam_module_code,
                exams.exam_date
            FROM attendance
            LEFT JOIN students ON attendance.attendance_student_id = students.student_id
            LEFT JOIN exams ON attendance.attendance_exam_id = exams.exam_id
        """)
        records = [dict(row) for row in cursor.fetchall()]
        return {"attendance": records}

@app.get("/api/attendance/exam/{exam_id}", dependencies=[Depends(verify_api_key)])
def get_attendance_by_exam(exam_id: int):
    """Get attendance records for a specific exam"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT attendance.*,
                students.student_full_name,
                students.student_admin_number
            FROM attendance
            LEFT JOIN students ON attendance.attendance_student_id = students.student_id
            WHERE attendance.attendance_exam_id = ?
        """, (exam_id,))
        records = [dict(row) for row in cursor.fetchall()]
        return {"attendance": records}

@app.get("/api/attendance/student/{student_id}", dependencies=[Depends(verify_api_key)])
def get_attendance_by_student(student_id: int):
    """Get attendance records for a specific student"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT attendance.*,
                exams.exam_name,
                exams.exam_module_code,
                exams.exam_date
            FROM attendance
            LEFT JOIN exams ON attendance.attendance_exam_id = exams.exam_id
            WHERE attendance.attendance_student_id = ?
        """, (student_id,))
        records = [dict(row) for row in cursor.fetchall()]
        return {"attendance": records}

@app.post("/api/attendance", status_code=201, dependencies=[Depends(verify_api_key)])
def create_attendance(record: AttendanceCreate):
    """Create an attendance record"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO attendance (attendance_student_id, attendance_exam_id, attendance_status) VALUES (?, ?, ?)",
                (record.attendance_student_id, record.attendance_exam_id, record.attendance_status)
            )
            attendance_id = cursor.lastrowid
            return {
                "attendance_id": attendance_id,
                "attendance_student_id": record.attendance_student_id,
                "attendance_exam_id": record.attendance_exam_id,
                "attendance_status": record.attendance_status
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/attendance/{attendance_id}", dependencies=[Depends(verify_api_key)])
def update_attendance(attendance_id: int, record: AttendanceUpdate):
    """Update an attendance record's status"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE attendance SET attendance_status = ? WHERE attendance_id = ?",
            (record.attendance_status, attendance_id)
        )

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Attendance record not found")

        return {"message": "Attendance updated successfully"}

@app.delete("/api/attendance/{attendance_id}", dependencies=[Depends(verify_api_key)])
def delete_attendance(attendance_id: int):
    """Delete an attendance record"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance WHERE attendance_id = ?", (attendance_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Attendance record not found")

        return {"message": "Attendance record deleted successfully"}

# ==================== HEALTH CHECK ====================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat()
    }
    

@app.get("/")
def root():
    return {"message": "Attendance System API is running", "docs": "/docs"}

