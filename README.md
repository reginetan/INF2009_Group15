# 🎓 Attendance System API — INF2009 Group 15

A FastAPI-based attendance management system with face recognition support, built for edge computing.

---

## 📋 Table of Contents
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [Mac](#mac)
  - [Windows](#windows)
- [Running the Server](#running-the-server)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Exporting Data](#exporting-data)

---

## Requirements
- Python 3.11
- pip
- Git

---

## Setup Instructions

### Mac

**1. Clone the repository**
```bash
git clone https://github.com/reginetan/INF2009_Group15.git
cd INF2009_Group15
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create a `.env` file in the project root**
```bash
echo "API_KEY=your-secret-key-here" > .env
```

---

### Windows

**1. Clone the repository**
```cmd
git clone https://github.com/reginetan/INF2009_Group15.git
cd INF2009_Group15
```

**2. Create a virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```cmd
pip install -r requirements.txt
```

**4. Create a `.env` file in the project root**
```cmd
echo API_KEY=your-secret-key-here > .env
```

---

## Running the Server

### Mac
```bash
uvicorn app.main:app --reload
```

### Windows
```cmd
uvicorn app.main:app --reload
```

The server will start at **http://localhost:8000**

---

## API Documentation

Once the server is running, open your browser and go to:

| Page | URL |
|---|---|
| **Swagger UI** (interactive) | http://localhost:8000/docs |
| **ReDoc** (documentation) | http://localhost:8000/redoc |
| **Health Check** | http://localhost:8000/health |

### Authentication
All endpoints require an API key. In Swagger UI:
1. Click the 🔒 **Authorize** button
2. Enter your API key from `.env`

For `curl` requests:
```bash
curl -H "X-API-Key: your-secret-key-here" http://localhost:8000/api/students
```

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/students` | Get all students |
| POST | `/api/students` | Add a student |
| PUT | `/api/students/{id}` | Update a student |
| DELETE | `/api/students/{id}` | Delete a student |
| GET | `/api/embeddings` | Get all face embeddings |
| POST | `/api/embeddings` | Upload a student face image |
| DELETE | `/api/embeddings/{id}` | Delete a face embedding |
| GET | `/api/exams` | Get all exams |
| POST | `/api/exams` | Add an exam |
| PUT | `/api/exams/{id}` | Update an exam |
| DELETE | `/api/exams/{id}` | Delete an exam |
| GET | `/api/attendance` | Get all attendance records |
| POST | `/api/attendance` | Mark attendance |
| PUT | `/api/attendance/{id}` | Update attendance |
| DELETE | `/api/attendance/{id}` | Delete attendance record |

---

## Exporting Data

To export all database tables as CSV files:
```bash
python export_data.py
```

This creates CSV files in the `exports/` folder:
```
exports/
├── students.csv
├── face_embedding.csv
├── exams.csv
└── attendance.csv
```

Or use the API export endpoint in your browser:
```
http://localhost:8000/api/export/students
http://localhost:8000/api/export/exams
http://localhost:8000/api/export/attendance
```

---

## Docker (Optional)

**Build the image**
```bash
docker build -t attendance-api .
```

**Run the container**
```bash
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/app/images:/app/app/images \
  --name attendance-api \
  attendance-api
```

Then open **http://localhost:8080/docs**

---

## ⚠️ Notes
- The `data/` folder and database are created automatically on first run
- Never commit your `.env` file — it's already in `.gitignore`
- Deleting a student will automatically delete their face embeddings, attendance records, and uploaded images