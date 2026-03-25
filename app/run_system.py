#!/usr/bin/env python3
"""
run_system.py
=============
Closed-Loop Attendance Verification System — Full Pipeline Orchestrator
INF2009 Group 15 · Raspberry Pi 5

Uses:
  T1: rd03d.py          → mmWave LD2450 radar (direction detection)
  T2: dlib face_recognition + encodings.pickle (from facial_recognition_v2)
  T3: FastAPI REST API   → attendance state machine
  Out: rpi5_serial_sender → M5StickC feedback

Flow:
  ENTRY: mmWave → camera → face match → POST(status=0) → M5:BLUE "ENTRY"
  EXIT:  mmWave → camera → face match → PUT(status=1)  → M5:ORANGE "EXIT"
  NO MATCH: → M5:RED "NO_MATCH", nothing written to DB

Usage:
  Terminal 1:  uvicorn app.main:app --host 0.0.0.0 --port 8000
  Terminal 2:  python3 app/dashboard/app.py
  Terminal 3:  cd app && python3 run_system.py
"""

import os
import sys
import time
import signal
import requests
import numpy as np
import cv2
import pickle
import face_recognition

# ─── Ensure app/ is on the path ─────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from rd03d import RD03D
from rpi5_serial_sender import send_result


# ═════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════

API_BASE        = os.environ.get("API_BASE", "http://localhost:8000")
API_KEY         = os.environ.get("API_KEY", "changeme")
EXAM_ID         = int(os.environ.get("EXAM_ID", "1"))
DETECT_RANGE_MM = int(os.environ.get("DETECT_RANGE", "600"))
COOLDOWN_SEC    = float(os.environ.get("COOLDOWN", "3.0"))
POLL_INTERVAL   = float(os.environ.get("POLL_INTERVAL", "0.5"))

# Face recognition threshold (same as facial_recognition_v2.py)
STRANGER_THRESHOLD = float(os.environ.get("STRANGER_THRESHOLD", "0.35"))

# Path to encodings file (relative to app/ directory)
ENCODINGS_PATH  = os.environ.get("ENCODINGS_PATH", "encodings.pickle")

HEADERS = {"X-API-Key": API_KEY}

# Per-student cooldown tracker: { name: last_timestamp }
_recent_scans: dict[str, float] = {}


# ═════════════════════════════════════════════════════════════════
#  FACE RECOGNITION SETUP (from facial_recognition_v2.py)
# ═════════════════════════════════════════════════════════════════

print("[T2] Loading face encodings...")
if not os.path.exists(ENCODINGS_PATH):
    print(f"[T2] ERROR: {ENCODINGS_PATH} not found!")
    print(f"[T2] Run model_training_v2.py first to generate encodings.")
    sys.exit(1)

with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.loads(f.read())

known_face_encodings = data["encodings"]
known_face_names     = data["names"]

print(f"[T2] Loaded {len(known_face_encodings)} encodings for {len(set(known_face_names))} people")
print(f"[T2] Enrolled: {sorted(set(known_face_names))}")
print(f"[T2] Stranger threshold: {STRANGER_THRESHOLD}")

# ─── Camera init ─────────────────────────────────────────────────
print("[T2] Initialising camera...")
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("[T2] ERROR: Could not open webcam. Check USB connection.")
    sys.exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)  # camera warm-up
print("[T2] Camera ready")


def recognise_face() -> dict:
    """
    Capture a frame, run dlib face recognition, return result.

    Returns:
        {
            "match": bool,
            "name": str,           # person name or "STRANGER"
            "distance": float,     # best face distance
        }
    """
    ret, frame = cam.read()
    if not ret:
        return {"match": False, "name": "NO_FRAME", "distance": 1.0}

    # Downscale for speed (same as facial_recognition_v2.py cv_scaler=4)
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations, model='large')

    if len(face_encodings) == 0:
        return {"match": False, "name": "NO_FACE", "distance": 1.0}

    # Use the first detected face
    face_encoding = face_encodings[0]
    best_distance = 1.0
    name = "STRANGER"

    if len(known_face_encodings) > 0:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance < STRANGER_THRESHOLD:
            name = known_face_names[best_match_index]

    matched = name != "STRANGER"
    return {"match": matched, "name": name, "distance": best_distance}


# ═════════════════════════════════════════════════════════════════
#  API HELPERS
# ═════════════════════════════════════════════════════════════════

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=HEADERS, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[API ERROR] GET {endpoint}: {e}")
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", headers=HEADERS, json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[API ERROR] POST {endpoint}: {e}")
        return None


def api_put(endpoint: str, payload: dict):
    try:
        r = requests.put(f"{API_BASE}{endpoint}", headers=HEADERS, json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[API ERROR] PUT {endpoint}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════
#  STUDENT LOOKUP
# ═════════════════════════════════════════════════════════════════

# Cache: { name_lowercase: (student_id, admin_number) }
_student_cache: dict[str, tuple[int, str]] = {}


def build_student_cache():
    """Fetch all students from API and build a name→ID lookup."""
    data = api_get("/api/students")
    if data and "students" in data:
        for s in data["students"]:
            # Match by lowercase full name (dataset folder name → DB full name)
            key = s["student_full_name"].strip().lower()
            _student_cache[key] = (s["student_id"], s["student_admin_number"])
        print(f"[STARTUP] Cached {len(_student_cache)} students")


def lookup_student(name: str) -> tuple[int, str] | None:
    """
    Map a recognised face name to (student_id, admin_number).
    The 'name' comes from encodings.pickle which uses the dataset folder name.
    """
    key = name.strip().lower()

    if key in _student_cache:
        return _student_cache[key]

    # Rebuild cache in case new students were added
    build_student_cache()
    return _student_cache.get(key)


def find_open_attendance(student_id: int) -> dict | None:
    """Find an INCOMPLETE attendance record (status=0) for this student + exam."""
    data = api_get(f"/api/attendance/exam/{EXAM_ID}")
    if data and "attendance" in data:
        for rec in data["attendance"]:
            if rec["attendance_student_id"] == student_id and not rec["attendance_status"]:
                return rec
    return None


# ═════════════════════════════════════════════════════════════════
#  CORE — HANDLE ONE DETECTION EVENT
# ═════════════════════════════════════════════════════════════════

def handle_detection(direction: str):
    """
    Called when mmWave detects a person within range.

    ENTRY:
      1. Face match → M5StickC: GREEN "MATCHED"
      2. POST /api/attendance (status=0) → M5StickC: BLUE "ENTRY"
         Dashboard: INCOMPLETE

    EXIT:
      3. Face match → M5StickC: GREEN "MATCHED"
      4. PUT /api/attendance/{id} (status=1) → M5StickC: ORANGE "EXIT"
         Dashboard: PRESENT

    No match:
      6. M5StickC: RED "NO_MATCH" → nothing written to DB
    """

    print(f"\n{'='*60}")
    print(f"[EVENT] Direction: {direction} | {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # ── Run face recognition ─────────────────────────────────
    result = recognise_face()

    if result["name"] == "NO_FRAME":
        print("[EVENT] Camera frame skip")
        return

    if result["name"] == "NO_FACE":
        print("[EVENT] No face detected in frame")
        return

    if not result["match"]:
        # ── Step 6: Face not recognised ──────────────────────
        print(f"[EVENT] STRANGER detected (distance: {result['distance']:.3f})")
        send_result("NO_MATCH", "")
        return

    # ── Face matched ─────────────────────────────────────────
    name = result["name"]
    print(f"[EVENT] MATCHED: {name} (distance: {result['distance']:.3f})")

    # Look up student in DB
    student_info = lookup_student(name)
    if student_info is None:
        print(f"[EVENT] WARNING: '{name}' matched by face but not found in DB — skipping")
        send_result("MATCHED", name)
        return

    student_id, admin_no = student_info
    send_result("MATCHED", admin_no)

    # ── Cooldown check ───────────────────────────────────────
    now = time.time()
    cooldown_key = f"{admin_no}_{direction}"
    if cooldown_key in _recent_scans:
        elapsed = now - _recent_scans[cooldown_key]
        if elapsed < COOLDOWN_SEC:
            print(f"[EVENT] Cooldown active ({elapsed:.1f}s < {COOLDOWN_SEC}s) — skipping")
            return
    _recent_scans[cooldown_key] = now

    # ── ENTRY (Steps 1–2) ────────────────────────────────────
    if direction == "ENTRY":
        existing = find_open_attendance(student_id)
        if existing:
            print(f"[EVENT] {admin_no} already has open ENTRY (id={existing['attendance_id']}) — skipping")
            send_result("ENTRY", admin_no)
            return

        payload = {
            "attendance_student_id": student_id,
            "attendance_exam_id": EXAM_ID,
            "attendance_status": False,
        }
        resp = api_post("/api/attendance", payload)
        if resp:
            print(f"[EVENT] ENTRY recorded → id={resp.get('attendance_id')}")
            print(f"[EVENT] Dashboard: INCOMPLETE (awaiting exit)")
            send_result("ENTRY", admin_no)
        else:
            print(f"[EVENT] ERROR: Failed to create attendance record")

    # ── EXIT (Steps 3–4) ─────────────────────────────────────
    elif direction == "EXIT":
        open_record = find_open_attendance(student_id)
        if open_record is None:
            print(f"[EVENT] {admin_no} has no open ENTRY — cannot exit without entry")
            send_result("INCOMPLETE", admin_no)
            return

        attendance_id = open_record["attendance_id"]
        resp = api_put(f"/api/attendance/{attendance_id}", {"attendance_status": True})
        if resp:
            print(f"[EVENT] EXIT recorded → id={attendance_id} → PRESENT ✓")
            send_result("EXIT", admin_no)
        else:
            print(f"[EVENT] ERROR: Failed to update attendance record")


# ═════════════════════════════════════════════════════════════════
#  T1 — mmWave SENSOR LOOP
# ═════════════════════════════════════════════════════════════════

def run_sensor_loop():
    """Poll LD2450 radar. On detection → determine direction → handle_detection()."""
    print("[T1] Initialising LD2450 mmWave radar...")
    radar = RD03D(uart_port='/dev/ttyAMA10')
    radar.set_multi_mode(False)
    print(f"[T1] Radar ready — range: {DETECT_RANGE_MM}mm")
    print()

    while True:
        #print("Starting while loop...")
        
        
        #if radar.update():
            #print("Received update from mmvwave")
            #target = radar.get_target(1)
            #if target and target.distance < DETECT_RANGE_MM:
               # direction = "ENTRY" if target.angle < -1 else "EXIT"
                #print(f"[T1] Human detected | dist={target.distance:.0f}mm | "
                               #       f"angle={target.angle:.1f}° | → {direction}")
        direction = "EXIT"
        handle_detection(direction)

        #except KeyboardInterrupt:
           # raise
        #except Exception as e:
         #   print(f"[T1] Error: {e}")

        time.sleep(POLL_INTERVAL)


# ═════════════════════════════════════════════════════════════════
#  STARTUP
# ═════════════════════════════════════════════════════════════════

def check_api_health() -> bool:
    print(f"[STARTUP] Checking API at {API_BASE}...")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"[STARTUP] API is UP — {r.json()}")
        return True
    except Exception as e:
        print(f"[STARTUP] WARNING: API not reachable — {e}")
        print(f"[STARTUP] Run: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False


def check_exam_exists() -> bool:
    data = api_get(f"/api/exams/{EXAM_ID}")
    if data and "exam" in data:
        exam = data["exam"]
        print(f"[STARTUP] Active exam: [{exam['exam_module_code']}] "
              f"{exam['exam_name']} ({exam['exam_date']})")
        return True
    print(f"[STARTUP] WARNING: Exam ID {EXAM_ID} not found — run seed_test_data.py first")
    return False


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Closed-Loop Attendance Verification System            ║")
    print("║   INF2009 Group 15 · Raspberry Pi 5                     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                          ║")
    print("║   T1: mmWave LD2450        → detect + direction          ║")
    print("║   T2: dlib face_recognition → match vs encodings.pickle  ║")
    print("║   T3: FastAPI + SQLite      → attendance state machine   ║")
    print("║   Out: M5StickC serial      → LED + buzzer feedback      ║")
    print("║   Web: Flask dashboard      → live occupancy @ :5000     ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  ENTRY: mmWave → camera → face match → POST(status=0) → M5:BLUE")
    print("  EXIT:  mmWave → camera → face match → PUT(status=1)  → M5:ORANGE")
    print("  NO MATCH: → M5:RED, nothing written to DB")
    print()


def graceful_shutdown(signum, frame):
    print("\n[SHUTDOWN] Stopping...")
    cam.release()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    print_banner()

    api_ok = check_api_health()
    if api_ok:
        check_exam_exists()
        build_student_cache()

    print()
    print(f"[CONFIG] API_BASE      = {API_BASE}")
    print(f"[CONFIG] EXAM_ID       = {EXAM_ID}")
    print(f"[CONFIG] DETECT_RANGE  = {DETECT_RANGE_MM}mm")
    print(f"[CONFIG] COOLDOWN      = {COOLDOWN_SEC}s")
    print(f"[CONFIG] THRESHOLD     = {STRANGER_THRESHOLD}")
    print()
    print("[STARTUP] Starting T1 sensor loop... (Ctrl+C to stop)")
    print()

    run_sensor_loop()


if __name__ == "__main__":
    main()
