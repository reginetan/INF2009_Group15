"""
face_recogniser.py — ArcFace ResNet100 ONNX + YuNet alignment + USB Webcam
Group 15 — INF2009 Edge Computing
"""
import argparse
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import os

os.environ["ORT_LOGGING_LEVEL"] = "3"
ort.set_default_logger_severity(3)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).parent
MODEL_DIR    = BASE_DIR / "models"
ARCFACE_PATH = MODEL_DIR / "arcfaceresnet100-8.onnx"
YUNET_PATH   = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
DB_PATH      = BASE_DIR / "enrolled.pkl"

# Matching — with alignment, genuine pairs ~0.40-0.55, impostors ~0.10-0.29
COSINE_THRESHOLD = 0.35
MARGIN_MIN       = 0.05   # top score must beat 2nd place by at least this

# Camera / detection
CAM_SIZE    = (640, 480)
DETECT_SIZE = (320, 320)
MODEL_SIZE  = (112, 112)

# Vote buffer
WINDOW_SIZE    = 5
MIN_MATCH_RATE = 0.60
MIN_AVG_SCORE  = 0.35

# Quality gate
BLUR_MIN       = 50
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 230

# Enrollment
ENROLL_FRAMES   = 15
ENROLL_MIN_GOOD = 5

# ArcFace reference landmarks for 112x112 output
ARCFACE_REF_PTS = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
])


# ---------------------------------------------------------------------------
# Face alignment
# ---------------------------------------------------------------------------

def align_face(img: np.ndarray, landmarks: list) -> np.ndarray:
    src = np.float32(landmarks)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_REF_PTS, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, MODEL_SIZE, flags=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Camera (USB webcam via OpenCV)
# ---------------------------------------------------------------------------

def open_camera():
    """Open USB webcam using OpenCV VideoCapture."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        # Try index 1 if 0 fails
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        raise RuntimeError("Could not open any webcam. Check USB connection.")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_SIZE[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_SIZE[1])

    log.info("Warming up camera (30 frames)...")
    for _ in range(30):
        cam.read()
    log.info("Camera ready.")
    return cam


def read_frame(cam) -> np.ndarray:
    ret, bgr = cam.read()
    if not ret or bgr is None:
        raise RuntimeError("Failed to read frame from webcam.")
    return cv2.resize(bgr, DETECT_SIZE)


def release_camera(cam):
    """Release the webcam."""
    if cam is not None:
        cam.release()


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def is_good_frame(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0:
        return False
    gray       = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    return blur >= BLUR_MIN and BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX


# ---------------------------------------------------------------------------
# ArcFace ONNX
# ---------------------------------------------------------------------------

class ArcFaceModel:
    def __init__(self):
        if not ARCFACE_PATH.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {ARCFACE_PATH}\n"
                "Download: wget https://github.com/onnx/models/raw/main/validated/"
                "vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx "
                "-O models/arcfaceresnet100-8.onnx"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess       = ort.InferenceSession(
            str(ARCFACE_PATH), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name
        log.info("ArcFace ResNet100 ONNX loaded.")

    def embed(self, aligned_112: np.ndarray) -> np.ndarray:
        if aligned_112 is None or aligned_112.shape[:2] != MODEL_SIZE:
            return None
        try:
            blob = np.transpose(aligned_112.astype(np.float32), (2, 0, 1))[np.newaxis]
            raw  = self._sess.run(None, {self._input_name: blob})[0][0]
            norm = np.linalg.norm(raw)
            return raw / norm if norm > 0 else raw
        except Exception as e:
            log.debug("Embedding error: %s", e)
            return None


# ---------------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------------

class Detector:
    def __init__(self):
        if YUNET_PATH.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH), "", DETECT_SIZE,
                    score_threshold=0.5, nms_threshold=0.3, top_k=5,
                )
                self._mode = "yunet"
                log.info("YuNet detector loaded.")
                return
            except Exception as e:
                log.warning("YuNet failed: %s", e)
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._mode = "haar"
        log.info("Haar Cascade loaded.")

    def detect_and_align(self, frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]
        if self._mode == "yunet":
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return None, None, []
            best    = max(faces, key=lambda f: f[14])
            x, y, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
            lm      = [(best[4 + i*2], best[4 + i*2 + 1]) for i in range(5)]
            aligned = align_face(frame, lm)
            return aligned, (x, y, bw, bh), faces
        else:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            if len(faces) == 0:
                return None, None, []
            x, y, bw, bh = faces[0]
            crop    = frame[y:y+bh, x:x+bw]
            aligned = cv2.resize(crop, MODEL_SIZE)
            return aligned, (x, y, bw, bh), faces

    def detect_and_align_hires(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        for detect_w in [w, 1280, 640, 320]:
            if detect_w > w:
                continue
            scale    = detect_w / w
            detect_h = int(h * scale)
            resized  = cv2.resize(img, (detect_w, detect_h))
            aligned, bbox, faces = self.detect_and_align(resized)
            if aligned is not None:
                log.info("Enrolled at detection size %dx%d", detect_w, detect_h)
                return aligned
        return None


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    def __init__(self):
        self._store: dict = {}
        if DB_PATH.exists():
            with open(DB_PATH, "rb") as f:
                self._store = pickle.load(f)
            log.info("DB loaded: %d student(s)", len(self._store))
        else:
            log.info("Empty DB.")

    def _save(self):
        with open(DB_PATH, "wb") as f:
            pickle.dump(self._store, f)

    def add(self, student_id: str, name: str, embedding: np.ndarray):
        if student_id in self._store:
            self._store[student_id]["embeddings"].append(embedding.copy())
        else:
            self._store[student_id] = {
                "name":        name,
                "embeddings":  [embedding.copy()],
                "enrolled_at": datetime.now(timezone.utc).isoformat(),
            }
        self._save()
        n = len(self._store[student_id]["embeddings"])
        log.info("Stored embedding for %s (%s) — total=%d", name, student_id, n)

    def remove(self, student_id: str) -> bool:
        if student_id in self._store:
            del self._store[student_id]
            self._save()
            return True
        return False

    def best_match(self, query: np.ndarray) -> tuple:
        if not self._store:
            return None, None, 0.0, 0.0

        scores = {}
        for sid, rec in self._store.items():
            centroid = np.mean(rec["embeddings"], axis=0)
            norm     = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            scores[sid] = (float(np.dot(query, centroid)), rec["name"])

        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        log.info("Scores — %s", " | ".join(f"{v[1]}={v[0]:.4f}" for _, v in sorted_scores))

        best_id, (best_score, best_name) = sorted_scores[0]
        margin = best_score - sorted_scores[1][1][0] if len(sorted_scores) > 1 else 1.0
        return best_id, best_name, best_score, margin

    def rebuild(self):
        for sid, rec in self._store.items():
            embs     = rec["embeddings"]
            centroid = np.mean(embs, axis=0)
            norm     = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            rec["embeddings"] = [centroid]
            print(f"  {rec['name']} ({sid}): {len(embs)} -> 1 centroid")
        self._save()

    def list_all(self):
        return [
            {"student_id": sid, "name": v["name"],
             "enrolled_at": v["enrolled_at"], "num_embeddings": len(v["embeddings"])}
            for sid, v in self._store.items()
        ]

    def __len__(self):
        return len(self._store)


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------

class Recogniser:
    def __init__(self):
        self.db       = Database()
        self.detector = Detector()
        self.arcface  = ArcFaceModel()

    def identify(self, aligned_112: np.ndarray) -> dict:
        emb = self.arcface.embed(aligned_112)
        if emb is None:
            return {"match": False, "student_id": None, "name": None,
                    "score": 0.0, "margin": 0.0}
        sid, name, score, margin = self.db.best_match(emb)
        matched = score >= COSINE_THRESHOLD and margin >= MARGIN_MIN
        return {"match": matched, "student_id": sid, "name": name,
                "score": round(score, 4), "margin": round(margin, 4)}

    def enroll_from_image(self, student_id: str, name: str, img: np.ndarray) -> bool:
        aligned = self.detector.detect_and_align_hires(img)
        if aligned is None:
            log.error("No face detected in image.")
            return False
        emb = self.arcface.embed(aligned)
        if emb is None:
            log.error("Could not extract embedding.")
            return False
        self.db.add(student_id, name, emb)
        print(f"Enrollment SUCCESS — {name} ({student_id})")
        return True

    def enroll_from_camera(self, student_id: str, name: str) -> bool:
        cam  = open_camera()
        good = []
        attempts = 0
        print(f"Enrolling {name} — look at the camera...", flush=True)
        try:
            while len(good) < ENROLL_FRAMES and attempts < ENROLL_FRAMES * 4:
                attempts += 1
                frame = read_frame(cam)
                aligned, _, _ = self.detector.detect_and_align(frame)
                if aligned is None:
                    print(f"  No face (attempt {attempts})", flush=True)
                    continue
                if not is_good_frame(aligned):
                    continue
                emb = self.arcface.embed(aligned)
                if emb is not None:
                    good.append(emb)
                    print(f"  Captured {len(good)}/{ENROLL_FRAMES}", flush=True)
                time.sleep(0.15)
        finally:
            release_camera(cam)

        if len(good) < ENROLL_MIN_GOOD:
            log.error("Enrollment failed — only %d good frames.", len(good))
            return False
        avg  = np.mean(good, axis=0)
        avg /= np.linalg.norm(avg)
        self.db.add(student_id, name, avg)
        print(f"Enrollment SUCCESS — {name} from {len(good)} frames.")
        return True


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame, bbox, label, color):
    display = frame.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
    cv2.rectangle(display, (0, 0), (display.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(display, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return display


# ---------------------------------------------------------------------------
# Live recognition loop
# ---------------------------------------------------------------------------

def run_recognition(recogniser: Recogniser):
    cam         = open_camera()
    vote_buffer = []
    last_print  = ""

    print("Live recognition started. Press Q to quit.\n", flush=True)
    try:
        while True:
            frame = read_frame(cam)
            aligned, bbox, _ = recogniser.detector.detect_and_align(frame)

            if aligned is None:
                vote_buffer.clear()
                display = draw_overlay(frame, None, "No face detected", (0, 255, 255))
                msg = "No face detected"
                if msg != last_print:
                    print(msg, flush=True); last_print = msg
            else:
                result = recogniser.identify(aligned)
                vote_buffer.append(result)
                if len(vote_buffer) > WINDOW_SIZE:
                    vote_buffer.pop(0)

                if len(vote_buffer) < WINDOW_SIZE:
                    label   = f"Scanning... ({len(vote_buffer)}/{WINDOW_SIZE})"
                    display = draw_overlay(frame, bbox, label, (255, 255, 0))
                    if label != last_print:
                        print(label, flush=True); last_print = label
                else:
                    matches    = [v for v in vote_buffer if v["match"]]
                    match_rate = len(matches) / WINDOW_SIZE
                    avg_score  = sum(v["score"] for v in vote_buffer) / WINDOW_SIZE

                    if match_rate >= MIN_MATCH_RATE and avg_score >= MIN_AVG_SCORE:
                        ids = [v["student_id"] for v in matches]
                        if len(set(ids)) == 1:
                            sid   = matches[0]["student_id"]
                            name  = matches[0]["name"]
                            label = f"MATCH: {name} ({avg_score:.3f})"
                            color = (0, 255, 0)
                            msg   = (f"CONFIRMED MATCH | {sid} | {name} | "
                                     f"rate={match_rate:.0%} | avg={avg_score:.4f}")
                            print(msg, flush=True); last_print = msg
                            vote_buffer.clear()
                        else:
                            label = "NO MATCH | Conflict"
                            color = (0, 0, 255)
                            msg   = f"NO MATCH | Conflict | avg={avg_score:.4f}"
                            if msg != last_print:
                                print(msg, flush=True); last_print = msg
                    else:
                        label = f"NO MATCH ({avg_score:.3f})"
                        color = (0, 0, 255)
                        msg   = f"NO MATCH | rate={match_rate:.0%} | avg={avg_score:.4f}"
                        if msg != last_print:
                            print(msg, flush=True); last_print = msg

                    display = draw_overlay(frame, bbox, label, color)

            cv2.imshow("Face Recognition — ArcFace", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        release_camera(cam)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Face Recogniser — ArcFace ResNet100 ONNX + YuNet alignment"
    )
    parser.add_argument("--enroll-cam", action="store_true",
                        help="Enroll a student using the USB webcam")
    parser.add_argument("--enroll", action="store_true",
                        help="Enroll a student from an image file")
    parser.add_argument("--recognize", action="store_true",
                        help="Run live face recognition using USB webcam")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delete", type=str, metavar="STUDENT_ID")
    parser.add_argument("--rebuild-db", action="store_true")
    parser.add_argument("--image", type=str)
    parser.add_argument("--name",  type=str)
    parser.add_argument("--id",    type=str)
    args = parser.parse_args()

    if args.list:
        db = Database()
        students = db.list_all()
        if not students:
            print("No students enrolled."); return
        print(f"\n{'ID':<16} {'Name':<20} {'Enrolled At':<35} {'Embeddings'}")
        print("-" * 80)
        for s in students:
            print(f"{s['student_id']:<16} {s['name']:<20} "
                  f"{s['enrolled_at']:<35} {s['num_embeddings']}")
        return

    if args.delete:
        db = Database()
        print("Deleted." if db.remove(args.delete) else "ID not found.")
        return

    if args.rebuild_db:
        db = Database(); db.rebuild(); print("Done."); return

    recogniser = Recogniser()

    if args.enroll_cam:
        if not args.name or not args.id:
            parser.error("--enroll-cam requires --name and --id")
        recogniser.enroll_from_camera(args.id, args.name)
        return

    if args.enroll:
        if not args.image or not args.name or not args.id:
            parser.error("--enroll requires --image, --name, and --id")
        img = cv2.imread(args.image)
        if img is None:
            log.error("Cannot read image: %s", args.image); return
        recogniser.enroll_from_image(args.id, args.name, img)
        return

    if args.recognize:
        run_recognition(recogniser)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
