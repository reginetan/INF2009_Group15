import argparse
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    import onnxruntime as ort
    import os
    os.environ["ORT_LOGGING_LEVEL"] = "3"
    ort.set_default_logger_severity(3)
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not installed - falling back to OpenCV DNN backend.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR           = Path(__file__).parent
MODEL_DIR          = BASE_DIR / "models"
IMAGE_DIR          = BASE_DIR / "images"
YUNET_PATH         = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
MOBILEFACENET_PATH = MODEL_DIR / "mobilefacenet_int8.onnx"
DB_PATH            = BASE_DIR / "enrolled.pkl"

COSINE_THRESHOLD   = 0.85
MODEL_INPUT_SIZE   = (112, 112)
DETECT_INPUT_SIZE  = (320, 240)

WINDOW_SIZE        = 10
MIN_MATCH_RATE     = 0.80
MIN_AVG_SCORE      = 0.85


class EnrolledDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._store: dict = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                self._store = pickle.load(f)
            log.info("DB loaded: %d student(s)", len(self._store))
        else:
            log.info("No DB found at %s -- starting fresh.", self.db_path)

    def _save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self._store, f)

    def add(self, student_id: str, name: str, embedding: np.ndarray):
        if student_id in self._store:
            self._store[student_id]["embeddings"].append(embedding.copy())
            log.info("Added embedding for: %s -- total=%d", student_id, len(self._store[student_id]["embeddings"]))
        else:
            self._store[student_id] = {
                "name":        name,
                "embeddings":  [embedding.copy()],
                "enrolled_at": datetime.now(timezone.utc).isoformat(),
            }
            log.info("Enrolled: %s -- %s", student_id, name)
        self._save()

    def remove(self, student_id: str) -> bool:
        if student_id in self._store:
            del self._store[student_id]
            self._save()
            return True
        return False

    def list_all(self):
        return [
            {"student_id": sid, "name": v["name"],
             "enrolled_at": v["enrolled_at"],
             "num_embeddings": len(v["embeddings"])}
            for sid, v in self._store.items()
        ]

    def find_best_match(self, query: np.ndarray, sim_fn: Callable) -> tuple:
        if not self._store:
            return None, None, 0.0
        best_id, best_name, best_score = None, None, -1.0
        for sid, rec in self._store.items():
            scores = [sim_fn(query, emb) for emb in rec["embeddings"]]
            score  = max(scores)
            if score > best_score:
                best_score, best_id, best_name = score, sid, rec["name"]
        return best_id, best_name, best_score

    def __len__(self):
        return len(self._store)


class YuNetDetector:
    def __init__(self):
        self._detector = None
        self._fallback = False
        if YUNET_PATH.exists():
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH), "", DETECT_INPUT_SIZE,
                    score_threshold=0.6, nms_threshold=0.3, top_k=5,
                )
                log.info("YuNet loaded.")
            except Exception as e:
                log.warning("YuNet failed (%s) -- using Haar fallback.", e)
                self._init_haar()
        else:
            log.warning("YuNet model not found at %s -- using Haar fallback.", YUNET_PATH)
            self._init_haar()

    def _init_haar(self):
        self._fallback = True
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        log.info("Haar Cascade loaded.")

    def detect(self, frame: np.ndarray) -> list:
        if self._detector is None:
            return []
        if self._fallback:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            return [np.array([x, y, w, h], dtype=np.float32) for (x, y, w, h) in faces]
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, DETECT_INPUT_SIZE)
        self._detector.setInputSize(DETECT_INPUT_SIZE)
        _, faces = self._detector.detect(resized)
        if faces is None:
            return []
        sx, sy = w / DETECT_INPUT_SIZE[0], h / DETECT_INPUT_SIZE[1]
        out = []
        for f in faces:
            f = f.copy()
            f[0] *= sx; f[1] *= sy; f[2] *= sx; f[3] *= sy
            out.append(f)
        return out

    def crop_face(self, frame: np.ndarray, det) -> np.ndarray:
        x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        mx, my = int(w * 0.10), int(h * 0.10)
        fh, fw = frame.shape[:2]
        return frame[max(0, y-my):min(fh, y+h+my), max(0, x-mx):min(fw, x+w+mx)]


class FaceRecogniser:
    def __init__(self, threshold: float = COSINE_THRESHOLD):
        self.threshold = threshold
        self.db        = EnrolledDatabase()
        self.detector  = YuNetDetector()
        self._session  = self._load_model()

    def _load_model(self):
        if not MOBILEFACENET_PATH.exists():
            log.warning("MobileFaceNet model not found at %s. Run: bash download_models.sh", MOBILEFACENET_PATH)
            return None
        if ONNX_AVAILABLE:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                str(MOBILEFACENET_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            log.info("MobileFaceNet INT8 loaded via ONNX Runtime.")
            return sess
        else:
            net = cv2.dnn.readNetFromONNX(str(MOBILEFACENET_PATH))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            log.info("MobileFaceNet loaded via OpenCV DNN (fallback).")
            return net

    @staticmethod
    def _preprocess(face_img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_img, MODEL_INPUT_SIZE)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb     = (rgb - 127.5) / 128.0
        nchw    = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(nchw, axis=0)

    def get_embedding(self, face_img: np.ndarray):
        if self._session is None:
            return None
        blob = self._preprocess(face_img)
        if ONNX_AVAILABLE:
            input_name = self._session.get_inputs()[0].name
            raw = self._session.run(None, {input_name: blob})[0][0]
        else:
            self._session.setInput(blob)
            raw = self._session.forward()[0]
        norm = np.linalg.norm(raw)
        return (raw / norm) if norm > 0 else raw

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def identify(self, face_img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        embedding = self.get_embedding(face_img)
        if embedding is None:
            return _no_match(0.0)
        best_id, best_name, best_score = self.db.find_best_match(embedding, self.cosine_similarity)
        latency_ms = (time.perf_counter() - t0) * 1000
        if best_score >= self.threshold:
            log.info("MATCH -> %s (%s) | score=%.3f | %.1f ms", best_id, best_name, best_score, latency_ms)
            return {
                "match": True, "student_id": best_id, "name": best_name,
                "confidence": round(best_score, 4), "latency_ms": round(latency_ms, 2),
            }
        else:
            log.info("NO MATCH | best=%.3f < %.2f | %.1f ms", best_score, self.threshold, latency_ms)
            return _no_match(best_score, latency_ms)

    def enroll(self, student_id: str, name: str, image: np.ndarray) -> bool:
        faces = self.detector.detect(image)
        if not faces:
            log.error("Enrollment failed: no face detected.")
            return False
        crop = self.detector.crop_face(image, faces[0])
        embedding = self.get_embedding(crop)
        if embedding is None:
            return False
        self.db.add(student_id, name, embedding)
        return True

    def process_frame(self, frame: np.ndarray) -> tuple:
        faces   = self.detector.detect(frame)
        results = []
        for face in faces:
            crop   = self.detector.crop_face(frame, face)
            result = self.identify(crop)
            result["bbox"] = face[:4].astype(int)
            results.append(result)
        return results, faces


def _no_match(score: float, latency_ms: float = 0.0) -> dict:
    return {
        "match": False, "student_id": None, "name": None,
        "confidence": round(score, 4), "latency_ms": round(latency_ms, 2),
    }


def draw_results(frame: np.ndarray, results: list, confirmed: bool = False) -> np.ndarray:
    for r in results:
        x, y, w, h = r["bbox"]
        if confirmed and r["match"]:
            color = (0, 255, 0)
            label = f"CONFIRMED: {r['name']} ({r['confidence']:.2f})"
        elif r["match"]:
            color = (0, 165, 255)
            label = f"{r['name']} ({r['confidence']:.2f})"
        else:
            color = (0, 0, 220)
            label = f"NO MATCH ({r['confidence']:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Face Recognition -- MobileFaceNet INT8")
    parser.add_argument("--enroll",     action="store_true")
    parser.add_argument("--webcam",     action="store_true")
    parser.add_argument("--image",      type=str)
    parser.add_argument("--name",       type=str)
    parser.add_argument("--id",         type=str)
    parser.add_argument("--threshold",  type=float, default=COSINE_THRESHOLD)
    parser.add_argument("--camera-idx", type=int,   default=0)
    args = parser.parse_args()

    recogniser = FaceRecogniser(threshold=args.threshold)

    if args.enroll:
        if not args.image or not args.name or not args.id:
            parser.error("--enroll requires --image, --name, and --id")
        img = cv2.imread(args.image)
        if img is None:
            log.error("Cannot read image: %s", args.image)
            return
        ok = recogniser.enroll(args.id, args.name, img)
        print("Enrollment", "SUCCESS" if ok else "FAILED")
        return

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            log.error("Cannot read image: %s", args.image)
            return
        results, _ = recogniser.process_frame(img)
        if not results:
            print("No faces detected.")
        for r in results:
            status = "MATCH" if r["match"] else "NO MATCH"
            print(f"{status} | ID: {r['student_id']} | Name: {r['name']} | "
                  f"Confidence: {r['confidence']:.4f} | Latency: {r['latency_ms']:.1f} ms")
        cv2.imshow("Recognition Result", draw_results(img, results))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if args.webcam:
        cap = cv2.VideoCapture(args.camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS,          30)
        if not cap.isOpened():
            log.error("Cannot open camera index %d", args.camera_idx)
            return
        log.info("Webcam running. Press q to quit.")

        vote_buffer = []
        confirmed   = False

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            results, _ = recogniser.process_frame(frame)

            if not results:
                vote_buffer.clear()
                confirmed = False
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Face Recognition -- T2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            r = results[0]
            vote_buffer.append(r)
            if len(vote_buffer) > WINDOW_SIZE:
                vote_buffer.pop(0)

            if len(vote_buffer) < WINDOW_SIZE:
                progress = f"Checking... ({len(vote_buffer)}/{WINDOW_SIZE})"
                cv2.putText(frame, progress, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Face Recognition -- T2", draw_results(frame, results))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            matches    = [v for v in vote_buffer if v["match"] and v["student_id"] is not None]
            match_rate = len(matches) / WINDOW_SIZE
            avg_score  = sum(v["confidence"] for v in vote_buffer) / WINDOW_SIZE

            if match_rate >= MIN_MATCH_RATE and avg_score >= MIN_AVG_SCORE:
                ids = [v["student_id"] for v in matches]
                if len(set(ids)) == 1:
                    sid  = matches[0]["student_id"]
                    name = matches[0]["name"]
                    confirmed = True
                    log.info("CONFIRMED MATCH -> %s | rate=%.0f%% | avg=%.3f", sid, match_rate*100, avg_score)
                    print(f"CONFIRMED MATCH | {sid} | {name} | rate={match_rate:.0%} | avg={avg_score:.4f}")
                    vote_buffer.clear()
                else:
                    confirmed = False
                    log.info("CONFLICT -- multiple IDs, NO MATCH")
            else:
                confirmed = False

            cv2.imshow("Face Recognition -- T2", draw_results(frame, results, confirmed))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    parser.print_help()


if __name__ == "__main__":
    main()