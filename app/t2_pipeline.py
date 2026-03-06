import cv2
import time
from frame_capture   import init_camera, capture_frame
from face_detector   import init_yunet, detect_faces, extract_face_crop
from face_recogniser import FaceRecogniser

cap        = init_camera()
detector   = init_yunet(input_w=320, input_h=240)
recogniser = FaceRecogniser()

def run_t2_pipeline(direction: str):
    t_start = time.time()

    raw_frame, frame_320 = capture_frame(cap)
    if frame_320 is None:
        print("[T2] Frame skip -- logged")
        return None

    faces = detect_faces(detector, frame_320)
    if len(faces) == 0:
        print("[T2] No face detected")
        return None

    best_face = max(faces, key=lambda f: f["confidence"])

    face_crop = extract_face_crop(frame_320, best_face["bbox"])
    if face_crop is None:
        print("[T2] Crop extraction failed")
        return None

    t2_input = {
        "face_crop":  face_crop,
        "direction":  direction,
        "bbox":       best_face["bbox"],
        "landmarks":  best_face["landmarks"],
        "confidence": best_face["confidence"],
        "raw_frame":  raw_frame,
    }
    result = recogniser.identify_from_t2(t2_input)

    result["raw_frame"]  = raw_frame
    result["frame_320"]  = frame_320   # small frame for display

    elapsed_ms = (time.time() - t_start) * 1000
    status = "MATCH" if result["match"] else "NO MATCH"
    print(f"[T2] {status} | {result['student_id']} | conf={result['confidence']:.3f} | {elapsed_ms:.1f} ms total")

    return result
