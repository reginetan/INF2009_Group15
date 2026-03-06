import cv2
import numpy as np

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

SCORE_THRESHOLD = 0.7
NMS_THRESHOLD   = 0.3
TOP_K           = 5000

def init_yunet(input_w=320, input_h=240):
    detector = cv2.FaceDetectorYN.create(
        model           = MODEL_PATH,
        config          = "",
        input_size      = (input_w, input_h),
        score_threshold = SCORE_THRESHOLD,
        nms_threshold   = NMS_THRESHOLD,
        top_k           = TOP_K,
        backend_id      = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id       = cv2.dnn.DNN_TARGET_CPU
    )
    print("[YuNet] Model loaded")
    return detector

def detect_faces(detector, frame_320x240):
    """
    Input : 320x240 BGR frame
    Output: list of dicts with keys -- bbox, landmarks, confidence
            Returns [] if no faces found
    """
    _, faces = detector.detect(frame_320x240)

    results = []
    if faces is None:
        return results

    for face in faces:
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        confidence  = float(face[-1])
        landmarks   = face[4:14].reshape(5, 2).astype(int)

        results.append({
            "bbox":       (x, y, w, h),
            "landmarks":  landmarks,
            "confidence": confidence
        })

    return results

def extract_face_crop(frame_320x240, bbox, margin=10):
    """
    Crops and normalizes the face region for MobileFaceNet input.
    Returns: 112x112 float32 face crop normalized to [-1, 1], or None if out of bounds
    """
    x, y, w, h = bbox
    H, W = frame_320x240.shape[:2]

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(W, x + w + margin)
    y2 = min(H, y + h + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    crop     = frame_320x240[y1:y2, x1:x2]
    face_112 = cv2.resize(crop, (112, 112))
    face_norm = (face_112.astype(np.float32) - 127.5) / 127.5

    return face_norm
