import cv2
import numpy as np

CAPTURE_WIDTH  = 320
CAPTURE_HEIGHT = 240
TARGET_WIDTH   = 320
TARGET_HEIGHT  = 240
DEVICE_INDEX   = 0

def init_camera():
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam at /dev/video0")
    print("[Camera] Initialized")
    return cap

def capture_frame(cap):
    cap.grab()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[Camera] Frame capture failed -- skip event")
        return None, None

    resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    return frame, resized
