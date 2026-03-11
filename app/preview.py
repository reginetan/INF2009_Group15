from picamera2 import Picamera2
import cv2
import numpy as np

cam = Picamera2()
cfg = cam.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
cam.configure(cfg)
cam.start()

print("Camera preview started. Press Q to quit.", flush=True)

while True:
    rgb = cam.capture_array()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair
    cv2.line(bgr, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
    cv2.line(bgr, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

    # Centre crop sharpness indicator
    crop = bgr[cy-40:cy+40, cx-40:cx+40]
    if crop.size > 0:
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        color = (0, 255, 0) if sharp > 100 else (0, 165, 255) if sharp > 30 else (0, 0, 255)
        cv2.putText(bgr, f"Sharpness: {sharp:.0f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(bgr, "FOCUS OK" if sharp > 100 else "ADJUST FOCUS", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow("Camera Preview — Press Q to quit", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()