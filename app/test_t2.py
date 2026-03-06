import sys
import cv2
from frame_capture import capture_frame, init_camera
from face_detector import init_yunet, detect_faces, extract_face_crop
from face_recogniser import FaceRecogniser

direction  = sys.argv[1] if len(sys.argv) > 1 else "ENTRY"
cap        = init_camera()
detector   = init_yunet(input_w=320, input_h=240)
recogniser = FaceRecogniser()

print(f"Testing T2 pipeline (direction={direction}) -- press q to quit")
print("-" * 60)

last_label = "No face"
last_color = (0, 255, 255)
last_bbox  = None

try:
    while True:
        raw_frame, frame_320 = capture_frame(cap)
        if frame_320 is None:
            continue

        faces = detect_faces(detector, frame_320)
        display = frame_320.copy()

        if faces:
            best_face = max(faces, key=lambda f: f["confidence"])
            face_crop = extract_face_crop(frame_320, best_face["bbox"])

            if face_crop is not None:
                t2_input = {
                    "face_crop":  face_crop,
                    "direction":  direction,
                    "bbox":       best_face["bbox"],
                    "landmarks":  best_face["landmarks"],
                    "confidence": best_face["confidence"],
                }
                result = recogniser.identify_from_t2(t2_input)

                x, y, w, h = best_face["bbox"]
                if result["match"]:
                    last_color = (0, 255, 0)
                    last_label = f"{result['name']} ({result['confidence']:.2f})"
                    print(f"  MATCH | {result['student_id']} | conf={result['confidence']:.4f}")
                else:
                    last_color = (0, 0, 220)
                    last_label = f"NO MATCH ({result['confidence']:.2f})"

                cv2.rectangle(display, (x, y), (x + w, y + h), last_color, 2)
                cv2.putText(display, last_label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, last_color, 1)

        cv2.putText(display, f"Dir: {direction}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("T2 Pipeline", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cv2.destroyAllWindows()
