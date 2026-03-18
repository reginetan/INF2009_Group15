import face_recognition
import cv2
import numpy as np
import time
import pickle

# ============================================================
# STRANGER DETECTION THRESHOLD
# ============================================================
# Faces below this distance are considered a MATCH.
# Faces at or above this distance are labeled "STRANGER".
#
# Tuning guide:
#   0.35 = very strict
#   0.40 = strict
#   0.45 = moderate (good starting point)
#   0.50 = lenient
#   0.60 = default dlib (too loose)
# ============================================================
STRANGER_THRESHOLD = 0.35

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

print(f"[INFO] loaded {len(known_face_encodings)} encodings for {len(set(known_face_names))} people")
print(f"[INFO] enrolled: {sorted(set(known_face_names))}")
print(f"[INFO] stranger threshold: {STRANGER_THRESHOLD}")

# Initialize the USB webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("ERROR: Could not open webcam. Check USB connection.")
    exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allow camera to warm up
time.sleep(2)

# Initialize our variables
cv_scaler = 4  # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Track last logged name to avoid spamming the same result
last_logged_name = None


def process_frame(frame):
    global face_locations, face_encodings, face_names, last_logged_name

    # Resize the frame using cv_scaler to increase performance
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))

    # Convert BGR to RGB (face_recognition uses RGB, OpenCV uses BGR)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []

    for face_encoding in face_encodings:
        name = "STRANGER"
        best_distance = 1.0

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            if best_distance < STRANGER_THRESHOLD:
                name = known_face_names[best_match_index]

        face_names.append(name)

        # Log to terminal (only when the detected name changes)
        if name != last_logged_name:
            if name == "STRANGER":
                print(f"[STRANGER] Unknown face detected (distance: {best_distance:.3f})")
            else:
                print(f"[MATCH] {name} recognised (distance: {best_distance:.3f})")
            last_logged_name = name

    # Reset when no faces are in frame
    if len(face_encodings) == 0 and last_logged_name is not None:
        last_logged_name = None

    return frame


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Color: green for known, red for stranger
        if name == "STRANGER":
            box_color = (0, 0, 255)       # red
            label_color = (0, 0, 200)      # dark red
        else:
            box_color = (0, 200, 0)        # green
            label_color = (0, 160, 0)      # dark green

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)

        # Draw label background
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), label_color, cv2.FILLED)

        # Draw name only
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 0.8, (255, 255, 255), 1)

    return frame


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


while True:
    ret, frame = cam.read()
    if not ret:
        continue

    processed_frame = process_frame(frame)
    display_frame = draw_results(processed_frame)

    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', display_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cam.release()