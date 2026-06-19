import cv2
import threading
import time
import pandas as pd
from pynput import mouse
from pathlib import Path

# --- Configuration and Path Setup ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "attention_data.csv"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# --- Cascade Classifier Loading ---
face_path = str((MODEL_DIR / "haarcascade_frontalface_default.xml").resolve())
eye_path = str((MODEL_DIR / "haarcascade_eye.xml").resolve())

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

if face_cascade.empty() or eye_cascade.empty():
    print("WARNING: Haar Cascade XML files NOT loaded.")

# --- Global State Variables ---
capture_running = False
capture_thread = None
mouse_data = {"x": 0, "y": 0, "clicks": 0}


print("Background capture started. Logging to:", LOG_FILE)

    while capture_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        face_detected = False
        eye_gaze = "models_failed"
        eye_state = "unknown"
        head_pose = "unknown"

        if face_models_ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            face_detected = len(faces) > 0
            eye_gaze = "not_detected"

            if face_detected:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                roi_gray = gray[y:y+h, x:x+w]

                # --- Head Pose (face center heuristic) ---
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                face_center_x = x + w / 2
                face_center_y = y + h / 2

                dx = face_center_x - frame_center_x
                dy = face_center_y - frame_center_y

                if abs(dx) < w * 0.15 and abs(dy) < h * 0.15:
                    head_pose = "center"
                elif dx > w * 0.15:
                    head_pose = "right"
                elif dx < -w * 0.15:
                    head_pose = "left"
                elif dy < -h * 0.15:
                    head_pose = "up"
                elif dy > h * 0.15:
                    head_pose = "down"

                # --- Eye Detection ---
                if eye_models_ok:
                    eyes = eye_cascade.detectMultiScale(roi_gray)

                    if len(eyes) >= 2:
                        eye_state = "open"

                        roi_center_x = w / 2
                        eye_mean_x = sum([ex + ew / 2 for ex, ey, ew, eh in eyes]) / len(eyes)
                        deviation = eye_mean_x - roi_center_x

                        if deviation < -w / 10:
                            eye_gaze = "left"
                        elif deviation > w / 10:
                            eye_gaze = "right"
                        else:
                            eye_gaze = "center"
                    else:
                        eye_state = "closed"
                        eye_gaze = "face_only"

        # --- Collect Data ---
        data = {
            "timestamp": time.time(),
            "face_detected": face_detected,
            "eye_gaze": eye_gaze,
            "eye_state": eye_state,
            "head_pose": head_pose,
            "mouse_x": mouse_data["x"],
            "mouse_y": mouse_data["y"],
            "mouse_clicks": mouse_data["clicks"]
        }

        data_list.append(data)

        if len(data_list) >= save_interval:
            df = pd.DataFrame(data_list)
            df.to_csv(LOG_FILE, mode="a",
                      header=not LOG_FILE.exists(), index=False)
            data_list = []

        time.sleep(1)

    if data_list:
        pd.DataFrame(data_list).to_csv(
            LOG_FILE, mode="a",
            header=not LOG_FILE.exists(), index=False
        )

    cap.release()
    mouse_listener.stop()
    cv2.destroyAllWindows()
    print("Background capture process cleanly stopped.")

def stop_capture():
    global capture_running
    if capture_running:
        capture_running = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=2)
