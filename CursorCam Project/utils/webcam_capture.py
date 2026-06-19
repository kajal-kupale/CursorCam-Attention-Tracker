import cv2
import threading
import time
import pandas as pd
from pynput import mouse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "attention_data.csv"

LOG_DIR.mkdir(exist_ok=True)

capture_running = False
capture_thread = None

mouse_data = {
    "x": 0,
    "y": 0,
    "clicks": 0
}


def on_move(x, y):
    mouse_data["x"] = x
    mouse_data["y"] = y


def on_click(x, y, button, pressed):
    if pressed:
        mouse_data["clicks"] += 1


def capture_loop():
    global capture_running

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click
    )
    listener.start()

    while capture_running:

        ret, frame = cap.read()

        if not ret:
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        if len(faces) > 0:

            face_detected = True

            x, y, w, h = faces[0]

            center_x = x + w // 2
            frame_center = frame.shape[1] // 2

            # Head pose
            if center_x < frame_center - 100:
                head_pose = "right"
            elif center_x > frame_center + 100:
                head_pose = "left"
            else:
                head_pose = "center"

            if y > frame.shape[0] * 0.60:
                head_pose = "down"

            roi_gray = gray[y:y+h, x:x+w]

            # Eye state / gaze
            if roi_gray.mean() > 60:
                eye_state = "open"
                eye_gaze = "center"
            else:
                eye_state = "closed"
                eye_gaze = "face_only"

        else:

            face_detected = False
            eye_gaze = "not_detected"
            eye_state = "closed"
            head_pose = "not_detected"

        current_time = time.localtime()

        if face_detected:
            attention_status = "ATTENTIVE"
        else:
            attention_status = "NOT_ATTENTIVE"

        row = {
            "date": time.strftime("%Y-%m-%d", current_time),
            "time": time.strftime("%H:%M:%S", current_time),
            "face_detected": face_detected,
            "eye_gaze": eye_gaze,
            "eye_state": eye_state,
            "head_pose": head_pose,
            "mouse_x": mouse_data["x"],
            "mouse_y": mouse_data["y"],
            "mouse_clicks": mouse_data["clicks"],
            "attention_status": attention_status
        }

        df = pd.DataFrame([row])

        if LOG_FILE.exists():
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)

        time.sleep(0.2)

    cap.release()
    listener.stop()


def start_capture():
    global capture_running
    global capture_thread

    if capture_running:
        return

    capture_running = True

    capture_thread = threading.Thread(
        target=capture_loop,
        daemon=True
    )

    capture_thread.start()


def stop_capture():
    global capture_running
    capture_running = False