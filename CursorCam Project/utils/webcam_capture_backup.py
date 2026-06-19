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

        face_detected = True
        eye_gaze = "center"
        eye_state = "open"
        head_pose = "center"

        row = {
            "timestamp": time.time(),
            "face_detected": face_detected,
            "eye_gaze": eye_gaze,
            "eye_state": eye_state,
            "head_pose": head_pose,
            "mouse_x": mouse_data["x"],
            "mouse_y": mouse_data["y"],
            "mouse_clicks": mouse_data["clicks"]
        }

        df = pd.DataFrame([row])

        if LOG_FILE.exists():
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)

        time.sleep(1)

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