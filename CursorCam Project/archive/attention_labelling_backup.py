import pandas as pd
from pathlib import Path


WINDOW_SIZE = 5  # 5 rows ≈ 5 seconds


def label_attention(input_csv: Path, output_csv: Path):
    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    labeled_data = []

    for i in range(1, len(df), WINDOW_SIZE):
        window = df.iloc[i:i + WINDOW_SIZE]

        if len(window) < WINDOW_SIZE:
            continue

                # --- Compute Ratios ---
        face_ratio = window["face_detected"].mean()
        eye_open_ratio = (window["eye_state"] == "open").mean()
        head_center_ratio = (window["head_pose"] == "center").mean()

            # --- Mouse Activity Check ---
        mouse_activity = (
            window["mouse_x"].diff().abs().sum() > 0 or
            window["mouse_y"].diff().abs().sum() > 0 or
            window["mouse_clicks"].diff().sum() > 0
        )
        labeled_data.append({
            "start_time": window.iloc[0]["timestamp"],
            "end_time": window.iloc[-1]["timestamp"],
            "face_ratio": face_ratio,
            "eye_open_ratio": eye_open_ratio,
            "head_center_ratio": head_center_ratio,
            "mouse_activity": mouse_activity,
            "attention_label": attention_label
        })

    labeled_df = pd.DataFrame(labeled_data)
    labeled_df.to_csv(output_csv, index=False)

    return labeled_df
