import pandas as pd
from pathlib import Path

WINDOW_SIZE = 5


def label_attention(input_csv: Path, output_csv: Path):

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    labeled_rows = []

    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):

        window = df.iloc[i:i + WINDOW_SIZE]

        face_ratio = window["face_detected"].mean()

        eye_open_ratio = (
            window["eye_state"] == "open"
        ).mean()

        head_center_ratio = (
            window["head_pose"] == "center"
        ).mean()

        mouse_activity = (
            window["mouse_x"].diff().abs().sum()
            +
            window["mouse_y"].diff().abs().sum()
            +
            window["mouse_clicks"].diff().fillna(0).sum()
        )

        attention_label = int(
            face_ratio >= 0.6
            and eye_open_ratio >= 0.5
            and head_center_ratio >= 0.5
        )

        labeled_rows.append({
            "start_time": window.iloc[0]["timestamp"],
            "end_time": window.iloc[-1]["timestamp"],
            "face_ratio": face_ratio,
            "eye_open_ratio": eye_open_ratio,
            "head_center_ratio": head_center_ratio,
            "mouse_activity": mouse_activity,
            "attention_label": attention_label
        })

    labeled_df = pd.DataFrame(labeled_rows)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    labeled_df.to_csv(
        output_csv,
        index=False
    )

    return labeled_df