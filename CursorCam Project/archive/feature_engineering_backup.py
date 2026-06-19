import pandas as pd
from pathlib import Path

WINDOW_SIZE = 5


def generate_features(raw_csv: Path, labeled_csv: Path, output_csv: Path):
    raw_df = pd.read_csv(raw_csv)
    labeled_df = pd.read_csv(labeled_csv)

    feature_rows = []

    prev_attention = None

    for _, label_row in labeled_df.iterrows():
        start_time = label_row["start_time"]
        end_time = label_row["end_time"]

        window = raw_df[
            (raw_df["timestamp"] >= start_time) &
            (raw_df["timestamp"] <= end_time)
        ]

        if window.empty:
            continue

        # Mouse click rate
        click_rate = window["mouse_clicks"].diff().clip(lower=0).sum() / WINDOW_SIZE

        # Feature extraction 
        mouse_movement_intensity = None
        mouse_click_rate =  window["mouse_x"].diff().abs().sum(
        gaze_focus_score = window["mouse_y"].diff().abs().sum()


        feature_rows.append({
            "face_ratio": label_row.get("face_ratio"),
            "eye_open_ratio": label_row.get("eye_open_ratio"),
            "head_center_ratio": label_row.get("head_center_ratio"),
            "mouse_activity": label_row.get("mouse_activity"),
            "mouse_movement_intensity": mouse_movement_intensity,
            "mouse_click_rate": mouse_click_rate,
            "gaze_focus_score": gaze_focus_score,
            "attention_continuity": prev_attention,
            "attention_label": label_row.get("attention_label")
        })

        prev_attention = label_row.get("attention_label")

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(output_csv, index=False)

    return features_df
