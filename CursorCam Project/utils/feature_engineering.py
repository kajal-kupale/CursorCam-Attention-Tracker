import pandas as pd
from pathlib import Path

WINDOW_SIZE = 5


def generate_features(raw_csv: Path, labeled_csv: Path, output_csv: Path):

    raw_df = pd.read_csv(raw_csv)
    labeled_df = pd.read_csv(labeled_csv)

    feature_rows = []

    prev_attention = 0

    for _, label_row in labeled_df.iterrows():

        start_time = label_row["start_time"]
        end_time = label_row["end_time"]

        window = raw_df[
            (raw_df["timestamp"] >= start_time)
            &
            (raw_df["timestamp"] <= end_time)
        ]

        if window.empty:
            continue

        mouse_movement_intensity = (
            window["mouse_x"].diff().abs().fillna(0).sum()
            +
            window["mouse_y"].diff().abs().fillna(0).sum()
        )

        mouse_click_rate = (
            window["mouse_clicks"].diff().clip(lower=0).fillna(0).sum()
            / WINDOW_SIZE
        )

        gaze_focus_score = (
            window["eye_gaze"] == "center"
        ).mean()

        feature_rows.append({
            "face_ratio": label_row["face_ratio"],
            "eye_open_ratio": label_row["eye_open_ratio"],
            "head_center_ratio": label_row["head_center_ratio"],
            "mouse_activity": label_row["mouse_activity"],
            "mouse_movement_intensity": mouse_movement_intensity,
            "mouse_click_rate": mouse_click_rate,
            "gaze_focus_score": gaze_focus_score,
            "attention_continuity": prev_attention,
            "attention_label": label_row["attention_label"]
        })

        prev_attention = label_row["attention_label"]

    features_df = pd.DataFrame(feature_rows)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        output_csv,
        index=False
    )

    return features_df