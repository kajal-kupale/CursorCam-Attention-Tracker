import pandas as pd

def predict_attention(feature_dict):

    face_ratio = feature_dict.get("face_ratio", 0)
    eye_open_ratio = feature_dict.get("eye_open_ratio", 0)
    gaze_focus_score = feature_dict.get("gaze_focus_score", 0)

    confidence = (
        face_ratio +
        eye_open_ratio +
        gaze_focus_score
    ) / 3

    prediction = 1 if confidence >= 0.5 else 0

    return prediction, confidence