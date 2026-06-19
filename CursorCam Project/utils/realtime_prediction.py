


def predict_attention(feature_dict):

    face_ratio = feature_dict.get("face_ratio", 0)

    if face_ratio > 0.5:
        return 1, 1.0
    else:
        return 0, 0.0