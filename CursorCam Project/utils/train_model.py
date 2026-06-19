import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_attention_model(feature_csv: Path, model_path: Path):

    df = pd.read_csv(feature_csv)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    X = df.drop(columns=["attention_label"])

    y = df["attention_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nModel Accuracy:", round(accuracy, 3))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved to:", model_path)

    return model