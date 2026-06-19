import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


def _load_dataset(feature_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(feature_csv)

# Train-test split (small data → keep it simple)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

class AttentionModel:
    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.label = values[np.argmax(counts)]
        self.classes = values.tolist()
        return self

    def predict(self, X):
        return np.full(len(X), self.label)

    def predict_proba(self, X):
        probs = np.zeros((len(X), len(self.classes)))
        idx = self.classes.index(self.label)
        probs[idx] = 1.0
        return probs


def train_attention_model(feature_csv: Path, model_path: Path):
    df = _load_dataset(feature_csv)
    X, y = _prepare_data(df)

    y_pred = model.predict(X_test)
    accuracy = float((y_pred == y_test.values).mean())


    print("Model Accuracy:", round(accuracy, 3))
    print(report)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print("Model saved to:", model_path)


    return model
