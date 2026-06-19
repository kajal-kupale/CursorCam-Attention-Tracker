from pathlib import Path
from utils.train_model import train_attention_model

BASE_DIR = Path(__file__).resolve().parent

FEATURE_CSV = BASE_DIR / "outputs" / "attention_features.csv"
MODEL_PATH = BASE_DIR / "outputs" / "attention_model.pkl"

if __name__ == "__main__":
    print("Running Step-3: Model Training")

    train_attention_model(FEATURE_CSV, MODEL_PATH)

    print("Step-3 completed successfully")
