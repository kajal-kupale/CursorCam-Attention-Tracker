from pathlib import Path
from utils.feature_engineering import generate_features

BASE_DIR = Path(__file__).resolve().parent

RAW_CSV = BASE_DIR / "logs" / "attention_data.csv"
LABELED_CSV = BASE_DIR / "outputs" / "attention_labeled.csv"
OUTPUT_CSV = BASE_DIR / "outputs" / "attention_features.csv"

if __name__ == "__main__":
    print("Running Step-1: Feature Engineering")

    df = generate_features(RAW_CSV, LABELED_CSV, OUTPUT_CSV)

    print("Feature engineering completed")
    print("Final dataset shape:", df.shape)
    print("Saved to:", OUTPUT_CSV)
