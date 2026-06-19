from pathlib import Path
from utils.attention_labelling import label_attention

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "logs" / "attention_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "attention_labeled.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    print("Running Step-2: Attention Labeling")

    labeled_df = label_attention(INPUT_CSV, OUTPUT_CSV)

    print("Labeling completed successfully")
    print("Total labeled windows:", len(labeled_df))
    print("Saved to:", OUTPUT_CSV)
