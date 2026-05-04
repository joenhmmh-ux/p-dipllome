from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
WEB_DIR = BASE_DIR / "Web"

ALBANIAN_DATASET = BASE_DIR / "com-shqip.csv"
ENGLISH_DATASETS = tuple(sorted(DATA_DIR.glob("train_data_en_part_*.csv")))
LANGUAGE_TEXT_FILE = DATA_DIR / "language_text.json"

VECTORIZER_FILE = SOURCE_DIR / "vectorizer.pkl"
SENTIMENT_MODEL_FILE = SOURCE_DIR / "sentiment_model.pkl"
