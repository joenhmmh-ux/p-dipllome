import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from .paths import ALBANIAN_DATASET, ENGLISH_DATASETS, SENTIMENT_MODEL_FILE, VECTORIZER_FILE
except ImportError:
    try:
        from paths import ALBANIAN_DATASET, ENGLISH_DATASETS, SENTIMENT_MODEL_FILE, VECTORIZER_FILE
    except ImportError:
        from Source.paths import ALBANIAN_DATASET, ENGLISH_DATASETS, SENTIMENT_MODEL_FILE, VECTORIZER_FILE

# LABEL_MAP = {0.0: 0, 1.0: 1, 0.5: 2}
LABEL_MAP = {"0":0, "1":1, "2":2, "0/1":2}
DATASETS = [
    {
        "path": ALBANIAN_DATASET,
        "text_column": "Comment",
        "label_column": "Sentiment",
        "language": "sq",
    },
    {
        "path": ENGLISH_DATASETS,
        "text_column": "sentence",
        "label_column": "sentiment",
        "language": "en",
    },
]
MAX_SAMPLES_PER_LABEL = {
    "sq": None,
    "en": 100000,
}
MIN_SAMPLES_PER_LABEL = {
    0: 5000,
    1: 5000,
    2: 5000,
}


frames = []
for dataset in DATASETS:
    if isinstance(dataset["path"], (list, tuple)):
        df = pd.concat((pd.read_csv(path) for path in dataset["path"]), ignore_index=True)
    else:
        df = pd.read_csv(dataset["path"])
    print(f"\nDataset: {dataset['path']}")
    print("Kolonat:", df.columns.tolist())

    df = df.dropna(subset=[dataset["text_column"], dataset["label_column"]]).copy()
    df["text"] = df[dataset["text_column"]].astype(str).str.strip()
    df["sentiment_raw"] = df[dataset["label_column"]].astype(str).str.strip()
    df = df[df["sentiment_raw"].isin(LABEL_MAP)]
    df["label"] = df["sentiment_raw"].map(LABEL_MAP)
    df["language"] = dataset["language"]

    max_samples = MAX_SAMPLES_PER_LABEL.get(dataset["language"])
    if max_samples is not None:
        sampled_parts = []
        for label, group in df.groupby("label", sort=True):
            sampled_parts.append(group.sample(n=min(len(group), max_samples), random_state=42))
        df = pd.concat(sampled_parts, ignore_index=True)

    print("Shperndarja e klasave:")
    print(df["sentiment_raw"].value_counts())
    frames.append(df[["text", "label", "language"]])

df = pd.concat(frames, ignore_index=True)

balanced_parts = []
for label, group in df.groupby("label", sort=True):
    target = max(len(group), MIN_SAMPLES_PER_LABEL.get(label, len(group)))
    balanced_parts.append(group.sample(n=target, replace=len(group) < target, random_state=42))
df = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nTotali i komenteve per trajnim:", len(df))
print("Shperndarja totale e klasave:")
print(df["label"].value_counts().sort_index())

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),
    min_df=2,
    max_features=60000,
    sublinear_tf=True,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negativ", "Pozitiv", "Neutral"]))

with open(VECTORIZER_FILE, "wb") as f:
    pickle.dump(vectorizer, f)

with open(SENTIMENT_MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("\nModelet u ruajten me sukses.")
