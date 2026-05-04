import csv, json, pickle, re
import os
from flask import Flask, render_template, request

try:
    from .paths import (
        ALBANIAN_DATASET,
        DATA_DIR,
        ENGLISH_DATASETS,
        LANGUAGE_TEXT_FILE,
        SENTIMENT_MODEL_FILE,
        VECTORIZER_FILE,
        WEB_DIR,
    )
except ImportError:
    try:
        from paths import (
            ALBANIAN_DATASET,
            DATA_DIR,
            ENGLISH_DATASETS,
            LANGUAGE_TEXT_FILE,
            SENTIMENT_MODEL_FILE,
            VECTORIZER_FILE,
            WEB_DIR,
        )
    except ImportError:
        from Source.paths import (
            ALBANIAN_DATASET,
            DATA_DIR,
            ENGLISH_DATASETS,
            LANGUAGE_TEXT_FILE,
            SENTIMENT_MODEL_FILE,
            VECTORIZER_FILE,
            WEB_DIR,
        )

# ... / Model-Fjale-Mire-Keq / ...
app = Flask(
    __name__,
    template_folder=WEB_DIR,
    static_folder=WEB_DIR,
    static_url_path="/static",
)

# LANGUAGE_TEXT
with open(LANGUAGE_TEXT_FILE, "r", encoding="utf-8") as file:
    LANGUAGE_TEXT = json.load(file)

# Ngarkimi i modelit dhe vectorizer
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

with open(SENTIMENT_MODEL_FILE, "rb") as f:
    model = pickle.load(f)

def load_terms(path):
    terms = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            terms.extend(term.strip().lower() for term in row if term.strip())
    return set(terms)


POSITIVE_WORDS = load_terms(DATA_DIR / "positive_words.csv")
NEGATIVE_WORDS = load_terms(DATA_DIR / "negative_words.csv")
NEGATIONS = load_terms(DATA_DIR / "negation_words.csv")
POSITIVE_PHRASES = load_terms(DATA_DIR / "positive_phrases.csv")
NEGATIVE_PHRASES = load_terms(DATA_DIR / "negative_phrases.csv")
NEUTRAL_WORDS = load_terms(DATA_DIR / "neutral_words.csv")
NEUTRAL_PHRASES = load_terms(DATA_DIR / "neutral_phrases.csv")


def build_stat_items(counts, total, label_order, stat_labels):
    return [
        {
            "label_key": label_key,
            "label": stat_labels[label_key],
            "count": counts.get(source_label, 0),
            "percent": round(counts.get(source_label, 0) * 100 / total, 2) if total else 0,
            "css": css_class,
        }
        for source_label, label_key, css_class in label_order
    ]


def read_label_counts(paths, field_name, allowed_labels, encoding):
    counts = {label: 0 for label in allowed_labels}
    total = 0

    if not isinstance(paths, (list, tuple)):
        paths = (paths,)

    for path in paths:
        with open(path, newline="", encoding=encoding) as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = str(row.get(field_name, "")).strip()
                if label in counts:
                    counts[label] += 1
                    total += 1

    return counts, total


def load_dataset_stats(language):
    text = LANGUAGE_TEXT[language]
    if language == "en":
        counts, total = read_label_counts(
            ENGLISH_DATASETS,
            "sentiment",
            ("1", "0", "2"),
            "utf-8",
        )
        return {
            "key": "en",
            "title": text["stats_title_en"],
            "total": total,
            "items": build_stat_items(
                counts,
                total,
                (
                    ("1", "positive", "positive"),
                    ("0", "negative", "negative"),
                    ("2", "neutral", "neutral"),
                ),
                text["stat_labels"],
            ),
            "note": text["stats_note_en"],
        }

    counts, total = read_label_counts(
        ALBANIAN_DATASET,
        "Sentiment",
        ("1", "0", "2"),
        "utf-8-sig",
    )
    return {
        "key": "sq",
        "title": text["stats_title_sq"],
        "total": total,
        "items": build_stat_items(
            counts,
            total,
            (
                ("1", "positive", "positive"),
                ("0", "negative", "negative"),
                ("2", "neutral", "neutral"),
            ),
            text["stat_labels"],
        ),
        "note": None,
    }

def normalize_text(text):
    lowered = text.lower()
    cleaned = re.sub(r"[^0-9a-zA-Zçë\s']", " ", lowered)
    return ''.join(re.sub(r"\s+", " ", cleaned).strip())


def keyword_sentiment(text):
    normalized = normalize_text(text)
    if not normalized:
        return None

    tokens = normalized.split()
    if len(tokens) < 2:
        return None
    for phrase in NEUTRAL_PHRASES:
        if phrase in normalized:
            return 2

    score = 0

    for phrase in NEGATIVE_WORDS:
        if phrase in normalized:
            score -= 4
    for phrase in POSITIVE_WORDS:
        if phrase in normalized:
            score += 3

    for phrase in POSITIVE_PHRASES:
        if " " in phrase and phrase in normalized:
            score += 2
    for phrase in NEGATIVE_PHRASES:
        if " " in phrase and phrase in normalized:
            score -= 2

    has_neutral = any(token in NEUTRAL_WORDS for token in tokens)
    has_positive = False
    has_negative = False

    for index in range(len(tokens)):
        window_two = " ".join(tokens[index:index + 2])
        window_three = " ".join(tokens[index:index + 3])

        if window_three in POSITIVE_PHRASES or window_two in POSITIVE_PHRASES:
            score += 3
            has_positive = True
        if window_three in NEGATIVE_PHRASES or window_two in NEGATIVE_PHRASES:
            score -= 3
            has_negative = True

    for index, token in enumerate(tokens):
        if token in NEUTRAL_WORDS:
            has_neutral = True

        if token in NEGATIONS:
            window = tokens[index + 1:index + 4]
            if any(candidate in POSITIVE_WORDS for candidate in window):
                score -= 2
                has_negative = True
            if any(candidate in NEGATIVE_WORDS for candidate in window):
                score += 2
                has_positive = True

    if has_neutral or (has_positive and has_negative):
        return 2

    if score >= 2:
        return 1
    if score <= -2:
        return 0
    return None


def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vectorized)[0]
    ml_prediction = int(model.predict(text_vectorized)[0])
    keyword_prediction = keyword_sentiment(text)
    confidence = float(max(probabilities))

    if keyword_prediction is not None:
        if ml_prediction == 2 and keyword_prediction in (0, 1) and confidence < 0.85:
            return keyword_prediction
        if len(text.split()) <= 4 and confidence < 0.60:
            return keyword_prediction
        if confidence < 0.70 and keyword_prediction != ml_prediction:
            return keyword_prediction

    return ml_prediction


def build_stats_sections(language):
    return [load_dataset_stats(language)]


@app.route("/", methods=["GET", "POST"])
def home():
    language = request.values.get("language") or request.args.get("lang") or "sq"
    if language not in LANGUAGE_TEXT:
        language = "sq"

    text = LANGUAGE_TEXT[language]
    result = None
    result_key = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("comment", "")

        if user_text.strip() != "":
            prediction = predict_sentiment(user_text)
            result_key = str(prediction)
            if result_key not in text["labels"]:
                result_key = "2"
            result = text["labels"][result_key]

    return render_template(
        "index.html",
        language=language,
        texts=text,
        translations=LANGUAGE_TEXT,
        result=result,
        result_key=result_key,
        user_text=user_text,
        stats_sections=build_stats_sections(language),
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
