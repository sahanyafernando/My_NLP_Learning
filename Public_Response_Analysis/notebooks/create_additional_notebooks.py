import json
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_NOTEBOOK = BASE_DIR / "01_data_loading_and_preprocessing.ipynb"


def base_metadata():
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    return nb.get("metadata", {}), nb.get("nbformat", 4), nb.get("nbformat_minor", 2)


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }


def code(source: str):
    # store as list of lines
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def save_nb(cells, filename, metadata, nbformat, nbformat_minor):
    nb = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": nbformat,
        "nbformat_minor": nbformat_minor,
    }
    (BASE_DIR / filename).write_text(json.dumps(nb, indent=2), encoding="utf-8")


def nb04_sentiment_classification(meta, nf, nfm):
    cells = []
    cells.append(
        md(
            "# 04 – Supervised Sentiment Classification\n\n"
            "This notebook trains classical machine learning classifiers on the multilingual policy dataset\n"
            "to predict sentiment labels (positive / negative / neutral) using TF-IDF and other features."
        )
    )
    cells.append(
        code(
            """
from google.colab import drive
drive.mount('/content/drive')
"""
        )
    )
    cells.append(
        code(
            """
import pickle, pathlib

artifacts_root = pathlib.Path("/content/drive/MyDrive/My_NLP_Learning/Public_Response_Analysis")
artifacts_path = artifacts_root / "artifacts/preprocessing_outputs.pkl"

if artifacts_path.exists():
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    df = artifacts["df"]
    tfidf_vectorizer = artifacts["tfidf_vectorizer"]
    tfidf_matrix = artifacts["tfidf_matrix"]
    print("Loaded preprocessing artifacts and TF-IDF features.")
else:
    raise FileNotFoundError(
        "Artifacts not found. Please run 01_data_loading_and_preprocessing.ipynb first "
        "and execute the 'Save preprocessing artifacts' cell."
    )
"""
        )
    )
    cells.append(
        md(
            "## Prepare train/test splits\n\n"
            "We use the TF-IDF matrix as features and the `sentiment_label` column as the target."
        )
    )
    cells.append(
        code(
            """
from sklearn.model_selection import train_test_split

X = tfidf_matrix
y = df["sentiment_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print(y_train.value_counts())
"""
        )
    )
    cells.append(
        md(
            "## Train baseline classifiers\n\n"
            "We train and evaluate Naïve Bayes, Linear SVM, and Decision Tree classifiers."
        )
    )
    cells.append(
        code(
            """
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    "MultinomialNB": MultinomialNB(),
    "LinearSVC": LinearSVC(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
}

results = {}

for name, clf in models.items():
    print(f"\\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    results[name] = acc

print("\\nSummary of model accuracies:", results)
"""
        )
    )
    cells.append(
        md(
            "## Language-wise performance (optional)\n\n"
            "Evaluate how well the best model performs across different languages."
        )
    )
    cells.append(
        code(
            """
best_model_name = max(results, key=results.get)
print("Best model:", best_model_name)
best_clf = models[best_model_name]

import numpy as np

y_pred_all = best_clf.predict(X)

df_eval = df.copy()
df_eval["y_pred"] = y_pred_all

for lang, group in df_eval.groupby("language"):
    print(f"\\nLanguage: {lang}")
    print(classification_report(group["sentiment_label"], group["y_pred"]))
"""
        )
    )

    save_nb(cells, "04_sentiment_classification.ipynb", meta, nf, nfm)


def nb05_ner_aspect_temporal(meta, nf, nfm):
    cells = []
    cells.append(
        md(
            "# 05 – Entity Extraction, Aspect-Based Sentiment, and Temporal Analysis\n\n"
            "This notebook performs transformer-based NER, simple aspect-based sentiment analysis\n"
            "around policy topics, and temporal sentiment trend exploration."
        )
    )
    cells.append(
        code(
            """
from google.colab import drive
drive.mount('/content/drive')
"""
        )
    )
    cells.append(
        code(
            """
import pickle, pathlib

artifacts_root = pathlib.Path("/content/drive/MyDrive/My_NLP_Learning/Public_Response_Analysis")
artifacts_path = artifacts_root / "artifacts/preprocessing_outputs.pkl"

if artifacts_path.exists():
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    df = artifacts["df"]
    print("Loaded preprocessing artifacts and DataFrame.")
else:
    raise FileNotFoundError(
        "Artifacts not found. Please run 01_data_loading_and_preprocessing.ipynb first "
        "and execute the 'Save preprocessing artifacts' cell."
    )
"""
        )
    )
    cells.append(
        md(
            "## Transformer-based Named Entity Recognition (NER)\n\n"
            "We use a multilingual transformer model for NER to extract entities mentioned in posts."
        )
    )
    cells.append(
        code(
            """
!pip install -q transformers sentencepiece

from transformers import pipeline

ner_pipeline = pipeline(
    task="ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple",
)

sample_texts = df["text"].head(5).tolist()
for text in sample_texts:
    print("\\nText:", text)
    ents = ner_pipeline(text)
    print("Entities:", ents)
"""
        )
    )
    cells.append(
        md(
            "## Aspect-based sentiment around policy topics\n\n"
            "We use the existing sentiment labels and topics as a simple ABSA setting:\n"
            "for each `topic`, we analyze the distribution of sentiment and key example posts."
        )
    )
    cells.append(
        code(
            """
topic_sent = (
    df.groupby(["topic", "sentiment_label"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

print("Sentiment distribution per topic:")
print(topic_sent)

for topic in df["topic"].unique():
    print(f"\\n=== Topic: {topic} ===")
    subset = df[df["topic"] == topic]
    print("Example positive posts:")
    print(subset[subset["sentiment_label"] == "positive"]["text"].head(3).to_string(index=False))
    print("\\nExample negative posts:")
    print(subset[subset["sentiment_label"] == "negative"]["text"].head(3).to_string(index=False))
"""
        )
    )
    cells.append(
        md(
            "## Temporal sentiment trends\n\n"
            "We convert timestamps to datetime, aggregate sentiment over time,\n"
            "and visualize changes across events and languages."
        )
    )
    cells.append(
        code(
            """
import pandas as pd
import matplotlib.pyplot as plt

df_time = df.copy()
df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])

sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
df_time["sentiment_score"] = df_time["sentiment_label"].map(sentiment_map)

daily = df_time.set_index("timestamp").groupby([pd.Grouper(freq="D")])["sentiment_score"].mean()

plt.figure(figsize=(10, 4))
daily.plot(marker="o")
plt.title("Average sentiment over time")
plt.xlabel("Date")
plt.ylabel("Average sentiment score")
plt.grid(True)
plt.show()
"""
        )
    )
    cells.append(
        md(
            "### Simple change-point style analysis\n\n"
            "We flag days where the sentiment deviates strongly from the overall mean\n"
            "as potential change-points related to key events."
        )
    )
    cells.append(
        code(
            """
import numpy as np

mean_sent = daily.mean()
std_sent = daily.std()
threshold = mean_sent + 1.0 * std_sent

print("Global mean sentiment:", mean_sent)
print("Global std sentiment:", std_sent)

anomalies = daily[daily > threshold]
print("\\nPotential positive sentiment spikes:")
print(anomalies)

threshold_neg = mean_sent - 1.0 * std_sent
anomalies_neg = daily[daily < threshold_neg]
print("\\nPotential negative sentiment drops:")
print(anomalies_neg)
"""
        )
    )

    save_nb(cells, "05_ner_aspect_temporal.ipynb", meta, nf, nfm)


def nb06_summarization(meta, nf, nfm):
    cells = []
    cells.append(
        md(
            "# 06 – Extractive and Abstractive Summarization\n\n"
            "This notebook summarizes multilingual policy response posts using TextRank (extractive)\n"
            "and transformer-based models (BART/T5) for abstractive summaries on English text."
        )
    )
    cells.append(
        code(
            """
from google.colab import drive
drive.mount('/content/drive')
"""
        )
    )
    cells.append(
        code(
            """
import pickle, pathlib

artifacts_root = pathlib.Path("/content/drive/MyDrive/My_NLP_Learning/Public_Response_Analysis")
artifacts_path = artifacts_root / "artifacts/preprocessing_outputs.pkl"

if artifacts_path.exists():
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    df = artifacts["df"]
    print("Loaded preprocessing artifacts and DataFrame.")
else:
    raise FileNotFoundError(
        "Artifacts not found. Please run 01_data_loading_and_preprocessing.ipynb first "
        "and execute the 'Save preprocessing artifacts' cell."
    )
"""
        )
    )
    cells.append(
        md(
            "## Extractive summarization with TextRank\n\n"
            "We apply a simple TextRank-style approach over English posts to extract the most\n"
            "representative sentences for each policy topic."
        )
    )
    cells.append(
        code(
            """
!pip install -q summa

from summa.summarizer import summarize

english_df = df[df["language"] == "en"]

for topic in english_df["topic"].unique():
    subset = english_df[english_df["topic"] == topic]
    long_text = "\\n".join(subset["text"].tolist())
    print(f"\\n===== Topic: {topic} =====")
    try:
        summary = summarize(long_text, ratio=0.3)
        print("Extractive summary:")
        print(summary)
    except ValueError:
        print("Not enough text for summarization.")
"""
        )
    )
    cells.append(
        md(
            "## Abstractive summarization with BART/T5\n\n"
            "We use a pretrained transformer summarization pipeline (BART) on English text.\n"
            "For other languages, you can translate to English first or use multilingual T5 models."
        )
    )
    cells.append(
        code(
            """
!pip install -q transformers sentencepiece

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

sample_posts = english_df["text"].head(5).tolist()
for i, text in enumerate(sample_posts, start=1):
    print(f"\\n--- Post {i} ---")
    print("Original:", text)
    summary = summarizer(text, max_length=60, min_length=15, do_sample=False)[0]["summary_text"]
    print("Abstractive summary:", summary)
"""
        )
    )

    save_nb(cells, "06_summarization.ipynb", meta, nf, nfm)


def main():
    meta, nf, nfm = base_metadata()
    nb04_sentiment_classification(meta, nf, nfm)
    nb05_ner_aspect_temporal(meta, nf, nfm)
    nb06_summarization(meta, nf, nfm)


if __name__ == "__main__":
    main()


