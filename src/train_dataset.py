"""
Baseline Roman Urdu / English code-switched emotion classifier.

Usage:
    python train_dataset.py --data path/to/dataset.csv --augment path/to/augment.csv
"""
import argparse
import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import matplotlib.pyplot as plt

TEXT_CANDIDATES = ["text", "sentence", "message", "tweet", "content", "review"]
LABEL_CANDIDATES = ["emotion", "label", "sentiment", "class", "category"]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_THIS_DIR, "..", "models")
_OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "output")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_OUTPUTS_DIR, exist_ok=True)


def autodetect_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def clean_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\u0600-\u06FF\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_data(path, text_col, label_col):
    df = pd.read_csv(path)
    if text_col is None:
        text_col = autodetect_column(df.columns, TEXT_CANDIDATES)
    if label_col is None:
        label_col = autodetect_column(df.columns, LABEL_CANDIDATES)

    if text_col is None or label_col is None:
        print(f"Could not auto-detect columns. Found columns: {list(df.columns)}")
        print("Pass --text_col and --label_col explicitly.")
        sys.exit(1)

    print(f"Using text column: '{text_col}'  |  label column: '{label_col}'")
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "emotion"]
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    return df


def balance_classes(df, target_count=2000):
    balanced_parts = []
    for label, group in df.groupby("emotion"):
        if len(group) < target_count:
            group = resample(group, replace=True, n_samples=target_count, random_state=42)
        balanced_parts.append(group)
    return pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--augment", default=None, help="Optional path to a small hand-crafted CSV (text,emotion) merged into the TRAINING split only")
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--balance", action="store_true", default=True)
    parser.add_argument("--no-balance", dest="balance", action="store_false")
    parser.add_argument("--target_count", type=int, default=2000)
    args = parser.parse_args()

    df = load_data(args.data, args.text_col, args.label_col)
    print(f"\nLoaded {len(df)} rows after cleaning.")
    print("\nClass distribution (full dataset):")
    print(df["emotion"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], test_size=args.test_size,
        random_state=42, stratify=df["emotion"] if df["emotion"].nunique() > 1 else None,
    )

    if args.augment:
        aug_df = pd.read_csv(args.augment)
        aug_df["text"] = aug_df["text"].apply(clean_text)
        print(f"\nMerging {len(aug_df)} augmentation rows into TRAINING split only.")
        X_train = pd.concat([X_train, aug_df["text"]], ignore_index=True)
        y_train = pd.concat([y_train, aug_df["emotion"]], ignore_index=True)

    if args.balance:
        train_df = pd.DataFrame({"text": X_train, "emotion": y_train})
        train_df = balance_classes(train_df, target_count=args.target_count)
        X_train, y_train = train_df["text"], train_df["emotion"]
        print("\nTraining class distribution after balancing:")
        print(y_train.value_counts())
    else:
        print("\nTraining on raw distribution.")

    print(f"\nTest set size: {len(X_test)} (untouched, original data only)")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\nAccuracy: {acc:.3f}\n")
    print(report)

    with open(os.path.join(_OUTPUTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write(report)

    labels = sorted(df["emotion"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (accuracy={acc:.2f})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = os.path.join(_OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\nSaved {cm_path}")

    model_path = os.path.join(_MODELS_DIR, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()