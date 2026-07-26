"""
Loads the trained emotion classifier and classifies new text -
either one sentence at a time, or edit the `tests` list below to try your own.
"""
import re
import os
import joblib
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "..", "models", "model.joblib")


def clean_text(s: str) -> str:
    """Same cleaning logic used during training - must match train_dataset.py"""
    s = str(s).lower().strip()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\u0600-\u06FF\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict_emotion(text: str, model=None) -> str:
    if model is None:
        model = load_model()
    cleaned = clean_text(text)
    if not cleaned:
        return "Neutral"
    return model.predict([cleaned])[0]

def predict_batch(df: pd.DataFrame, text_col: str = "message", model=None) -> pd.DataFrame:
    if model is None:
        model = load_model()
    cleaned = df[text_col].apply(clean_text)
    df = df.copy()
    df["emotion"] = model.predict(cleaned)
    return df


if __name__ == "__main__":
    m = load_model()
    tests = [
        # Anger
        "mujhe bohat gussa aa raha hai",
        "yaar tum hamesha late kyun aate ho, gussa aa jata hai",
        "is se zyada frustrating kuch nahi ho sakta",
        "chup kar warna gussa aa jayega mujhe",

        # Fear
        "thora scared hoon exam ke bare mein",
        "kal mujhe interview hai bohat tension ho rahi hai",
        "akele ghar jaate huay dar lagta hai raat ko",
        "result aane wala hai dar lag raha hai bohat",

        # Sad
        "yar mood off ho gaya today",
        "dil bohat udaas hai aj kal",
        "kisi se baat karne ka dil nahi kar raha",
        "bohat akela feel ho raha hai mujhe",

        # Happy
        "I am so happy yr, best din ever!",
        "yeh news sun kar to mera din ban gaya",
        "aj bohat maza aya sab ke sath",
        "finally result acha aya, khushi ka thikana nahi",

        # Surprise
        "yeh kya ho gaya achanak se",
        "mujhe bilkul yaqeen nahi aa raha",
        "achanak purana dost mil gaya raste mein",

        # Neutral
        "kal meeting hai office mein",
        "bas ghar par hoon abhi kuch khaas nahi",
        "assignment submit karni hai kal tak",

        # Genuinely ambiguous / hard cases - worth discussing, not just reporting
        "acha",                                   # too short/vague - real limitation
        "nahi yaar chor do",                       # could be sad, angry, or neutral depending on tone
        "aj mummy ne khana banaya hai",
    ]
    for t in tests:
        print(f"{t!r:55s} -> {predict_emotion(t, m)}")