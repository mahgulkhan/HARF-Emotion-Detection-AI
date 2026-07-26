import joblib
import pandas as pd
import re

def clean_text(s):
    s = str(s).lower().strip()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\u0600-\u06FF\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Load model
model = joblib.load('../models/model.joblib')

# Load data
df = pd.read_csv('../data/ruemocorp_annotated.csv')

print("Checking if your test examples exist in training data:\n")

test_texts = [
    "yar mood off ho gaya today",
    "I am so happy yr, best din ever!",
    "mujhe bohat gussa aa raha hai",
    "thora scared hoon exam ke bare mein",
]

for test in test_texts:
    # Find similar texts
    words = test.split()[:3]
    pattern = '|'.join(words)
    similar = df[df['text'].str.contains(pattern, case=False, na=False)]
    
    print(f"YOUR TEXT: {test}")
    print(f"Found {len(similar)} similar examples in training data")
    if len(similar) > 0:
        print("Examples from training data:")
        for i, row in similar.head(3).iterrows():
            print(f"  -> {row['text'][:50]}... = {row['emotion']}")
    else:
        print("   No similar examples found! Model doesn't know these phrases.")
    print("-" * 50)