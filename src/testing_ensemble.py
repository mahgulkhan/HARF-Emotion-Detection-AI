import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from predict_ensemble import predict_ensemble

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(_THIS_DIR, "..", "data", "test_data.csv")

df = pd.read_csv(TEST_PATH)

def get_prediction(text):
    label, reason = predict_ensemble(text)
    return label

df["predicted"] = df["text"].apply(get_prediction)
df["correct"] = df["predicted"] == df["true_emotion"]

acc = accuracy_score(df["true_emotion"], df["predicted"])
print(f"Accuracy (ensemble): {acc:.1%}  ({df['correct'].sum()}/{len(df)} correct)\n")
print(classification_report(df["true_emotion"], df["predicted"], zero_division=0))