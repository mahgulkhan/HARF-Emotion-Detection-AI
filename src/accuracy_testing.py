"""
Evaluates the tabularisai pretrained transformer model against the same
hand-labeled test set used for the Logistic Regression baseline, for a
direct, fair comparison.
"""
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from predict_pretrained import predict


os.environ["HF_HUB_OFFLINE"] = "1"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(_THIS_DIR, "..", "data", "test_data.csv")

df = pd.read_csv(TEST_PATH)

def get_prediction(text):
    mapped, raw, conf = predict(text)
    return mapped

df["predicted"] = df["text"].apply(get_prediction)
df["correct"] = df["predicted"] == df["true_emotion"]

acc = accuracy_score(df["true_emotion"], df["predicted"])
print(f"Accuracy on manual test set (tabularisai transformer): {acc:.1%}  ({df['correct'].sum()}/{len(df)} correct)\n")
print(classification_report(df["true_emotion"], df["predicted"], zero_division=0))

print("\nWrong predictions:")
print(df[~df["correct"]][["text", "true_emotion", "predicted"]].to_string(index=False))