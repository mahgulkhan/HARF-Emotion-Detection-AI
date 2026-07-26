from datasets import load_dataset
import pandas as pd

ds = load_dataset("Khubaib01/RUEmoCorp", "ruemocorp-annotated")
df = ds["train"].to_pandas()

df = df.rename(columns={"message": "text", "emotion_label": "emotion"})
df["emotion"] = df["emotion"].str.capitalize()   # match your existing Anger/Happy/... style

df.to_csv("../data/ruemocorp_annotated.csv", index=False)
print(f"Saved {len(df)} rows")
print(df["emotion"].value_counts())