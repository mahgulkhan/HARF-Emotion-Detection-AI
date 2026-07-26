import pandas as pd

SOURCE_PATH = r"E:\Summer Courses\Maisons Consultings\RU-EN-Emotion Dataset.xlsx"
OUTPUT_PATH = "../data/emotion_dataset.csv"

df = pd.read_excel(SOURCE_PATH, sheet_name="Annotation Dataset")
df = df[["Tweets", "Level 2"]].dropna()
df.columns = ["text", "emotion"]
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
print(df["emotion"].value_counts())