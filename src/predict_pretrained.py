from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
os.environ["HF_HUB_OFFLINE"] = "1"


MODEL_NAME = "tabularisai/multilingual-emotion-classification"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Confirmed directly from the model's own config - do not hand-edit this
id_to_label = {
    0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "frustration",
    5: "gratitude", 6: "joy", 7: "love", 8: "neutral", 9: "sadness", 10: "surprise"
}

# Maps the model's 11 labels down to your project's 6 categories
merge_map = {
    "anger": "Anger", "contempt": "Anger", "frustration": "Anger",
    "fear": "Fear",
    "sadness": "Sad",
    "joy": "Happy", "love": "Happy", "gratitude": "Happy",
    "surprise": "Surprise",
    "neutral": "Neutral", "disgust": "Neutral",
}

def predict(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0]  # multi-label -> sigmoid, not softmax

    # Take the single highest-probability label as the primary prediction
    top_id = torch.argmax(probs).item()
    top_label = id_to_label[top_id]
    confidence = probs[top_id].item()
    mapped = merge_map[top_label]
    return mapped, top_label, confidence

if __name__ == "__main__":
    tests = [
        "yar mood off ho gaya today",
        "I am so happy yr, best din ever!",
        "mujhe bohat gussa aa raha hai",
        "thora scared hoon exam ke bare mein",
    ]
    for t in tests:
        mapped, raw, conf = predict(t)
        print(f"{t!r:50s} -> {mapped:10s} (raw: {raw}, confidence: {conf:.2f})")