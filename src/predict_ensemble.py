import sys
sys.path.insert(0, ".")
from predict import load_model, predict_emotion as lr_predict
from predict_pretrained import predict as transformer_predict

lr_model = load_model()

def predict_ensemble(text):
    lr_label = lr_predict(text, lr_model)

    trans_mapped, trans_raw, trans_conf = transformer_predict(text)

    # Rule 1: agreement - high trust
    if lr_label == trans_mapped:
        return lr_label, "agreement"

    # Rule 2: transformer says Neutral but LR disagrees - trust LR
    # (the transformer defaults to Neutral when unsure, especially for Anger)
    if trans_mapped == "Neutral":
        return lr_label, "LR override (transformer defaulted to Neutral)"

    # Rule 3: disagreement, transformer confident and specific - trust transformer
    if trans_conf >= 0.6:
        return trans_mapped, "transformer (high confidence)"

    # Fallback: trust LR
    return lr_label, "LR fallback (low transformer confidence)"

if __name__ == "__main__":
    tests = [
        "mujhe bohat gussa aa raha hai",
        "is beizzati ka badla zaroor loonga",
        "thora scared hoon exam ke bare mein",
    ]
    for t in tests:
        label, reason = predict_ensemble(t)
        print(f"{t!r:50s} -> {label:10s} ({reason})")