import sys
sys.path.insert(0, ".")
from predict import load_model, predict_emotion as lr_predict
from predict_pretrained import predict as transformer_predict

lr_model = load_model()

def predict_ensemble(text):
    lr_label = lr_predict(text, lr_model)
    trans_mapped, trans_raw, trans_conf = transformer_predict(text)

    if lr_label == trans_mapped:
        return lr_label, "agreement"
    if trans_mapped == "Neutral":
        return lr_label, "LR override (transformer defaulted to Neutral)"
    if trans_conf >= 0.85:   # <- the fix, changed from 0.6
        return trans_mapped, "transformer (high confidence)"
    return lr_label, "LR fallback (low transformer confidence)"

if __name__ == "__main__":
    tests = [
    # ===== ANGER =====
    "ap bohat buray insaan hain",
    "kya masla ha",
    "bas kar do",
    "mujhe tang mat karo",
    "yeh kya bakwas hai",
    "bohat gussa aa raha hai",
    "insaaf nahi hai yeh",
    "tum ne yeh kya kar diya",
    "kitna bura insaan hai",
    "ghussa aa gaya",
    "naraz hoon main",
    "bura laga yeh sun ke",
    
    # ===== FEAR =====
    "mujhe dar lag raha hai",
    "kya hoga kal",
    "bohat tension hai",
    "safety ki fikr hai",
    "pata nahi kya ho ga",
    "darr lagta hai mujhe",
    "yeh halat dekh ke dar lagta",
    "kuch bura hone wala hai",
    
    # ===== SAD =====
    "dil toot gaya",
    "bohat dukh hai",
    "bura lag raha hai",
    "ammi ki yaad aa rahi",
    "zindagi mushkil ho gayi",
    "kash woh waqt wapas aa jaye",
    "akela pan bohat takleef deta",
    "ab kuch acha nahi lagta",
    "woh din kabhi wapas nahi aayenge",
    "dil mein dard hai",
    "bohat udaas hoon",
    
    # ===== HAPPY =====
    "maza aa gaya",
    "bohat khushi ho rahi",
    "best din ever",
    "mubarak ho",
    "Alhamdulillah sab theek ho gaya",
    "yaar kya din hai aaj",
    "khush hoon bohat",
    "acha laga sun ke",
    "wow kya baat hai",
    
    # ===== SURPRISE =====
    "yeh kya ho raha hai",
    "sach hai yeh",
    "yakeen nahi ho raha",
    "hairan hoon main",
    "kya dekh liya",
    "uff ye kya ker rhy",
    "yaar yeh toh unexpected tha",
    
    # ===== DISGUST =====
    "yeh acha nahi hai",
    "bohat buri cheez hai",
    "nafrat hai mujhe is se",
    "ghin aa gayi",
    "pasand nahi yeh",
    "bekaar hai yeh",
    "bura taste hai",
    
    # ===== NEUTRAL =====
    "theek hai",
    "acha hai",
    "koi baat nahi",
    "bas yahi hai",
    "samajh gaya",
    "chalo theek hai",
    "koi masla nahi",
    "yeh scene hai",
    "main samajh gaya",
    "theek hai chalo",
    
    # ===== MIXED / CODE-SWITCHED =====
    "yar mood off ho gaya today",
    "I am so happy yr best din ever",
    "thora scared hoon exam ke bare mein",
    "mujhe bohat gussa aa raha hai",
    "yaar kya scene hai yeh",
    "literally crying rn",
    "feeling so low today",
    "can't believe yeh ho gaya",
    "wow this is amazing",
    "so done with this",
]
    for t in tests:
        lr_only = lr_predict(t, lr_model)          # <- new line
        ensemble, reason = predict_ensemble(t)      # <- new line
        print(f"{t!r:50s} LR: {lr_only:10s} Ensemble: {ensemble:10s} ({reason})")