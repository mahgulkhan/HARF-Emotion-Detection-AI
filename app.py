import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from whatsapp_parser import parse_chat
from predict import predict_batch, predict_emotion, load_model

st.set_page_config(page_title="harf-حرف", page_icon="\u2726", layout="wide")

# ---------------- Background image (embedded so it works locally AND once deployed) ----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BG_PATH = os.path.join(_THIS_DIR, "pic_source", "bg.jpg")

bg_css = ""
if os.path.exists(_BG_PATH):
    with open(_BG_PATH, "rb") as f:
        bg_b64 = base64.b64encode(f.read()).decode()
    bg_css = f"""
    .stApp {{
        background-image: linear-gradient(rgba(251,241,225,0.55), rgba(245,228,200,0.7)),
                           url("data:image/jpg;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    bg_css = """
    .stApp {
        background: linear-gradient(180deg, #FBF1E1 0%, #F5E4C8 100%);
    }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Katibeh&display=swap');

html {{ font-size: 20px; }}

body, .stApp, p, h1, h2, h3, h4, h5, button,
input, textarea, label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stWidgetLabel"] p,
[data-testid="stFileUploaderFileName"],
[data-testid="stDataFrame"] {{
    font-family: 'Katibeh', -apple-system, "Segoe UI", Arial, sans-serif !important;
    font-size: 1.2rem;
}}

[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
span[class*="material-symbols"] {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}}

div[data-testid="stTextInput"] input{{
    font-size: 1.5rem !important;
    padding: 0.8rem !important;
    height: auto !important;
}}
{bg_css}

[data-testid="stAppViewContainer"] .main .block-container {{
    background-color: rgba(251, 241, 225, 0.93);
    border-radius: 18px;
    padding: 2.5rem 3rem 3rem 3rem;
    margin-top: 1.5rem;
    margin-bottom: 2rem;
    max-width: 1100px;
}}

[data-testid="stHeader"] {{
    background-color: #EDDBBE  !important;
    position: relative;
}}

[data-testid="stHeader"]::before {{
    content: "حرف - Roman Urdu Emotion Detector AI";
    color: #6D2E32;
    font-size: 1.7rem;
    font-family: 'Katibeh', -apple-system, sans-serif;
    letter-spacing: 0.03em;
    position: absolute;
    left: 1.5rem;
    top: 50%;
    transform: translateY(-50%);
    white-space: nowrap;
}}

.harf-header {{
    text-align: center;
    padding: 1rem 0 0.6rem 0;
    border-bottom: 2px solid #B76E27;
    margin-bottom: 1.5rem;
}}

.harf-title-en {{
    font-size: 5.5rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em;
    background: linear-gradient(90deg, #8B5A2B 0%, #C9A227 50%, #6D2E32 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}}
.harf-title-ur {{
    font-size: 4rem !important;
    color: #6D2E32;
    margin: 0.2rem 0 0 0;
    direction: rtl;
}}
.harf-tagline {{
    color: #8B5A2B;
    font-size: 1.4rem !important;
    margin-top: 0.5rem;
}}
div.stButton > button {{
    background-color: #B76E27;
    color: #FBF1E1;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 1.2rem;
    padding: 0.5rem 1.5rem;
}}
div.stButton > button:hover {{
    background-color: #6D2E32;
    color: #FBF1E1;
}}
section[data-testid="stFileUploader"] {{
    background-color: #EDDBBE;
    border: 1px solid #B76E27;
    border-radius: 8px;
    padding: 0.5rem;
}}
h2, h3 {{
    color: #6D2E32 !important;
}}

@media (max-width: 768px) {{
    .harf-title-en {{ font-size: 3rem !important; }}
    .harf-title-ur {{ font-size: 2.2rem !important; }}
    .harf-tagline {{ font-size: 1rem !important; }}
    [data-testid="stAppViewContainer"] .main .block-container {{
        padding: 1.2rem;
        margin: 0.5rem;
    }}
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="harf-header">
    <p class="harf-title-en">harf</p>
    <p class="harf-title-ur">حرف</p>
    <p class="harf-tagline">Roman Urdu &amp; English Chat Emotion Detector</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


st.subheader("Check a single message")
single_text = st.text_input("Type a sentence to classify:")
if st.button("Check emotion"):
    if single_text.strip():
        model = get_model()
        result = predict_emotion(single_text, model)
        st.success(f"Predicted emotion: **{result}**")
    else:
        st.warning("Please type something first.")

st.divider()

st.subheader("Or analyze a full WhatsApp chat")

with st.expander("How do I export a chat from WhatsApp on iPhone?"):
    st.markdown(
        "1. Open the chat in WhatsApp\n"
        "2. Tap the contact/group name at the top\n"
        "3. Scroll down and tap **Export Chat**\n"
        "4. Choose **Without Media**\n"
        "5. Save the resulting `.txt` file, then upload it below"
    )

uploaded_file = st.file_uploader("Upload exported chat (.txt)", type=["txt"])

if uploaded_file is not None:
    temp_path = "temp_uploaded_chat.txt"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        df = parse_chat(temp_path)
    except ValueError as e:
        st.error(str(e))
        df = None

    if df is not None and not df.empty:
        model = get_model()
        result = predict_batch(df, text_col="message", model=model)

        st.success(f"Parsed {len(result)} messages from {result['sender'].nunique()} sender(s).")

        st.markdown("**Emotion breakdown**")
        counts = result["emotion"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values, color="#B76E27")
        ax.set_ylabel("Number of messages")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

        st.markdown("**Emotion by sender**")
        sender_emotion = pd.crosstab(result["sender"], result["emotion"])
        st.bar_chart(sender_emotion)

        st.markdown("**Message-by-message results**")
        display_df = result[["timestamp", "sender", "message", "emotion"]].sort_values("timestamp")
        st.dataframe(display_df, width="stretch", hide_index=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv, "emotion_results.csv", "text/csv")
    elif df is not None and df.empty:
        st.warning("No messages could be parsed from this file.")

    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    st.info("Waiting for a chat export file to analyze.")
