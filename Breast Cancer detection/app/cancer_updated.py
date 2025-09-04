import numpy as np
import streamlit as st
import joblib


# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    layout="centered",
    page_icon="🎗️"
)


# -------------------------------
# Custom CSS with animated gradient & styled inputs
# -------------------------------
st.markdown("""
<style>
/* Animated gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #a1c4fd, #c2e9fb, #fbc2eb, #fad0c4);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    min-height: 100vh;
}


/* Gradient animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}


/* App title */
.main-title {
    font-size: 56px; 
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    letter-spacing: 3px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.25);
    margin-bottom: 2rem;
}


/* Input container - fully transparent */
.input-container {
    background-color: rgba(255, 255, 255, 0.0);
    border-radius: 20px;
    max-width: 560px;
    margin: 0 auto 3rem auto;
    padding: 0px;
}


/* Streamlit number inputs - much bigger bars */
input[data-baseweb="number"] {
    font-size: 26px;                /* even bigger text */
    border-radius: 16px;
    border: 2.5px solid #adb5bd;
    padding: 20px 28px;             /* more padding for height & space inside */
    width: 98vw !important;         /* use viewport width for maximum stretch */
    max-width: 940px;               /* set a high max-width for desktop, adjust as desired */
    min-width: 400px;               /* ensure it's big even on small screens */
    box-sizing: border-box;
    transition: border-color 0.3s, box-shadow 0.3s, transform 0.2s;
    outline: none;
    background-color: rgba(255, 255, 255, 0.88);
    margin-bottom: 18px;
    margin-top: 10px;
}

@media (max-width: 600px) {
  input[data-baseweb="number"] {
    min-width: 90vw;
    max-width: 100vw;
    font-size: 20px;
    padding: 14px 12px;
  }
}


/* Focus effect with glowing shadow */
input[data-baseweb="number"]:focus {
    border-color: #0069d9;
    box-shadow: 0 0 14px rgba(0,123,255,0.7);
    background-color: rgba(255,255,255,0.95);
    transform: scale(1.02);
}


/* Hover effect with subtle glow */
input[data-baseweb="number"]:hover {
    box-shadow: 0 0 10px rgba(0,123,255,0.4);
    transform: scale(1.01);
    transition: all 0.3s ease;
}


/* Labels - bigger text */
label {
    font-weight: 700;
    font-size: 26px;        /* increased font size */
    margin-top: 1.5rem;
    color: #34495e;
    display: block;
    margin-bottom: 0.5rem;
}


/* Predict button */
.stButton > button {
    background-color: #007bff;
    color: white;
    font-size: 22px;
    font-weight: 700;
    padding: 15px 30px;
    border-radius: 40px;
    box-shadow: 0 6px 14px rgba(0, 123, 255, 0.35);
    transition: background-color 0.35s, box-shadow 0.35s, transform 0.2s;
    width: 100%;
    cursor: pointer;
    margin-top: 25px;
    letter-spacing: 1.2px;
}
.stButton > button:hover {
    background-color: #0056b3;
    box-shadow: 0 8px 20px rgba(0, 86, 179, 0.65);
    transform: translateY(-2px);
}
.stButton > button:focus {
    outline: 3px solid #80bdff;
    outline-offset: 2px;
}


/* Result box with hover & fade-in */
.result-box {
    padding: 1.5rem 2rem;
    font-size: 24px;
    font-weight: 700;
    border-radius: 18px;
    max-width: 560px;
    margin: 1rem auto;
    text-align: center;
    letter-spacing: 0.7px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    opacity: 0;
    transform: translateY(20px);
    animation: fadeIn 0.8s forwards;
}
.result-box:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}
.malignant {
    background-color: #dc3545 !important;
    color: white !important;
    text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
}
.benign {
    background-color: #28a745 !important;
    color: white !important;
    text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
}


@keyframes fadeIn {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>


<script>
const inputs = Array.from(document.querySelectorAll('input[type=number]'));
inputs.forEach((input, idx) => {
    input.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            if (idx < inputs.length - 1) {
                inputs[idx+1].focus();
            } else {
                const btn = document.querySelector('button[kind="primary"]');
                if(btn) btn.focus();
            }
        }
    });
});
</script>
""", unsafe_allow_html=True)


# -------------------------------
# App Title
# -------------------------------
st.markdown('<h1 class="main-title">Breast Cancer Prediction App 🎗️</h1>', unsafe_allow_html=True)


# -------------------------------
# Input Fields
# -------------------------------
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    mean_concave_points = st.number_input('Mean Concave Points', value=0.0, format="%.4f")
    worst_radius = st.number_input('Worst Radius', value=0.0, format="%.4f")
    worst_perimeter = st.number_input('Worst Perimeter', value=0.0, format="%.4f")
    worst_area = st.number_input('Worst Area', value=0.0, format="%.4f")
    worst_concave_points = st.number_input('Worst Concave Points', value=0.0, format="%.4f")
    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------
# Load Model
# -------------------------------
try:
    saved_model = joblib.load('Cancer_model_AI.joblib')
    model = saved_model['model']
except FileNotFoundError:
    st.error("❌ Model file not found! Make sure 'Cancer_model_AI.joblib' is in the same folder.")
    model = None


# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔮 Predict"):
    if model:
        input_array = np.array([
            mean_concave_points,
            worst_radius,
            worst_perimeter,
            worst_area,
            worst_concave_points
        ]).reshape(1, -1)
        prediction = model.predict(input_array)


        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Prediction Result:")


        if prediction[0] == 0:
            st.markdown("<div class='result-box malignant'>🩸 The tumor is <b>Malignant (Cancerous)</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box benign'>✅ The tumor is <b>Benign (Not Cancerous)</b></div>", unsafe_allow_html=True)
    else:
        st.warning("Model not loaded, prediction unavailable.")
