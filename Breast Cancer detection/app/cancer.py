import numpy as np
import streamlit as st
import joblib

# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== CUSTOM CSS ===================
page_bg = """
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    text-align: center;
    color: #ffffff;
    font-size: 48px !important;
    font-weight: bold;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
    margin-bottom: 15px;
}

/* Subtitle */
h5 {
    text-align: center;
    color: #f0f0f0 !important;
    font-size: 20px !important;
    margin-bottom: 40px;
}

/* Input Labels */
label {
    color: #f5f5f5 !important;
    font-size: 15px !important;
    font-weight: bold !important;
}

/* Number Input Styling */
.stNumberInput input {
    border-radius: 8px;
    padding: 8px;
    font-size: 15px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    padding: 0.7em 1.5em;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    transition: 0.3s ease-in-out;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #e60039, #ff1e56);
    transform: scale(1.05);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.4);
}

/* Prediction Result Box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =================== TITLE ===================
st.title("🩺 Breast Cancer Prediction App")
st.markdown("<h5>Fill in the patient’s diagnostic details and click Predict</h5>", unsafe_allow_html=True)

# =================== INPUT LAYOUT ===================
col1, col2, col3 = st.columns(3)

with col1:
    mean_radius = st.number_input('Mean Radius', value=0.0, format="%.4f")
    mean_texture = st.number_input('Mean Texture', value=0.0, format="%.4f")
    mean_perimeter = st.number_input('Mean Perimeter', value=0.0, format="%.4f")
    mean_area = st.number_input('Mean Area', value=0.0, format="%.4f")
    mean_smoothness = st.number_input('Mean Smoothness', value=0.0, format="%.4f")
    mean_compactness = st.number_input('Mean Compactness', value=0.0, format="%.4f")
    mean_concavity = st.number_input('Mean Concavity', value=0.0, format="%.4f")
    mean_concave_points = st.number_input('Mean Concave Points', value=0.0, format="%.4f")
    mean_symmetry = st.number_input('Mean Symmetry', value=0.0, format="%.4f")
    mean_fractal_dimension = st.number_input('Mean Fractal Dimension', value=0.0, format="%.4f")

with col2:
    radius_error = st.number_input('Radius Error', value=0.0, format="%.4f")
    texture_error = st.number_input('Texture Error', value=0.0, format="%.4f")
    perimeter_error = st.number_input('Perimeter Error', value=0.0, format="%.4f")
    area_error = st.number_input('Area Error', value=0.0, format="%.4f")
    smoothness_error = st.number_input('Smoothness Error', value=0.0, format="%.4f")
    compactness_error = st.number_input('Compactness Error', value=0.0, format="%.4f")
    concavity_error = st.number_input('Concavity Error', value=0.0, format="%.4f")
    concave_points_error = st.number_input('Concave Points Error', value=0.0, format="%.4f")
    symmetry_error = st.number_input('Symmetry Error', value=0.0, format="%.4f")
    fractal_dimension_error = st.number_input('Fractal Dimension Error', value=0.0, format="%.4f")

with col3:
    worst_radius = st.number_input('Worst Radius', value=0.0, format="%.4f")
    worst_texture = st.number_input('Worst Texture', value=0.0, format="%.4f")
    worst_perimeter = st.number_input('Worst Perimeter', value=0.0, format="%.4f")
    worst_area = st.number_input('Worst Area', value=0.0, format="%.4f")
    worst_smoothness = st.number_input('Worst Smoothness', value=0.0, format="%.4f")
    worst_compactness = st.number_input('Worst Compactness', value=0.0, format="%.4f")
    worst_concavity = st.number_input('Worst Concavity', value=0.0, format="%.4f")
    worst_concave_points = st.number_input('Worst Concave Points', value=0.0, format="%.4f")
    worst_symmetry = st.number_input('Worst Symmetry', value=0.0, format="%.4f")
    worst_fractal_dimension = st.number_input('Worst Fractal Dimension', value=0.0, format="%.4f")

# =================== MODEL LOAD ===================
try:
    saved_model = joblib.load('breast_cancer_model.joblib')
    model = saved_model['model']
    cols = saved_model['cols']
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'breast_cancer_model.joblib' is in the same directory.")
    model = None

# =================== PREDICTION ===================
# =================== PREDICTION ===================
if st.button("🔮 Predict"):
    if model is not None:
        input_data = [
            mean_radius, mean_texture, mean_perimeter, mean_area,
            mean_smoothness, mean_compactness, mean_concavity,
            mean_concave_points, mean_symmetry, mean_fractal_dimension,
            radius_error, texture_error, perimeter_error, area_error,
            smoothness_error, compactness_error, concavity_error,
            concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_perimeter, worst_area,
            worst_smoothness, worst_compactness, worst_concavity,
            worst_concave_points, worst_symmetry, worst_fractal_dimension
        ]
        
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Prediction Result:")

        # ✅ Correct mapping based on sklearn dataset
        if prediction[0] == 0:
            st.markdown("<div class='result-box' style='background-color:#ff4d4d; color:white;'>🩸 The model predicts the tumor is <b>Malignant (Cancerous)</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box' style='background-color:#28a745; color:white;'>✅ The model predicts the tumor is <b>Benign (Not Cancerous)</b></div>", unsafe_allow_html=True)
