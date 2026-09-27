import streamlit as st
import joblib
import numpy as np

# ---------- Page config ----------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS Theme ----------
st.markdown("""
<style>
    /* Gradient dark background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* Main title - gradient text */
    h1 {
        font-family: 'Segoe UI', 'Poppins', sans-serif !important;
        background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
        letter-spacing: -0.5px;
    }

    /* Section headers */
    h3 {
        color: #92fe9d !important;
        font-family: 'Segoe UI', sans-serif !important;
        border-bottom: 1px solid rgba(146, 254, 157, 0.3);
        padding-bottom: 6px;
    }

    /* Body text */
    p, label, span, div {
        color: #e0f2f7 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Slider thumb - cyan */
    .stSlider [role="slider"] {
        background-color: #00c9ff !important;
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.8) !important;
        border: 2px solid #ffffff !important;
    }

    /* Slider track - gradient */
    .stSlider [data-baseweb="slider"] > div > div > div > div {
        background: linear-gradient(90deg, #00c9ff, #92fe9d) !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(0, 201, 255, 0.08);
        border: 1px solid rgba(0, 201, 255, 0.3);
        border-radius: 14px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #00c9ff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #92fe9d !important;
        font-weight: 600 !important;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
        color: #0f2027 !important;
        border: none;
        border-radius: 12px;
        padding: 0.75em 2em;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(0, 201, 255, 0.5);
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border-left: 5px solid !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(0, 201, 255, 0.08) !important;
        border: 1px solid rgba(0, 201, 255, 0.3) !important;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load models ----------
reg_model    = joblib.load("diabetes_reg_model.joblib")
clf_model    = joblib.load("diabetes_clf_model.joblib")
median_score = joblib.load("diabetes_median.joblib")

# ---------- Real-world statistics (sklearn diabetes original values) ----------
REAL_STATS = {
    "age":               (48.518, 13.109),
    "bmi":               (26.376, 4.418),
    "blood_pressure":    (94.647, 13.831),
    "total_cholesterol": (189.140, 34.809),
    "ldl":               (115.445, 30.413),
    "hdl":               (49.788, 12.934),
    "chol_hdl_ratio":    (4.070, 1.291),
    "triglycerides":     (4.477, 1.281),
    "blood_sugar":       (91.445, 11.547),
}
SQRT_N = 21.024   # sqrt(442), sklearn's scaling factor

SEX_SCALED = {"Female": -0.0446, "Male": 0.0507}

def to_model_scale(value, feature):
    """Convert real-world value to sklearn-scaled value."""
    mean, std = REAL_STATS[feature]
    return (value - mean) / (std * SQRT_N)

# ---------- Header ----------
st.title("🩺 Diabetes Progression & Risk Predictor")
st.write("Enter the patient's measurements below. The model will estimate the "
         "diabetes progression score and classify the risk level.")
st.caption("⚠️ Educational demo — not a clinical tool.")

st.markdown("---")

# ---------- Input section ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Patient Info")
    age    = st.slider("Age (years)",                  19, 80, 48, 1)
    sex    = st.selectbox("Sex",                       ["Female", "Male"])
    bmi    = st.slider("BMI (kg/m²)",                  15.0, 50.0, 26.4, 0.1)
    bp     = st.slider("Blood Pressure (mm Hg)",       60, 130, 95, 1)
    chol   = st.slider("Total Cholesterol (mg/dL)",    100, 320, 189, 1)

with col2:
    st.markdown("### 🧪 Lab Results")
    ldl    = st.slider("LDL Cholesterol (mg/dL)",      50, 220, 115, 1)
    hdl    = st.slider("HDL Cholesterol (mg/dL)",      20, 110, 50, 1)
    ratio  = st.slider("Chol/HDL Ratio",               2.0, 8.0, 4.1, 0.1)
    trig   = st.slider("Triglycerides (log scale)",    3.0, 6.5, 4.5, 0.1)
    sugar  = st.slider("Blood Sugar (mg/dL)",          60, 140, 91, 1)

st.markdown("---")

# ---------- Predict ----------
if st.button("🔮 Predict"):

    # Convert real values → model-scaled values
    scaled = np.array([[
        to_model_scale(age,   "age"),
        SEX_SCALED[sex],
        to_model_scale(bmi,   "bmi"),
        to_model_scale(bp,    "blood_pressure"),
        to_model_scale(chol,  "total_cholesterol"),
        to_model_scale(ldl,   "ldl"),
        to_model_scale(hdl,   "hdl"),
        to_model_scale(ratio, "chol_hdl_ratio"),
        to_model_scale(trig,  "triglycerides"),
        to_model_scale(sugar, "blood_sugar"),
    ]])

    score = reg_model.predict(scaled)[0]
    risk  = clf_model.predict(scaled)[0]
    proba = clf_model.predict_proba(scaled)[0]

    st.markdown("### 📊 Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Progression Score", f"{score:.1f}",
              help=f"Dataset median: {median_score:.1f}")
    m2.metric("High Risk",  f"{proba[1]*100:.1f}%")
    m3.metric("Low Risk",   f"{proba[0]*100:.1f}%")

    st.markdown("#### Risk Classification")
    if risk == 1:
        st.error(f"🚨 **HIGH RISK** — Score likely above median ({median_score:.0f}).")
    else:
        st.success(f"✅ **LOW RISK** — Score likely below median ({median_score:.0f}).")

    st.markdown("#### Confidence")
    st.progress(float(proba[1]), text=f"High Risk: {proba[1]*100:.1f}%")
    st.progress(float(proba[0]), text=f"Low Risk:  {proba[0]*100:.1f}%")

st.markdown("---")
st.caption("Model: Linear Regression + Random Forest · Dataset: sklearn load_diabetes()")
