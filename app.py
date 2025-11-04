
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('C:/Users/VAISHNAVINA/Desktop/XGBOOST/heart_disease_model.pkl')

st.title("❤️ Heart Disease Prediction App")
st.write("Answer the questions below to estimate your risk of heart disease.")

# --- Inputs with HELP buttons ---

st.subheader("Personal Information")

age = st.number_input("Age (years)", min_value=1, max_value=120)
with st.expander("ℹ️ What does 'Age' mean?"):
    st.write("""
    Enter your current age in years. Age is a major factor — risk usually increases with age.
    """)

sex = st.selectbox("Sex", ("Male", "Female"))
with st.expander("ℹ️ What does 'Sex' mean?"):
    st.write("""
    Biological sex affects baseline risk.  
    **Male (1):** Higher average risk  
    **Female (0):** Lower average risk
    """)
sex = 1 if sex == "Male" else 0

st.subheader("Chest Pain Details")
cp = st.selectbox(
    "Chest Pain Type",
    ("Typical Angina (1)", "Atypical Angina (2)", "Non-anginal Pain (3)", "Asymptomatic (4)")
)
with st.expander("ℹ️ What are the chest pain types?"):
    st.write("""
    - **Typical angina (1):** Classic chest discomfort caused by exertion or stress.  
    - **Atypical angina (2):** Chest pain feels different from typical heart pain.  
    - **Non-anginal (3):** Pain not related to the heart.  
    - **Asymptomatic (4):** No chest pain felt, but heart disease may still exist.
    """)
cp = int(cp.split("(")[1][0])

st.subheader("Blood Pressure & Cholesterol")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
with st.expander("ℹ️ What is resting blood pressure?"):
    st.write("""
    The blood pressure measured when you are resting (not exercising).  
    **Normal:** Around 120/80 mm Hg  
    **High:** Over 130/90 mm Hg may indicate hypertension.
    """)

chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)
with st.expander("ℹ️ What is serum cholesterol?"):
    st.write("""
    Measures the total cholesterol level in your blood.  
    **Normal:** Below 200 mg/dL  
    **Borderline high:** 200–239 mg/dL  
    **High:** 240 mg/dL or more.
    """)

st.subheader("Blood Sugar & ECG Results")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ("Yes", "No"))
with st.expander("ℹ️ What is fasting blood sugar?"):
    st.write("""
    Indicates whether your blood sugar (after fasting) exceeds 120 mg/dL.  
    - **Yes (1):** Possible diabetes or insulin resistance  
    - **No (0):** Normal range.
    """)
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox(
    "Resting ECG Results",
    ("Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)")
)
with st.expander("ℹ️ What is a resting ECG?"):
    st.write("""
    An **ECG (electrocardiogram)** measures your heart’s electrical activity while resting.  
    - **Normal (0):** No abnormalities  
    - **ST-T abnormality (1):** May indicate heart strain  
    - **LV hypertrophy (2):** Thickened heart wall from high blood pressure.
    """)
restecg = int(restecg.split("(")[1][0])

st.subheader("Exercise & Stress Test")

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)
with st.expander("ℹ️ What is maximum heart rate?"):
    st.write("""
    Highest heart rate achieved during exercise.  
    Approximate formula: **220 − your age.**
    """)

exang = st.selectbox("Exercise Induced Angina?", ("Yes", "No"))
with st.expander("ℹ️ What is exercise induced angina?"):
    st.write("""
    Chest pain caused by physical activity.  
    - **Yes (1):** Chest pain occurs with exercise.  
    - **No (0):** No pain during exertion.
    """)
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
with st.expander("ℹ️ What does 'Oldpeak' mean?"):
    st.write("""
    Difference between your ECG reading during exercise and at rest.  
    - **Higher values:** Indicate more stress on the heart.
    """)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ("Upsloping (1)", "Flat (2)", "Downsloping (3)")
)
with st.expander("ℹ️ What does slope mean?"):
    st.write("""
    Describes how your ECG line behaves during peak exercise:  
    - **Upsloping:** Usually healthy response  
    - **Flat:** May suggest limited heart response  
    - **Downsloping:** Often abnormal, linked to heart issues.
    """)
slope = int(slope.split("(")[1][0])

st.subheader("Coronary & Thalassemia Details")

ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3)
with st.expander("ℹ️ What does this mean?"):
    st.write("""
    Number of major blood vessels visible during an angiography (0–3).  
    - **0:** All clear  
    - **3:** Multiple blockages detected.
    """)

thal = st.selectbox(
    "Thalassemia Type",
    ("Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)")
)
with st.expander("ℹ️ What is thalassemia type?"):
    st.write("""
    Thalassemia in this dataset refers to **blood flow condition**, not genetic disease:  
    - **Normal (1):** Normal blood flow  
    - **Fixed defect (2):** Permanent reduced blood flow  
    - **Reversible defect (3):** Blood flow improves with rest.
    """)
thal = int(thal.split("(")[1][0])

# --- Prediction ---
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts a **high risk** of heart disease.")
    else:
        st.success("✅ The model predicts a **low risk** of heart disease.")
