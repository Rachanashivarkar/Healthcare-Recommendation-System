import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Personalized Medicine Recommendation",
    page_icon="💊",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #d62828;
        color: white;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #d62828;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.markdown(
    '<p class="title">💊 Personalized Medicine Recommendation System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Based Heart Disease & Medicine Recommendation</p>',
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- INPUT SECTION ----------------
st.subheader("🩺 Enter Patient Health Details")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 1, 120, 30)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    sex = 1 if gender == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )

    cp_mapping = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }

    cp = cp_mapping[cp]

    trestbps = st.number_input(
        "Resting Blood Pressure",
        value=120
    )

    chol = st.number_input(
        "Cholesterol Level",
        value=200
    )

    fbs_option = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["Yes", "No"]
    )

    fbs = 1 if fbs_option == "Yes" else 0

with col2:

    restecg = st.selectbox(
        "Rest ECG",
        [0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        value=150
    )

    exang_option = st.selectbox(
        "Exercise Induced Angina",
        ["Yes", "No"]
    )

    exang = 1 if exang_option == "Yes" else 0

    oldpeak = st.number_input(
        "Oldpeak",
        value=1.0
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thal",
        [0, 1, 2, 3]
    )

# ---------------- PREDICTION ----------------
if st.button("Get Personalized Recommendation"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)[0][1]

    st.markdown("---")

    # ---------------- HIGH RISK ----------------
    if prediction[0] == 1:

        st.error("⚠️ Heart Disease Risk Detected")

        st.metric(
            "Risk Probability",
            f"{round(probability * 100, 2)}%"
        )

        st.subheader("💊 Personalized Medicine Recommendation")

        medicines = []

        if chol > 240:
            medicines.append("Statins - Helps control cholesterol")

        if trestbps > 140:
            medicines.append("ACE Inhibitors - Helps control blood pressure")

        if thalach < 100:
            medicines.append("Beta Blockers - Helps regulate heart rate")

        if oldpeak > 2:
            medicines.append("Aspirin - Helps reduce clot risk")

        if len(medicines) == 0:
            medicines.append("General cardiac consultation recommended")

        for med in medicines:
            st.write(f"• {med}")

        st.subheader("🥗 Lifestyle Recommendations")

        st.write("• Avoid oily and junk food")
        st.write("• Reduce salt intake")
        st.write("• Daily walking for 30 minutes")
        st.write("• Reduce stress and anxiety")
        st.write("• Regular health checkups")
        st.write("• Maintain healthy sleep schedule")

    # ---------------- LOW RISK ----------------
    else:

        st.success("✅ Low Risk of Heart Disease")

        st.metric(
            "Risk Probability",
            f"{round(probability * 100, 2)}%"
        )

        st.subheader("🌿 Preventive Health Recommendations")

        st.write("• Maintain balanced diet")
        st.write("• Exercise regularly")
        st.write("• Drink enough water")
        st.write("• Avoid smoking and alcohol")
        st.write("• Continue regular health checkups")

# ---------------- DISCLAIMER ----------------
st.markdown("---")

st.warning(
    "Disclaimer: This AI system provides educational recommendations only and is not a substitute for professional medical advice."
)