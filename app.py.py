import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("data/student_performance.csv")


# -----------------------------
# Train Model
# -----------------------------

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Sleep_Hours"
]

X = data[features]
y = data["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


# -----------------------------
# Application Interface
# -----------------------------

st.title("🎓 Student Performance Predictor")

st.write(
    "Enter the student's academic and lifestyle information "
    "to predict their final performance."
)

st.divider()


# -----------------------------
# User Inputs
# -----------------------------

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

previous_score = st.number_input(
    "Previous Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

assignment_score = st.number_input(
    "Assignment Score",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=1.0
)

sleep_hours = st.number_input(
    "Sleep Hours per Day",
    min_value=0.0,
    max_value=15.0,
    value=7.0,
    step=0.5
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Performance"):

    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Score": [previous_score],
        "Assignment_Score": [assignment_score],
        "Sleep_Hours": [sleep_hours]
    })

    prediction = model.predict(new_student)[0]

    prediction = max(0, min(100, prediction))


    # Performance category

    if prediction >= 90:
        performance = "Excellent 🌟"

    elif prediction >= 75:
        performance = "Good 👍"

    elif prediction >= 50:
        performance = "Average 📚"

    else:
        performance = "Needs Improvement 📈"


    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Final Score",
        f"{prediction:.2f}/100"
    )

    st.success(
        f"Performance Category: {performance}"
    )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Student Performance Predictor | Machine Learning Project"
)