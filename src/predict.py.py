import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv("data/student_performance.csv")


# Input features
X = data[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Assignment_Score",
        "Sleep_Hours"
    ]
]

# Target
y = data["Final_Score"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


print("===================================")
print("   STUDENT PERFORMANCE PREDICTOR")
print("===================================")

print("\nEnter the student's details:")

study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
previous_score = float(input("Previous exam score: "))
assignment_score = float(input("Assignment score: "))
sleep_hours = float(input("Sleep hours per day: "))


# Create new student data
new_student = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Previous_Score": [previous_score],
    "Assignment_Score": [assignment_score],
    "Sleep_Hours": [sleep_hours]
})


# Make prediction
prediction = model.predict(new_student)[0]


# Keep prediction between 0 and 100
prediction = max(0, min(100, prediction))


# Determine performance category
if prediction >= 90:
    performance = "Excellent"
elif prediction >= 75:
    performance = "Good"
elif prediction >= 50:
    performance = "Average"
else:
    performance = "Needs Improvement"


print("\n===================================")
print("        PREDICTION RESULT")
print("===================================")

print(f"Predicted Final Score: {prediction:.2f}")
print(f"Performance Category: {performance}")

print("===================================")