import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("data/student_performance.csv")


# Select input features
X = data[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Assignment_Score",
        "Sleep_Hours"
    ]
]


# Select target
y = data["Final_Score"]


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)


print("===== STUDENT PERFORMANCE PREDICTOR =====")

print("\nModel: Linear Regression")

print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))


# Show actual vs predicted values
print("\nActual vs Predicted:")

for actual, predicted in zip(y_test, predictions):
    print(
        "Actual:",
        round(actual, 2),
        "| Predicted:",
        round(predicted, 2)
    )