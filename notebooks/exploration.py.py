import pandas as pd

# Load dataset
data = pd.read_csv("../data/student_performance.csv")

# Display first 5 rows
print("FIRST 5 STUDENTS:")
print(data.head())

# Display dataset information
print("\nDATASET INFORMATION:")
print(data.info())

# Display statistics
print("\nSTATISTICAL SUMMARY:")
print(data.describe())