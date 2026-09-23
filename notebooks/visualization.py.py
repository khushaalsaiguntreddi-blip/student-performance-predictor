import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/student_performance.csv")

# 1. Study Hours vs Final Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Study_Hours",
    y="Final_Score",
    data=data
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.tight_layout()
plt.show()


# 2. Attendance vs Final Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Attendance",
    y="Final_Score",
    data=data
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.tight_layout()
plt.show()


# 3. Previous Score vs Final Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Previous_Score",
    y="Final_Score",
    data=data
)

plt.title("Previous Score vs Final Score")
plt.xlabel("Previous Score")
plt.ylabel("Final Score")
plt.tight_layout()
plt.show()


# 4. Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Student Performance Correlation")
plt.tight_layout()
plt.show()