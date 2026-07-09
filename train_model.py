
import pandas as pd
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# Generate Synthetic Dataset
# ----------------------------

data = []

for i in range(500):

    study_hours = random.randint(1, 10)

    attendance = random.randint(50, 100)

    assignment = random.randint(40, 100)

    internal = random.randint(35, 100)

    total = (
        study_hours * 5
        + attendance * 0.3
        + assignment * 0.3
        + internal * 0.4
    )

    if total >= 90:
        grade = 2      # A

    elif total >= 70:
        grade = 1      # B

    else:
        grade = 0      # C

    data.append([
        study_hours,
        attendance,
        assignment,
        internal,
        grade
    ])

columns = [
    "StudyHours",
    "Attendance",
    "Assignment",
    "InternalMarks",
    "Grade"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("student_marks.csv", index=False)

print("Dataset Created Successfully")

# ----------------------------
# Train Model
# ----------------------------

X = df.drop("Grade", axis=1)

y = df["Grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nModel Accuracy :", round(accuracy * 100, 2), "%")

pickle.dump(model, open("model.pkl", "wb"))

print("model.pkl Saved Successfully")