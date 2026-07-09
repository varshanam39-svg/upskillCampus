
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

grade_names = {
    0: "Grade C",
    1: "Grade B",
    2: "Grade A"
}

print("==============")
print("Student Marks Predictor")
print("==============")

study_hours = float(input("Study Hours (1-10): "))
attendance = float(input("Attendance (%): "))
assignment = float(input("Assignment Marks: "))
internal = float(input("Internal Marks: "))

sample = np.array([[study_hours, attendance, assignment, internal]])

prediction = model.predict(sample)[0]

print("\nPredicted Result")
print("----------------")
print("Grade :", grade_names[prediction])