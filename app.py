from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

grade_names = {
    0: "Grade C",
    1: "Grade B",
    2: "Grade A"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    assignment = float(request.form["assignment"])
    internal = float(request.form["internal"])

    features = np.array([[study_hours, attendance, assignment, internal]])

    prediction = model.predict(features)[0]

    grade = grade_names[prediction]

    return render_template(
        "result.html",
        grade=grade,
        study_hours=study_hours,
        attendance=attendance,
        assignment=assignment,
        internal=internal
    )


if __name__ == "__main__":
    app.run(debug=True)