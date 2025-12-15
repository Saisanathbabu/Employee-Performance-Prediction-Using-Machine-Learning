from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model
model = pickle.load(open("gwp.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("Home.html")

# Predict route (GET shows form, POST processes input)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Collect form data
        quarter = int(request.form["quarter"])
        department = int(request.form["department"])
        day = int(request.form["day"])
        team = int(request.form["team"])
        targeted_productivity = float(request.form["targeted_productivity"])
        smv = float(request.form["smv"])
        over_time = int(request.form["over_time"])
        incentive = int(request.form["incentive"])
        idle_time = float(request.form["idle_time"])
        idle_men = int(request.form["idle_men"])
        no_of_style_change = int(request.form["no_of_style_change"])
        no_of_workers = float(request.form["no_of_workers"])
        month = int(request.form["month"])

        # Create feature array for prediction
        features = np.array([[quarter, department, day, team,
                              targeted_productivity, smv, over_time,
                              incentive, idle_time, idle_men,
                              no_of_style_change, no_of_workers, month]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert prediction into readable text
        if prediction < 0.5:
            prediction_text = "The employee is averagely productive."
        elif prediction < 0.75:
            prediction_text = "The employee is medium productive."
        else:
            prediction_text = "The employee is highly productive."

        return render_template("Submit.html", prediction_text=prediction_text)

    return render_template("Predict.html")

# About route
@app.route("/About")
def about():
    return render_template("About.html")

if __name__ == "__main__":
    app.run(debug=True)