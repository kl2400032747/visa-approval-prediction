from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/visa_random_forest.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        employer_name = request.form["employer_name"]
        soc_name = request.form["soc_name"]
        job_title = request.form["job_title"]
        full_time_position = request.form["full_time_position"]
        prevailing_wage = float(request.form["prevailing_wage"])
        year = int(request.form["year"])
        worksite = request.form["worksite"]

        # Feature engineering
        wage_log = np.log1p(prevailing_wage)
        years_from_2011 = year - 2011

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "EMPLOYER_NAME": employer_name.upper().strip(),
            "SOC_NAME": soc_name.upper().strip(),
            "JOB_TITLE": job_title.upper().strip(),
            "FULL_TIME_POSITION": full_time_position.upper().strip(),
            "PREVAILING_WAGE": prevailing_wage,
            "YEAR": year,
            "WORKSITE": worksite.upper().strip(),
            "WAGE_LOG": wage_log,
            "YEARS_FROM_2011": years_from_2011
        }])

        # Prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        class_0_probability = probabilities[0] * 100
        class_1_probability = probabilities[1] * 100

        if prediction == 1:
            result = "Positive Case Outcome"
        else:
            result = "Negative Case Outcome"

        return render_template(
            "index.html",
            result=result,
            prediction=prediction,
            class_0_probability=round(class_0_probability, 2),
            class_1_probability=round(class_1_probability, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)