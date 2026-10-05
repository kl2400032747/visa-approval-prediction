import joblib
import pandas as pd


MODEL_PATH = "model/visa_random_forest.pkl"

print("=" * 60)
print("H1B VISA MODEL - TEST PREDICTION")
print("=" * 60)

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# Sample application
sample = pd.DataFrame([{
    "EMPLOYER_NAME": "MICROSOFT CORPORATION",
    "SOC_NAME": "SOFTWARE DEVELOPERS",
    "JOB_TITLE": "SOFTWARE DEVELOPER",
    "FULL_TIME_POSITION": "Y",
    "PREVAILING_WAGE": 100000,
    "YEAR": 2016,
    "WORKSITE": "SEATTLE, WASHINGTON",
    "WAGE_LOG": 11.512925,
    "YEARS_FROM_2011": 5
}])

print("\nSample application:")
print(sample)

print("\nMaking prediction...")

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("\nPrediction result:")

if prediction == 1:
    print("Result: APPROVED / POSITIVE OUTCOME")
else:
    print("Result: NOT APPROVED / NEGATIVE OUTCOME")

print(f"Probability of class 0: {probability[0]:.2%}")
print(f"Probability of class 1: {probability[1]:.2%}")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)