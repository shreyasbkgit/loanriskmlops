# test_predict.py
import requests
import json

sample = {
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 100,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Gender_Male": True,
    "Married_Yes": True,
    "Dependents_1": False,
    "Dependents_2": False,
    "Dependents_3+": False,
    "Education_Not Graduate": False,
    "Self_Employed_Yes": False,
    "Property_Area_Semiurban": False,
    "Property_Area_Urban": True
}

res = requests.post("http://localhost:8000/predict", json=sample)
print("Prediction:", res.json())
