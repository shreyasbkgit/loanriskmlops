# --- app.py ---
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model.pkl")


class LoanFeatures(BaseModel):
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Gender_Male: bool
    Married_Yes: bool
    Dependents_1: bool
    Dependents_2: bool
    Dependents_3_plus: bool
    Education_Not_Graduate: bool
    Self_Employed_Yes: bool
    Property_Area_Semiurban: bool
    Property_Area_Urban: bool


@app.post("/predict")
def predict(features: LoanFeatures):
    df = pd.DataFrame([features.dict()])
    df.columns = df.columns.str.replace("_plus", "+")
    df.columns = df.columns.str.replace("_Not_Graduate", "_Not Graduate")
    prediction = model.predict(df)
    return {"Loan Approval Prediction": bool(prediction[0])}
