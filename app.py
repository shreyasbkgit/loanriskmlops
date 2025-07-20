from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

# Define FastAPI app
app = FastAPI()

# Corrected request schema
class InputData(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Gender_Male: bool
    Married_Yes: bool
    Dependents_1: bool
    Dependents_2: bool
    Dependents_3_plus: bool = Field(..., alias="Dependents_3+")
    Education_Not_Graduate: bool = Field(..., alias="Education_Not Graduate")
    Self_Employed_Yes: bool
    Property_Area_Semiurban: bool
    Property_Area_Urban: bool

    class Config:
        allow_population_by_field_name = True  # allows internal alias usage

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict(by_alias=True)])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
