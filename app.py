from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI()
model = joblib.load("models/model.pkl")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total", "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_request_duration_seconds", "Latency of prediction requests"
)

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
    start_time = time.time()
    REQUEST_COUNT.inc()

    df = pd.DataFrame([features.dict()])
    df.columns = df.columns.str.replace("_plus", "+")
    df.columns = df.columns.str.replace("_Not_Graduate", "_Not Graduate")

    prediction = model.predict(df)

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"Loan Approval Prediction": bool(prediction[0])}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
