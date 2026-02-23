import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

streamlit_app = FastAPI(title="Cali Wildfire Risk API")

MODEL_DIR = "model"

try:
    model = joblib.load(os.path.join(MODEL_DIR, "wildfire_rfc.pkl"))
    county_encoder = joblib.load(os.path.join(MODEL_DIR, "county_encoder.pkl"))
    county_list = joblib.load(os.path.join(MODEL_DIR, "county_list.pkl"))
    county_fips = joblib.load(os.path.join(MODEL_DIR, "county_fips.pkl"))

except FileNotFoundError:
    raise RuntimeError("Model files not found. Please run testrandomforest.py first.")


class PredictRequest(BaseModel):
    month: int = Field(
        ..., ge=1, le=12, description="Month as integer (1=January, 12=December)"
    )


class CountyRisk(BaseModel):
    county: str
    fips: str
    risk: str


class PredictResponse(BaseModel):
    month: int
    predictions: list[CountyRisk]


@streamlit_app.get("/")
def root():
    return {"message": "Welcome to the Cali Wildfire Risk API!"}


@streamlit_app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    month = request.month

    predictions = []
    for county in county_list:
        try:
            county_encoded = county_encoder.transform([county])[0]

        except Exception:
            raise HTTPException(
                status_code=500, detail=f"Could not encode county: {county}"
            )

        X = np.array([[month, county_encoded]])
        risk = model.predict(X)[0]

        predictions.append(
            CountyRisk(county=county, fips=county_fips[county], risk=risk)
        )

    return PredictResponse(month=month, predictions=predictions)


@streamlit_app.get("/counties")
def list_counties():
    return {"counties": county_list}
