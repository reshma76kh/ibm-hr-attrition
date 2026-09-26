from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel



# Paths


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"



# Load model and preprocessor


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# FastAPI application


app = FastAPI(
    title="IBM HR Employee Attrition API",
    description="API for predicting employee attrition",
    version="1.0.0"
)



# Input schema


class EmployeeInput(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int



# Health check


@app.get("/")
def home():
    return {
        "message": "IBM HR Attrition API is running"
    }



# Prediction endpoint


@app.post("/predict")
def predict(employee: EmployeeInput):

    # Convert input to DataFrame
    employee_data = pd.DataFrame([employee.model_dump()])

    # Apply saved preprocessing
    employee_transformed = preprocessor.transform(employee_data)

    # Make prediction
    prediction = model.predict(employee_transformed)[0]

    # Get probability
    probability = model.predict_proba(employee_transformed)[0][1]

    # Convert numeric prediction to original label
    prediction_label = {
        0: "No",
        1: "Yes"
    }[int(prediction)]

    return {
        "prediction": prediction_label,
        "attrition_probability": round(float(probability), 4)
    }