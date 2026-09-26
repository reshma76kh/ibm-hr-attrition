import joblib
import pandas as pd

MODEL_PATH = "models/final_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

# Load model and preprocessor
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Example employee
employee = pd.DataFrame([{
    "Age": 30,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 60,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 5000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 12,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 4,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2
}])

# Transform input
employee_transformed = preprocessor.transform(employee)

# Prediction
prediction = model.predict(employee_transformed)[0]

# Probability
probability = model.predict_proba(employee_transformed)[0][1]

# Convert prediction
prediction_label = {
    0: "No",
    1: "Yes"
}[prediction]

print("Employee Attrition Prediction")
print("=" * 40)

print("Prediction:", prediction_label)
print(f"Attrition Probability: {probability:.2%}")