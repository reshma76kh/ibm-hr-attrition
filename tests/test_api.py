#test api
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)



# Test 1: Home endpoint


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "IBM HR Attrition API is running"
    }


# Test 2: Prediction endpoint


def test_prediction():

    employee = {
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
    }

    response = client.post(
        "/predict",
        json=employee
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "attrition_probability" in result

    assert result["prediction"] in ["Yes", "No"]

    assert 0 <= result["attrition_probability"] <= 1



# Test 3: Invalid input


def test_invalid_input():

    employee = {
        "Age": 30
    }

    response = client.post(
        "/predict",
        json=employee
    )

    assert response.status_code == 422