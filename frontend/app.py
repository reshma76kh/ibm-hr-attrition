import requests
import streamlit as st



# Page configuration


st.set_page_config(
    page_title="IBM HR Attrition Predictor",
    page_icon="👥",
    layout="wide"
)



# Title


st.title("👥 IBM HR Employee Attrition Predictor")

st.write(
    "Enter employee information to predict the likelihood "
    "of employee attrition."
)


# FastAPI endpoint
API_URL = "http://api:8000/predict"



# Employee Information

st.header("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    education = st.number_input(
        "Education",
        min_value=1,
        max_value=5,
        value=3
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

with col2:
    department = st.selectbox(
        "Department",
        [
            "Research & Development",
            "Sales",
            "Human Resources"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_level = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2
    )

    job_involvement = st.number_input(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3
    )

    job_satisfaction = st.number_input(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

with col3:
    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=100,
        value=10
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=0,
        value=800
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=0,
        value=60
    )



# Work Information


st.header("Work Information")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=0,
        value=5000
    )

    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=0,
        max_value=100,
        value=12
    )

    performance_rating = st.number_input(
        "Performance Rating",
        min_value=1,
        max_value=4,
        value=3
    )

with col2:
    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=60,
        value=8
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=60,
        value=4
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=60,
        value=2
    )

with col3:
    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=60,
        value=1
    )

    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=60,
        value=2
    )

    num_companies_worked = st.number_input(
        "Number Of Companies Worked",
        min_value=0,
        max_value=50,
        value=2
    )

    training_times_last_year = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=20,
        value=3
    )



# Satisfaction Information


st.header("Employee Satisfaction")

col1, col2, col3 = st.columns(3)

with col1:
    environment_satisfaction = st.number_input(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

with col2:
    relationship_satisfaction = st.number_input(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

with col3:
    work_life_balance = st.number_input(
        "Work Life Balance",
        min_value=1,
        max_value=4,
        value=3
    )


# Other Information


st.header("Other Information")

stock_option_level = st.number_input(
    "Stock Option Level",
    min_value=0,
    max_value=3,
    value=0
)



# Prediction


st.divider()

if st.button(
    "🔮 Predict Attrition",
    use_container_width=True
):

    employee = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager
    }

    try:

        response = requests.post(
            API_URL,
            json=employee
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            probability = result["attrition_probability"]

            st.subheader("Prediction Result")

            if prediction == "Yes":
                st.error(" Employee may leave the company")
            else:
                st.success("Employee is predicted to stay")

            st.metric(
                "Attrition Probability",
                f"{probability:.2%}"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the FastAPI server is running."
        )