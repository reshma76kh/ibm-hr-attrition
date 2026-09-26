\# IBM HR Employee Attrition Prediction



An end-to-end Machine Learning project for predicting employee attrition using the IBM HR Analytics Employee Attrition \& Performance dataset.



The project covers the complete workflow from data preprocessing and model training to API deployment, Streamlit frontend, testing, and Docker containerization.



\## Project Overview



Employee attrition is an important HR problem. This project builds a machine learning system that predicts whether an employee is likely to leave the organization.



\### Workflow



Data

↓

Data Preprocessing

↓

Feature Engineering

↓

Train/Test Split

↓

Encoding + Scaling

↓

SMOTE

↓

Multiple ML Models

↓

Hyperparameter Tuning

↓

Model Evaluation

↓

Final Model

↓

FastAPI

↓

Streamlit

↓

Docker



\## Dataset



IBM HR Analytics Employee Attrition \& Performance dataset.



\- Rows: 1,470

\- Original columns: 35

\- Target: `Attrition`

\- Classes: `Yes` / `No`



Constant and identifier columns removed:



\- EmployeeCount

\- StandardHours

\- Over18

\- EmployeeNumber



\## Machine Learning



The project experiments with multiple classification algorithms:



\- Support Vector Classifier

\- K-Nearest Neighbors

\- Decision Tree

\- Logistic Regression

\- Random Forest

\- AdaBoost

\- Bagging

\- Extra Trees

\- Gradient Boosting

\- XGBoost



Because employee attrition is an imbalanced classification problem, SMOTE is applied to the training data.



Hyperparameter tuning is performed using `RandomizedSearchCV` with 5-fold cross-validation and F1 scoring.



\## Final Model



AdaBoost was selected for the production pipeline based on the evaluation results for the project's attrition-detection objective.



The final model and preprocessing pipeline are saved using Joblib:



```text

models/

├── final\_model.pkl

└── preprocessor.pkl

