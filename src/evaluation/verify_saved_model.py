import joblib
import pandas as pd

MODEL_PATH = "models/final_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


# Load saved artifacts
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Model loaded successfully!")
print("Preprocessor loaded successfully!")

print("\nModel type:")
print(type(model))

print("\nPreprocessor type:")
print(type(preprocessor))