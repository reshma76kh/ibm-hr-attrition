import joblib
from pathlib import Path


def save_model(model, preprocessor, model_path, preprocessor_path):

    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print("Model saved successfully!")
    print(f"Model: {model_path}")
    print(f"Preprocessor: {preprocessor_path}")