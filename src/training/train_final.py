from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import split_features_target
from src.features.split_data import split_data
from src.features.preprocessing_pipeline import create_preprocessing_pipeline
from src.features.apply_smote import apply_smote
from src.training.tune import tune_models
from src.training.save_model import save_model


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"


# 1. Load data
df = load_data(DATA_PATH)

# 2. Preprocess
df_clean = preprocess_data(df)

# 3. Features and target
X, y = split_features_target(df_clean)

# 4. Train-test split
X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Preprocessing
preprocessor = create_preprocessing_pipeline(X_train)

X_train_transformed = preprocessor.fit_transform(X_train)

# 6. SMOTE
X_train_resampled, y_train_resampled = apply_smote(
    X_train_transformed,
    y_train
)

# 7. Tune models
tuned_models = tune_models(
    X_train_resampled,
    y_train_resampled
)

# 8. Select final model
final_model = tuned_models["AdaBoost"].best_estimator_

# 9. Save
save_model(
    model=final_model,
    preprocessor=preprocessor,
    model_path="models/final_model.pkl",
    preprocessor_path="models/preprocessor.pkl"
)

print("\nFinal model: AdaBoost")
print("Production artifacts created successfully!")