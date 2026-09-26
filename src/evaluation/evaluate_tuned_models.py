from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import split_features_target
from src.features.split_data import split_data
from src.features.preprocessing_pipeline import create_preprocessing_pipeline
from src.features.apply_smote import apply_smote
from src.training.tune import tune_models
from src.evaluation.evaluate import evaluate_models


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"



# 1. LOAD DATA


df = load_data(DATA_PATH)


# 2. PREPROCESS DATA


df_clean = preprocess_data(df)



# 3. SEPARATE FEATURES AND TARGET


X, y = split_features_target(df_clean)



# 4. TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# 5. PREPROCESS FEATURES


preprocessor = create_preprocessing_pipeline(X_train)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)



# 6. APPLY SMOTE ONLY TO TRAINING DATA


X_train_resampled, y_train_resampled = apply_smote(
    X_train_transformed,
    y_train
)


# 7. HYPERPARAMETER TUNING

tuned_models = tune_models(
    X_train_resampled,
    y_train_resampled
)



# 8. EVALUATE TUNED MODELS ON TEST DATA

results = evaluate_models(
    tuned_models,
    X_test_transformed,
    y_test
)



# 9. DISPLAY RESULTS


print("\n")
print("=" * 80)
print("TUNED MODEL TEST SET RESULTS")
print("=" * 80)

print(results.to_string(index=False))