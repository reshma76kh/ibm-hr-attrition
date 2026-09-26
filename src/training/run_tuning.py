from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

from src.features.build_features import split_features_target
from src.features.split_data import split_data
from src.features.preprocessing_pipeline import create_preprocessing_pipeline
from src.features.apply_smote import apply_smote

from src.training.tune import tune_models


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"



# 1. Load data


df = load_data(DATA_PATH)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)



# 2. Preprocess


df_clean = preprocess_data(df)

print("\nAfter preprocessing:")
print(df_clean.shape)



# 3. Split X and y


X, y = split_features_target(df_clean)

print("\nFeatures:", X.shape)
print("Target:", y.shape)



# 4. Train/Test Split


X_train, X_test, y_train, y_test = split_data(
    X,
    y
)

print("\nTrain/Test Split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)



# 5. Encoding + Scaling


preprocessor = create_preprocessing_pipeline(
    X_train
)

X_train_transformed = preprocessor.fit_transform(
    X_train
)

X_test_transformed = preprocessor.transform(
    X_test
)

print("\nAfter preprocessing:")
print("X_train:", X_train_transformed.shape)
print("X_test:", X_test_transformed.shape)



# 6. SMOTE


X_train_resampled, y_train_resampled = apply_smote(
    X_train_transformed,
    y_train
)

print("\nAfter SMOTE:")
print("X_train:", X_train_resampled.shape)
print("y_train:", y_train_resampled.shape)



# 7. Hyperparameter Tuning


tuned_models = tune_models(
    X_train_resampled,
    y_train_resampled
)



# 8. Display Results


print("\n")
print("=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)

for name, search in tuned_models.items():

    print(f"\n{name}")

    print("Best Parameters:")
    print(search.best_params_)

    print("Best CV F1:")
    print(search.best_score_)