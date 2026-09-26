from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

from src.features.build_features import split_features_target
from src.features.split_data import split_data
from src.features.preprocessing_pipeline import create_preprocessing_pipeline
from src.features.apply_smote import apply_smote

from src.training.train import train_models


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"



# 1. Load data


df = load_data(DATA_PATH)

print("Data loaded successfully.")
print("Original shape:", df.shape)



# 2. Preprocess data


df_clean = preprocess_data(df)

print("\nAfter preprocessing:")
print(df_clean.shape)



# 3. Separate features and target


X, y = split_features_target(df_clean)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)



# 4. Train/Test Split


X_train, X_test, y_train, y_test = split_data(
    X,
    y
)

print("\nTrain/Test Split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


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

print("\nAfter preprocessing pipeline:")
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

print("\nTarget distribution after SMOTE:")
print(y_train_resampled.value_counts())



# 7. Train all models


trained_models = train_models(
    X_train_resampled,
    y_train_resampled
)



# 8. Display trained models


print("\n" + "=" * 50)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 50)

for name in trained_models:
    print(f"✓ {name}")