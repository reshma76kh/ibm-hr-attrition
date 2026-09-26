from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import split_features_target
from src.features.split_data import split_data
from src.features.preprocessing_pipeline import create_preprocessing_pipeline
from src.features.apply_smote import apply_smote


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"


df = load_data(DATA_PATH)

df_clean = preprocess_data(df)

X, y = split_features_target(df_clean)

X_train, X_test, y_train, y_test = split_data(X, y)

preprocessor = create_preprocessing_pipeline(X_train)

X_train_transformed = preprocessor.fit_transform(X_train)

X_test_transformed = preprocessor.transform(X_test)

X_train_resampled, y_train_resampled = apply_smote(
    X_train_transformed,
    y_train
)


print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

print("\nTraining shape before SMOTE:")
print(X_train_transformed.shape)

print("\nTraining shape after SMOTE:")
print(X_train_resampled.shape)

print("\nTest shape:")
print(X_test_transformed.shape)