from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import split_features_target
from src.features.split_data import split_data
from src.features.preprocessing_pipeline import create_preprocessing_pipeline


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"


df = load_data(DATA_PATH)

df_clean = preprocess_data(df)

X, y = split_features_target(df_clean)

X_train, X_test, y_train, y_test = split_data(X, y)

preprocessor = create_preprocessing_pipeline(X_train)

X_train_transformed = preprocessor.fit_transform(X_train)

X_test_transformed = preprocessor.transform(X_test)


print("Original X_train shape:", X_train.shape)

print("Transformed X_train shape:", X_train_transformed.shape)

print("\nOriginal X_test shape:", X_test.shape)

print("Transformed X_test shape:", X_test_transformed.shape)