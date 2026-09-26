from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import split_features_target
from src.features.split_data import split_data


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"


df = load_data(DATA_PATH)

df_clean = preprocess_data(df)

X, y = split_features_target(df_clean)

X_train, X_test, y_train, y_test = split_data(X, y)


print("Training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())