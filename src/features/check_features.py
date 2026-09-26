from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import split_features_target


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"


df = load_data(DATA_PATH)

df_clean = preprocess_data(df)

X, y = split_features_target(df_clean)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget:")
print(y.name)

print("\nX columns:")
print(X.columns.tolist())