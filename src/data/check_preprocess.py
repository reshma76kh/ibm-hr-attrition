from src.data.load_data import load_data
from src.data.preprocess import preprocess_data


DATA_PATH = "data/raw/IBM-HR-Employee-Attrition.csv"


df = load_data(DATA_PATH)

print("Before preprocessing:")
print("Shape:", df.shape)

df_clean = preprocess_data(df)

print("\nAfter preprocessing:")
print("Shape:", df_clean.shape)

print("\nRemaining columns:")
print(df_clean.columns.tolist())

print("\nDuplicate rows:")
print(df_clean.duplicated().sum())