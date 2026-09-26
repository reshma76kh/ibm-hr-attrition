from src.data.load_data import load_data

DATA_PATH="data/raw/IBM-HR-Employee-Attrition.csv" \
""

df = load_data(DATA_PATH)

print("Dataset loaded successfully !")
print("Shape:",df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())