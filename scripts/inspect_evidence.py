import pandas as pd
from pathlib import Path

p = Path(r"C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\data\processed\mirassist_backend_features.parquet")
df = pd.read_parquet(p)

print("shape:", df.shape)

print("\ncolumns:")
for c in df.columns:
    print(c)

print("\ndtypes:")
print(df.dtypes.to_string())

print("\nhead:")
print(df.head(3).to_string())