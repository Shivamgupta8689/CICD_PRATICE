import os
import pandas as pd
from sklearn.datasets import make_classification

def generate_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    columns = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    return df

def save_data(df):
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/train.csv", index=False)
    print("✅ Data saved to data/raw/train.csv")

if __name__ == "__main__":
    df = generate_data()
    save_data(df)