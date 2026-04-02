import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv("data/raw/train.csv")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("✅ Data preprocessing done")

if __name__ == "__main__":
    preprocess()