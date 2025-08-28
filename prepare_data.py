import pandas as pd, os
from sklearn.model_selection import train_test_split
from utils import clean_text, LABELS

def main():
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/sample_news.csv")
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["label"].isin(LABELS)].copy()

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, random_state=42, stratify=test_df["label"]
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Saved: data/processed/train.csv, val.csv, test.csv")

if __name__ == "__main__":
    main()
