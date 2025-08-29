import os, pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from utils import clean_text
# https://autonlp.ai/datasets/ag-news
# Map AG News (World, Sports, Business, Sci/Tech) -> our labels
MAP = {
    0: "Politics",   # World -> Politics (approximation for demo)
    1: "Sports",
    2: "Business",
    3: "Tech",
}

def main(samples_per_class=400, seed=42):
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    ds = load_dataset("ag_news")
    df = pd.DataFrame(ds["train"])[["text","label"]]
    df["label"] = df["label"].map(MAP)
    df["text"] = df["text"].astype(str).map(clean_text)

    # Balance classes
    frames = []
    for lab in sorted(df["label"].unique()):
        frames.append(df[df["label"]==lab].sample(n=samples_per_class, random_state=seed))
    bal = pd.concat(frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    bal.to_csv("data/ag_news_sample.csv", index=False)
    print(f"Saved balanced sample with {len(bal)} rows to data/ag_news_sample.csv")

    train_df, test_df = train_test_split(bal, test_size=0.3, random_state=seed, stratify=bal["label"])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed, stratify=test_df["label"])

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Overwrote data/processed/{train,val,test}.csv using AG News.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_per_class", type=int, default=400)
    args = ap.parse_args()
    main(args.samples_per_class)
