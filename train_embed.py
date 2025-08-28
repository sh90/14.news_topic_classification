import os, pandas as pd, numpy as np, joblib
from openai import OpenAI
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import LABELS
import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"

def embed_texts(client: OpenAI, texts):
    BATCH=1000
    vecs=[]
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        chunk = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vecs.extend([e.embedding for e in resp.data])
    return np.array(vecs, dtype=np.float32)

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    train = pd.read_csv("data/processed/train.csv")
    val   = pd.read_csv("data/processed/val.csv")
    test  = pd.read_csv("data/processed/test.csv")

    Xtr = embed_texts(client, train.text.tolist())
    Xv  = embed_texts(client, val.text.tolist())
    Xte = embed_texts(client, test.text.tolist())

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, train.label)

    for split_name, X, y in [("VAL", Xv, val.label), ("TEST", Xte, test.label)]:
        pred = clf.predict(X)
        print(f"\n[{split_name}] accuracy:", accuracy_score(y, pred))
        print(classification_report(y, pred, labels=LABELS))
        print("Confusion:\n", confusion_matrix(y, pred, labels=LABELS))

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(clf, "artifacts/embed_logreg.joblib")
    np.save("artifacts/val_embed.npy", Xv)
    np.save("artifacts/test_embed.npy", Xte)
    print("\nSaved artifacts to artifacts/")

if __name__ == "__main__":
    main()
