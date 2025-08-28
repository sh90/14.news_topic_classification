import pandas as pd, joblib, numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import LABELS

def eval_preds(y_true, y_pred, tag):
    print(f"\n== {tag} ==")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=LABELS))
    print("Confusion:\n", confusion_matrix(y_true, y_pred, labels=LABELS))

def main():
    test = pd.read_csv("data/processed/test.csv")

    # TF-IDF
    vec = joblib.load("artifacts/tfidf_vectorizer.joblib")
    clf_t = joblib.load("artifacts/tfidf_logreg.joblib")
    y_tfidf = clf_t.predict(vec.transform(test.text))
    eval_preds(test.label, y_tfidf, "TF-IDF + LR")

    # Embeddings
    clf_e = joblib.load("artifacts/embed_logreg.joblib")
    Xte = np.load("artifacts/test_embed.npy")
    y_embed = clf_e.predict(Xte)
    eval_preds(test.label, y_embed, "Embeddings + LR")

    # LLM Zero-shot
    try:
        llm_zero = pd.read_csv("artifacts/llm_preds_zero.csv")
        eval_preds(test.label, llm_zero.pred, "LLM Zero-shot (gpt-4o-mini)")
    except Exception:
        print("\nNo zero-shot file found. Run: python src/llm_classify.py --mode zero")

    # LLM Few-shot
    try:
        llm_few = pd.read_csv("artifacts/llm_preds_few.csv")
        eval_preds(test.label, llm_few.pred, "LLM Few-shot (gpt-4o-mini)")
    except Exception:
        print("No few-shot file found. Run: python src/llm_classify.py --mode few")

if __name__ == "__main__":
    main()
