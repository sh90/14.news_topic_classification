import pandas as pd, joblib, os, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import LABELS
from sklearn.utils.multiclass import unique_labels
def main():
    train = pd.read_csv("data/processed/train.csv")
    val   = pd.read_csv("data/processed/val.csv")
    test  = pd.read_csv("data/processed/test.csv")

    vec = TfidfVectorizer(
        ngram_range=(1,2), min_df=1, max_df=0.95, strip_accents="unicode"
    )
    Xtr = vec.fit_transform(train.text)
    Xv  = vec.transform(val.text)
    Xte = vec.transform(test.text)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, train.label)

    def eval_split(name, y_true, y_pred):
        present = list(unique_labels(y_true, y_pred))
        print(f"\n[{name}] accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, labels=present, zero_division=0))
        print("Confusion:\n", confusion_matrix(y_true, y_pred, labels=present))

    for split_name, X, y in [("VAL", Xv, val.label), ("TEST", Xte, test.label)]:
        pred = clf.predict(X)
        eval_split(split_name, y, pred)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(vec, "artifacts/tfidf_vectorizer.joblib")
    joblib.dump(clf, "artifacts/tfidf_logreg.joblib")
    print("\nSaved artifacts to artifacts/")

    # -------- Top features only for classes the model actually learned --------
    feature_names = np.array(vec.get_feature_names_out())
    coef = clf.coef_
    learned_labels = list(clf.classes_)  # e.g., ['Business','Politics','Sports','Tech']
    for i, cls in enumerate(learned_labels):
        top_idx = np.argsort(coef[i])[-8:][::-1]
        print(f"\nTop features for {cls}: {feature_names[top_idx]}")

if __name__ == "__main__":
    main()
