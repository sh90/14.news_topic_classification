import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# More realistic sample news snippets with overlapping terms
docs = [
    "The government announced new economic reforms for the business sector",
    "Stock markets react to government tax policy changes",
    "The local football team won the championship match in dramatic style",
    "Fans celebrate as the football team secures league victory",
    "Tech companies raise funding to expand AI business operations",
    "New AI breakthrough announced by leading tech companies",
]

# Initialize TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(docs)
features = vectorizer.get_feature_names_out()

def plot_top_words(doc_index, top_n=5):
    """Plot top TF-IDF words for a given document index"""
    row = X[doc_index].toarray().flatten()
    top_idx = np.argsort(row)[-top_n:][::-1]  # top N indices
    top_features = features[top_idx]
    top_scores = row[top_idx]

    plt.figure(figsize=(6,4))
    plt.barh(top_features[::-1], top_scores[::-1])  # reverse for top-to-bottom
    plt.xlabel("TF-IDF Score")
    plt.title(f"Top {top_n} words for Document {doc_index+1}")
    plt.tight_layout()
    plt.show()

# Show documents and plots
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}: {doc}")
    plot_top_words(i, top_n=5)
