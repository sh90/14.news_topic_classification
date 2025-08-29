"""
Tokenizes the documents into words.

Calculates TF-IDF values:

TF (Term Frequency): how often a word appears in a document.

IDF (Inverse Document Frequency): how rare a word is across all documents.

The product gives higher weight to words that are important in one document but not common across all.

"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents (like mini news snippets)
docs = [
    "The government passed a new law today",
    "The local team won their football match",
    "A new startup raises funding in the tech sector",
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents into a TF-IDF matrix
X = vectorizer.fit_transform(docs)

# Get feature names (unique words)
features = vectorizer.get_feature_names_out()

print("Features (words):")
print(features)

print("\nTF-IDF Matrix (rows=documents, cols=words):")
print(X.toarray())
