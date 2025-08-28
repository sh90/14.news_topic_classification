import os, joblib, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from utils import LABELS, json_safe_parse, backoff_retry

load_dotenv()
st.set_page_config(page_title="News Topic Classifier", page_icon="📰")

st.title("📰 News Topic Classifier — Evolution Demo")
st.caption("TF‑IDF → Embeddings → LLM (gpt‑4o‑mini) | Python 3.12")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Approach", ["TF-IDF + LR", "Embeddings + LR", "LLM Zero-shot", "LLM Few-shot"])
    action = st.radio("Action", ["Single Prediction", "Evaluate Test Set"])
    st.markdown("---")
    st.markdown("**Artifacts** expected after running CLI scripts:")
    st.code("python src/prepare_data.py\npython src/train_tfidf.py\npython src/train_embed.py\npython src/llm_classify.py --mode zero\npython src/llm_classify.py --mode few")

# Helpers
def tfidf_predict(texts):
    vec = joblib.load("artifacts/tfidf_vectorizer.joblib")
    clf = joblib.load("artifacts/tfidf_logreg.joblib")
    return clf.predict(vec.transform(texts)).tolist()

def embed_predict(texts):
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    X = np.array([e.embedding for e in resp.data], dtype=np.float32)
    clf = joblib.load("artifacts/embed_logreg.joblib")
    return clf.predict(X).tolist()

def llm_predict(texts, few=False):
    client = OpenAI()
    SYS = (
        "You are a strict news topic classifier. "
        f"Allowed labels: {', '.join(LABELS)}. "
        "Return ONLY valid JSON: {\"label\": <one of allowed labels>}"
    )
    FEWSHOTS = [
        ("Govt passes new data bill in upper house", "Politics"),
        ("Club seals comeback victory in league opener", "Sports"),
        ("Chipmaker posts record quarterly revenue", "Business"),
        ("Researchers unveil faster algorithm for LLM training", "Tech"),
        ("Actor confirms sequel at film festival", "Entertainment"),
    ]
    preds=[]
    for t in texts:
        msgs=[{"role":"system","content":SYS}]
        if few:
            for x,y in FEWSHOTS:
                msgs.append({"role":"user","content":x})
                msgs.append({"role":"assistant","content":json.dumps({"label": y})})
        msgs.append({"role":"user","content":t})
        def call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs,
                temperature=0,
                response_format={"type":"json_object"},
            )
        resp = backoff_retry(call)
        label = json_safe_parse(resp.choices[0].message.content).get("label","Politics")
        preds.append(label if label in LABELS else "Politics")
    return preds

if action == "Single Prediction":
    txt = st.text_area("Paste a news headline or short article:", height=120, placeholder="e.g., Central bank signals rate cuts amid slowing growth")
    if st.button("Classify"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            if mode == "TF-IDF + LR":
                pred = tfidf_predict([txt])[0]
            elif mode == "Embeddings + LR":
                pred = embed_predict([txt])[0]
            elif mode == "LLM Zero-shot":
                pred = llm_predict([txt], few=False)[0]
            else:
                pred = llm_predict([txt], few=True)[0]
            st.success(f"Predicted label: **{pred}**")

else:
    st.subheader("Evaluate on Test Set")
    try:
        test = pd.read_csv("data/processed/test.csv")
    except Exception as e:
        st.error("Missing data/processed/test.csv. Run `python src/prepare_data.py` or `python src/ag_news_loader.py`.")
        st.stop()

    if st.button("Run Evaluation"):
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        texts = test.text.tolist()
        y = test.label.tolist()

        if mode == "TF-IDF + LR":
            yhat = tfidf_predict(texts)
        elif mode == "Embeddings + LR":
            yhat = embed_predict(texts)
        elif mode == "LLM Zero-shot":
            yhat = llm_predict(texts, few=False)
        else:
            yhat = llm_predict(texts, few=True)

        acc = accuracy_score(y, yhat)
        st.metric("Accuracy", f"{acc:.3f}")
        st.markdown("**Classification report**")
        st.code(classification_report(y, yhat, labels=LABELS))
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, yhat, labels=LABELS)
        st.dataframe(pd.DataFrame(cm, index=LABELS, columns=LABELS))
