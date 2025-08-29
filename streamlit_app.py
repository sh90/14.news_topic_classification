import json
import os, joblib, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from utils import LABELS, json_safe_parse, backoff_retry
from typing import List, Tuple

load_dotenv()
st.set_page_config(page_title="News Topic Classifier", page_icon="📰")

st.title("📰 News Topic Classifier — Evolution Demo")
st.caption("TF-IDF → Embeddings → LLM (gpt-4o-mini) ")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Approach", ["TF-IDF + LR", "Embeddings + LR", "LLM Zero-shot", "LLM Few-shot"])
    action = st.radio("Action", ["Single Prediction", "Evaluate Test Set"])


# ---------------- Base predictors ----------------
def tfidf_predict(texts: List[str]) -> List[str]:
    vec = joblib.load("artifacts/tfidf_vectorizer.joblib")
    clf = joblib.load("artifacts/tfidf_logreg.joblib")
    return clf.predict(vec.transform(texts)).tolist()

def embed_predict(texts: List[str]) -> List[str]:
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    X = np.array([e.embedding for e in resp.data], dtype=np.float32)
    clf = joblib.load("artifacts/embed_logreg.joblib")
    return clf.predict(X).tolist()

def llm_predict(texts: List[str], few: bool=False) -> List[str]:
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

# ---------------- Enhanced helpers (probas + analysis) ----------------
def tfidf_predict_with_proba(texts: List[str]) -> Tuple[List[str], np.ndarray, List[str]]:
    vec = joblib.load("artifacts/tfidf_vectorizer.joblib")
    clf = joblib.load("artifacts/tfidf_logreg.joblib")
    X = vec.transform(texts)
    preds = clf.predict(X).tolist()
    proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else np.full((len(texts), 1), np.nan)
    return preds, proba, clf.classes_.tolist()

def embed_predict_with_proba(texts: List[str]) -> Tuple[List[str], np.ndarray, List[str]]:
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    X = np.array([e.embedding for e in resp.data], dtype=np.float32)
    clf = joblib.load("artifacts/embed_logreg.joblib")
    preds = clf.predict(X).tolist()
    proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else np.full((len(texts), 1), np.nan)
    return preds, proba, clf.classes_.tolist()

def llm_predict_with_proba(texts: List[str], few: bool=False) -> Tuple[List[str], np.ndarray, List[str]]:
    preds = llm_predict(texts, few=few)
    proba = np.full((len(texts), 1), np.nan)  # LLM has no calibrated proba
    return preds, proba, LABELS

def top1_conf_and_margin(proba: np.ndarray, classes: List[str]):
    """Return top1 prob, margin (top1-top2), and top1 labels per row. NaN-safe."""
    if proba.ndim != 2 or proba.shape[1] < 2:
        n = proba.shape[0] if proba.ndim == 2 else len(proba)
        return np.full(n, np.nan), np.full(n, np.nan), []
    idx_sorted = np.argsort(-proba, axis=1)
    top1_idx = idx_sorted[:, 0]; top2_idx = idx_sorted[:, 1]
    top1_p = proba[np.arange(len(proba)), top1_idx]
    top2_p = proba[np.arange(len(proba)), top2_idx]
    margin = top1_p - top2_p
    top1_lab = [classes[i] for i in top1_idx]
    return top1_p, margin, top1_lab

# --- TF-IDF distribution & per-class contribution for a single text ---
def tfidf_vector_breakdown(text: str, top_n: int = 20):
    vec = joblib.load("artifacts/tfidf_vectorizer.joblib")
    clf = joblib.load("artifacts/tfidf_logreg.joblib")

    X = vec.transform([text])                  # csr_matrix (1, V)
    row = X.toarray().ravel()                  # (V,)
    features = vec.get_feature_names_out()

    nz = np.flatnonzero(row)
    terms = features[nz]
    tfidf_vals = row[nz]
    tfidf_df = (
        pd.DataFrame({"term": terms, "tfidf": tfidf_vals})
        .sort_values("tfidf", ascending=False)
        .head(top_n)
    )

    contrib = {}
    if hasattr(clf, "coef_"):
        for i, cls in enumerate(clf.classes_):
            # class score (logit) — use ravel()[0] (or .item()) instead of .A1
            logit_arr = (X @ clf.coef_[i].reshape(-1, 1)) + clf.intercept_[i]
            logit = np.asarray(logit_arr).ravel()[0]     # <<< fix here

            weights = clf.coef_[i][nz]
            cont_df = (
                pd.DataFrame({
                    "term": terms,
                    "tfidf": tfidf_vals,
                    "weight": weights,
                    "contribution": weights * tfidf_vals
                })
                .sort_values("contribution", ascending=False)
                .head(top_n)
            )
            contrib[cls] = {"logit": float(logit), "top_terms": cont_df}

    return tfidf_df, contrib


# ---------------- UI ----------------
if action == "Single Prediction":
    txt = st.text_area(
        "Paste a news headline or short article:",
        height=120,
        placeholder="e.g., Central bank signals rate cuts amid slowing growth",
    )
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
    except Exception:
        st.error("Missing data/processed/test.csv. Run `python src/prepare_data.py` or `python src/ag_news_loader.py`.")
        st.stop()

    # -------- Compute once, persist in session --------
    if st.button("Run Evaluation"):
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        texts = test.text.tolist()
        y_true = test.label.tolist()

        if mode == "TF-IDF + LR":
            y_pred, proba, classes = tfidf_predict_with_proba(texts)
        elif mode == "Embeddings + LR":
            y_pred, proba, classes = embed_predict_with_proba(texts)
        elif mode == "LLM Zero-shot":
            y_pred, proba, classes = llm_predict_with_proba(texts, few=False)
        else:
            y_pred, proba, classes = llm_predict_with_proba(texts, few=True)

        acc = float(accuracy_score(y_true, y_pred))

        df = test.copy()
        df["pred"] = y_pred
        df["correct"] = df["label"] == df["pred"]

        top1_p, margin, _ = top1_conf_and_margin(proba, classes)
        if not np.all(np.isnan(top1_p)):
            df["confidence"] = top1_p
            df["margin"] = margin
        else:
            df["confidence"] = np.nan
            df["margin"] = np.nan

        from sklearn.metrics import classification_report, confusion_matrix
        st.session_state["eval_results"] = {
            "acc": acc,
            "classes": classes,
            "report": classification_report(df.label, df.pred, labels=classes, zero_division=0),
            "cm": confusion_matrix(df.label, df.pred, labels=classes).tolist(),  # list for session safety
            "df": df,
            "mode": mode,
        }

    # -------- Render from session (stable across widget changes) --------
    if "eval_results" in st.session_state:
        res = st.session_state["eval_results"]
        classes = res["classes"]
        df = res["df"]
        curr_mode = res.get("mode", mode)

        st.metric("Accuracy", f"{res['acc']:.3f}")

        st.markdown("**Classification report**")
        st.code(res["report"])

        st.markdown("**Confusion matrix**")
        cm_df = pd.DataFrame(np.array(res["cm"]), index=classes, columns=classes)
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("### 🔎 Misclassified examples")

        # Use a form so intermediate widget tweaks don't trigger resets
        with st.form(key="mis_filters"):
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                true_sel = st.multiselect("True label", classes, default=classes, key="true_sel")
            with c2:
                pred_sel = st.multiselect("Predicted label", classes, default=classes, key="pred_sel")
            with c3:
                keyword = st.text_input("Contains keyword (optional)", key="kw")

            colA, colB, colC = st.columns([1,1,1])
            with colA:
                k = st.slider("Max rows", 10, 500, 50, step=10, key="max_rows")
            with colB:
                sort_by = st.selectbox("Sort by", ["None", "confidence ↑", "confidence ↓", "margin ↑", "margin ↓"], key="sort_by")
            with colC:
                only_class = st.selectbox("Focus errors for class (optional)", ["—"] + classes, key="only_cls")

            submitted = st.form_submit_button("Apply filters")

        # Apply filters (state persists, no reset on widget interactions)
        mis = df[~df["correct"]].copy()
        mis = mis[mis["label"].isin(st.session_state["true_sel"]) & mis["pred"].isin(st.session_state["pred_sel"])]
        if st.session_state["kw"].strip():
            mis = mis[mis["text"].str.contains(st.session_state["kw"], case=False, na=False)]
        if st.session_state["only_cls"] != "—":
            mis = mis[mis["label"] == st.session_state["only_cls"]]

        sb = st.session_state["sort_by"]
        if sb != "None":
            if "confidence" in sb and "confidence" in mis.columns:
                mis = mis.sort_values("confidence", ascending=("↑" in sb))
            elif "margin" in sb and "margin" in mis.columns:
                mis = mis.sort_values("margin", ascending=("↑" in sb))

        st.write(f"Showing {min(st.session_state['max_rows'], len(mis))} of {len(mis)} misclassified rows")
        st.dataframe(
            mis[["text", "label", "pred", "confidence", "margin"]].head(st.session_state["max_rows"]),
            use_container_width=True
        )

        csv = mis.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download misclassifications as CSV",
            data=csv,
            file_name="misclassified.csv",
            mime="text/csv",
            key="dl_mis"
        )

        # ---------- TF-IDF term distribution for a selected misclassified row ----------
        if curr_mode == "TF-IDF + LR":
            st.markdown("#### 🧪 TF-IDF term distribution & class contributions (misclassified row)")
            if len(mis) == 0:
                st.info("No misclassified rows to inspect.")
            else:
                idx = st.number_input(
                    "Pick row index (from the misclassified table above)",
                    min_value=0,
                    max_value=max(len(mis)-1, 0),
                    value=0,
                    step=1,
                    key="tfidf_row_idx"
                )
                top_n_terms = st.slider("Top terms to show", 5, 30, 15, key="tfidf_topn")
                row_sel = mis.iloc[int(idx)]
                tfidf_df, contrib = tfidf_vector_breakdown(row_sel["text"], top_n=top_n_terms)

                st.markdown("**Top TF-IDF terms (this document):**")
                st.dataframe(tfidf_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    cls_true = row_sel["label"]
                    if cls_true in contrib:
                        st.markdown(f"**Contributions toward TRUE class: `{cls_true}`**  \nLogit: `{contrib[cls_true]['logit']:.3f}`")
                        st.dataframe(contrib[cls_true]["top_terms"][["term","tfidf","weight","contribution"]], use_container_width=True)
                with col2:
                    cls_pred = row_sel["pred"]
                    if cls_pred in contrib:
                        st.markdown(f"**Contributions toward PRED class: `{cls_pred}`**  \nLogit: `{contrib[cls_pred]['logit']:.3f}`")
                        st.dataframe(contrib[cls_pred]["top_terms"][["term","tfidf","weight","contribution"]], use_container_width=True)

                st.caption("Contribution = model weight × TF-IDF. Positive contributions push the score toward that class.")
    else:
        st.info("Click **Run Evaluation** to compute results.")
