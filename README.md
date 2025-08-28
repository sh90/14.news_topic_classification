# News Article Topic Classifier — 2‑Hour Demo (Python 3.12, gpt-4o-mini)

**What you'll build:** an end‑to‑end topic classifier for short news snippets with three approaches:
1) **TF‑IDF + Logistic Regression** (fast, interpretable baseline)  
2) **OpenAI Embeddings + Logistic Regression** (semantic generalization)  
3) **LLM (gpt‑4o‑mini) zero/few‑shot** (no training; prompt‑based JSON output)

**Skills gained:** text cleaning, train/val/test split, TF‑IDF vs. embeddings, model training, LLM prompting, metrics (accuracy, confusion matrix, per‑class F1), and production trade‑offs.

---

## Why this matters in the real world

- **Cold start**: With limited labeled data, LLM zero/few‑shot classification gives you usable accuracy *today*.  
- **Scale & cost**: As traffic grows, switch to **Embeddings + LR** or **TF‑IDF + LR** for low latency and predictable cost.  
- **Auditability**: TF‑IDF features provide **interpretability** for compliance and stakeholder trust.  
- **Multilingual robustness**: Embeddings capture semantics across paraphrases and languages better than n‑gram TF‑IDF.  
- **Human‑in‑the‑loop**: Use confidence thresholds; route uncertain cases to humans or an LLM for disambiguation.  
- **Typical use cases**: newsroom content routing, inbox/tag triage, CRM ticket bucketing, social listening, brand safety filters.

---

## Project layout

```
news_topic_classifier/
├─ .env.example
├─ requirements.txt
├─ data/
│  └─ sample_news.csv
└─ src/
   ├─ utils.py
   ├─ prepare_data.py
   ├─ train_tfidf.py
   ├─ train_embed.py
   ├─ llm_classify.py
   └─ compare.py
```

---

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # paste your real key in .env
```

### 1) Prepare data
```bash
python src/prepare_data.py
```

### 2) Baseline: TF‑IDF + LR
```bash
python src/train_tfidf.py
```

### 3) Embeddings: OpenAI `text-embedding-3-small` + LR
```bash
python src/train_embed.py
```

### 4) LLM topic classification with gpt‑4o‑mini
```bash
python src/llm_classify.py --mode zero   # zero-shot
python src/llm_classify.py --mode few    # few-shot
```

### 5) Compare all approaches side-by-side
```bash
python src/compare.py
```

---

## Teaching cues (for a 2‑hour session)

- **Evolution story**: Show how accuracy/behavior changes from TF‑IDF → Embeddings → LLM.  
- **Interpretability**: Print “Top features per class” from the TF‑IDF model.  
- **Confusions**: Inspect confusion matrices (e.g., **Business vs Tech** headlines).  
- **Ops tips**: Batch embeddings, cache vectors, add backoff/retries, measure cost and latency.  
- **Guardrails**: Force LLM to return strict JSON with `response_format={"type":"json_object"}` and validate label membership.

---

## Notes

- Python 3.12, minimal deps.  
- Uses `.env` for `OPENAI_API_KEY`.  
- Small included dataset to keep the demo quick. You can swap in AG News or your proprietary corpus later.
