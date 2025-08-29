# News Article Topic Classifier — 

**What you'll build:** an end‑to‑end topic classifier for news articles with three approaches:
1) **TF‑IDF + Logistic Regression** (fast, interpretable baseline)  
2) **OpenAI Embeddings + Logistic Regression** (semantic generalization)  
3) **LLM (gpt‑4o‑mini) zero/few‑shot** (no training; prompt‑based JSON output)

**Skills gained:** text cleaning, train/val/test split, TF‑IDF vs. embeddings, model training, LLM prompting, metrics (accuracy, confusion matrix, per‑class F1), and production trade‑offs.

---

## Project layout

```
news_topic_classifier/
├─ .env
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

```
python3.12 for virtual env
pip install -r requirements.txt
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

## Notes

- Python 3.12, minimal deps.  
- Uses `.env` for `OPENAI_API_KEY`.  
- Small included dataset to keep the demo quick. You can swap in AG News or your proprietary corpus.
