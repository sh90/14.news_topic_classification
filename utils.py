import re, json, time
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
load_dotenv()

LABELS = ["Politics", "Sports", "Business", "Tech", "Entertainment"]
LABEL_SET = set(LABELS)
LABEL_TO_ID = {c:i for i,c in enumerate(LABELS)}
ID_TO_LABEL = {i:c for c,i in LABEL_TO_ID.items()}

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

def json_safe_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        start = s.find('{'); end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            try: return json.loads(s[start:end+1])
            except Exception: return {}
        return {}

def backoff_retry(call, max_tries=5, base=1.5):
    for i in range(max_tries):
        try:
            return call()
        except Exception as e:
            if i == max_tries-1: raise
            time.sleep(base**i)
