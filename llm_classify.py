import pandas as pd, json, os, argparse
from openai import OpenAI
from tqdm import tqdm
from utils import LABELS, json_safe_parse, backoff_retry

CHAT_MODEL = "gpt-4o-mini"
import os
from dotenv import load_dotenv

load_dotenv()

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

def classify_batch(client: OpenAI, texts, fewshot=False):
    results=[]
    for t in tqdm(texts, desc="LLM classify"):
        msgs = [{"role":"system","content":SYS}]
        if fewshot:
            for x,y in FEWSHOTS:
                msgs.append({"role":"user","content":x})
                msgs.append({"role":"assistant","content":json.dumps({"label": y})})
        msgs.append({"role":"user","content":t})

        def call():
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=msgs,
                temperature=0,
                response_format={"type":"json_object"},
            )
            return resp
        resp = backoff_retry(call)
        content = resp.choices[0].message.content
        obj = json_safe_parse(content)
        label = obj.get("label", "")
        if label not in LABELS: label = "Politics"
        results.append(label)
    return results

def main(mode="zero"):
    client = OpenAI()
    test = pd.read_csv("data/processed/test.csv")
    preds = classify_batch(client, test.text.tolist(), fewshot=(mode=="few"))
    out = test.copy(); out["pred"] = preds
    os.makedirs("artifacts", exist_ok=True)
    out.to_csv(f"artifacts/llm_preds_{mode}.csv", index=False)
    print(f"Saved artifacts/llm_preds_{mode}.csv")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero","few"], default="zero")
    args=ap.parse_args()
    main(args.mode)
