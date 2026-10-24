"""v0.55 - RAG sur les documents du projet (README, DOCUMENTATION, ROADMAP), Kaggle T4.

1. Decoupe les trois fichiers Markdown en passages.
2. Compare deux retrievers sur des questions en francais dont on connait le passage source :
     supervise_en    : notre MiniLM entraine sur MS MARCO (anglais uniquement)
     multilingue_e5  : intfloat/multilingual-e5-small, modele multilingue du commerce
3. Fait repondre Mistral avec et sans le contexte retrouve.

Les documents viennent du jeu de donnees Kaggle prive pamperslegame/mistral-optimisation-docs,
notre retriever des sorties du script retrieval.py.
Controle local des questions, sans GPU :  RAG_CHECK_ONLY=1 DOCS_DIR=<dossier> python rag_docs.py
"""
import glob
import json
import os
import sys

DOCS = ["README.md", "DOCUMENTATION.md", "ROADMAP.md"]
MAX_CHARS = 1200
TOP_K = 3


def find(pattern):
    # les chemins sous /kaggle/input changent selon la facon dont la source est attachee
    hits = glob.glob(f"{os.environ.get('DOCS_DIR', '/kaggle/input')}/**/{pattern}", recursive=True)
    assert hits, f"introuvable : {pattern}"
    return hits[0]


def chunk(name, text):
    # un passage = un titre Markdown + les paragraphes qui suivent, coupe a MAX_CHARS
    chunks, title, buf, in_code = [], name, [], False

    def flush():
        if buf:
            chunks.append({"source": name, "title": title, "text": title + "\n\n" + "\n\n".join(buf)})
            buf.clear()

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        first, _, rest = para.partition("\n")
        if first.startswith("#") and not in_code:      # dans un bloc de code, '#' est un commentaire
            flush()
            title, para = first.lstrip("# ").strip(), rest.strip()
        in_code ^= para.count("```") % 2 == 1
        if para:
            if sum(map(len, buf)) + len(para) > MAX_CHARS:
                flush()
            buf.append(para)
    flush()
    return chunks


corpus = [c for name in DOCS for c in chunk(name, open(find(name), encoding="utf-8").read())]
questions = json.load(open(find("questions.json"), encoding="utf-8"))
for q in questions:
    q["relevant"] = {i for i, c in enumerate(corpus) if any(g in c["text"] for g in q["gold"])}
    assert q["relevant"], f"aucun passage ne contient la source attendue : {q['question']}"
print(f"{len(corpus)} passages, {len(questions)} questions | hasard pur : Recall@{TOP_K} = {TOP_K / len(corpus):.1%}")
if os.environ.get("RAG_CHECK_ONLY"):
    sys.exit()

import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46.1"], check=True)

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

LLM_ID = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
WORK = "/kaggle/working/v055_rag_docs"
os.makedirs(WORK, exist_ok=True)
DEVICE = "cuda"

# (modele, prefixe requete, prefixe passage) : e5 a ete entraine avec ces prefixes et en a besoin
RETRIEVERS = {
    "supervise_en": (os.path.dirname(find("supervise/config.json")), "", ""),
    "multilingue_e5": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
}


@torch.no_grad()
def embed(model, tok, texts):
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        mask = enc["attention_mask"].unsqueeze(-1)
        out.append(F.normalize((model(**enc).last_hidden_state * mask).sum(1) / mask.sum(1), dim=-1))
    return torch.cat(out)


rows, ranked = [], {}
for name, (model_id, q_prefix, p_prefix) in RETRIEVERS.items():
    tok, model = AutoTokenizer.from_pretrained(model_id), AutoModel.from_pretrained(model_id).to(DEVICE).eval()
    scores = (embed(model, tok, [q_prefix + q["question"] for q in questions])
              @ embed(model, tok, [p_prefix + c["text"] for c in corpus]).T)
    # 150 passages : un produit matriciel suffit, pas besoin d'index FAISS a cette taille
    ranked[name] = scores.topk(TOP_K, dim=-1).indices.tolist()
    row = {"retriever": name, "n_questions": len(questions), "corpus_size": len(corpus)}
    for k in (1, TOP_K):
        row[f"recall@{k}"] = sum(bool(q["relevant"] & set(r[:k])) for q, r in zip(questions, ranked[name])) / len(questions)
    rows.append(row)
    del model
pd.DataFrame(rows).to_csv(f"{WORK}/rag_docs.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- Generation ----------
llm_tok = AutoTokenizer.from_pretrained(LLM_ID)
llm = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map={"": 0})


def generate(prompt):
    enc = llm_tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=150, do_sample=False, pad_token_id=llm_tok.eos_token_id)
    return llm_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


examples = []
for q, top in zip(questions, ranked["multilingue_e5"]):
    context = "\n\n---\n\n".join(corpus[i]["text"] for i in top)
    row = {
        "question": q["question"],
        "bon_passage_retrouve": bool(q["relevant"] & set(top)),
        "sources": [f"{corpus[i]['source']} > {corpus[i]['title']}" for i in top],
        "sans_contexte": generate(f"[INST] Réponds brièvement en français.\n\nQuestion : {q['question']} [/INST]"),
        "avec_contexte": generate("[INST] Réponds brièvement en français, uniquement à partir du contexte ci-dessous. "
                                  "Si le contexte ne contient pas la réponse, dis-le.\n\n"
                                  f"Contexte :\n{context}\n\nQuestion : {q['question']} [/INST]"),
    }
    examples.append(row)
    print(json.dumps(row, indent=1, ensure_ascii=False))
with open(f"{WORK}/rag_docs_examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=1, ensure_ascii=False)
