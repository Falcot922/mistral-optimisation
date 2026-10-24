"""v0.55 - Embeddings contrastifs, retrieval et RAG (Kaggle T4).

On part d'un MiniLM "brut" (jamais entraine a comparer des phrases) et on mesure ce que
l'apprentissage contrastif lui apporte pour retrouver le bon passage MS MARCO d'une requete :

  minilm_brut   : le modele de depart, sans entrainement
  simcse        : + contrastif NON supervise (meme passage encode deux fois, dropout different)
  supervise     : + contrastif supervise (requete, passage pertinent)
  reference     : all-MiniLM-L6-v2, le meme modele entraine sur 1 milliard de paires

Puis le meilleur de nos modeles sert de retriever a Mistral (RAG).
Lancement : kaggle kernels push -p kaggle/v0.55-retrieval
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu", "bitsandbytes>=0.46.1"], check=True)

import json
import os

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ---------- Configuration ----------
MINI = False                # True : verifie en quelques minutes que tout s'enchaine
RAW_ID = "nreimers/MiniLM-L6-H384-uncased"
REF_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_ID = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
N_TRAIN, N_EVAL_QUERIES = 20000, 1000
BATCH, EPOCHS, LR = 64, 1, 3e-5
TEMPERATURE = 0.05
MAX_LEN = 128
RUN_RAG = True
N_RAG = 5
SEED = 42
WORK = "/kaggle/working/v055_retrieval"
if MINI:
    N_TRAIN, N_EVAL_QUERIES, N_RAG = 512, 100, 2

os.makedirs(WORK, exist_ok=True)
torch.manual_seed(SEED)
DEVICE = "cuda"


# ---------- Donnees ----------
def stream(split, n):
    # streaming : MS MARCO pese plusieurs Go, on ne lit que les n premieres requetes qui ont un passage pertinent
    out = []
    for ex in load_dataset("microsoft/ms_marco", "v2.1", split=split, streaming=True):
        if sum(ex["passages"]["is_selected"]) >= 1:
            out.append(ex)
            if len(out) == n:
                break
    return out


train = stream("train", N_TRAIN)
train_queries = [ex["query"] for ex in train]
train_passages = [ex["passages"]["passage_text"][ex["passages"]["is_selected"].index(1)] for ex in train]

# Evaluation : le corpus est l'ensemble des passages proposes pour les requetes de validation (~10 par requete).
# Trouver le bon passage parmi tous ceux-la est la tache de retrieval.
val = stream("validation", N_EVAL_QUERIES)
eval_queries, corpus, relevant = [], [], []
for ex in val:
    start = len(corpus)
    corpus.extend(ex["passages"]["passage_text"])
    eval_queries.append(ex["query"])
    relevant.append({start + i for i, s in enumerate(ex["passages"]["is_selected"]) if s == 1})
print(f"train : {len(train)} paires | eval : {len(eval_queries)} requetes, corpus de {len(corpus)} passages")
print(f"hasard pur : Recall@5 = {5 / len(corpus):.2%}")


# ---------- Embeddings et perte contrastive ----------
def embed(model, tok, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    # moyenne des tokens reels (pas du padding), puis norme 1 : le produit scalaire devient un cosinus
    return F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)


def info_nce(a, b, temperature=TEMPERATURE):
    # a[i] et b[i] forment la paire positive ; toutes les autres lignes du batch servent de negatifs
    sim = a @ b.T / temperature
    return F.cross_entropy(sim, torch.arange(len(a), device=a.device))


# controle : des paires parfaitement alignees et des negatifs orthogonaux donnent une perte nulle
assert info_nce(torch.eye(4), torch.eye(4)) < 1e-3
assert info_nce(torch.eye(4), torch.eye(4).roll(1, 0)) > 5


def train_contrastive(mode):
    tok = AutoTokenizer.from_pretrained(RAW_ID)
    model = AutoModel.from_pretrained(RAW_ID).to(DEVICE)
    model.train()            # indispensable : c'est le dropout qui cree les deux vues en SimCSE
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    step, running = 0, 0.0
    for _ in range(EPOCHS):
        perm = rng.permutation(len(train))
        for b in range(0, len(perm) - BATCH + 1, BATCH):      # batchs complets : meme nombre de negatifs partout
            idx = perm[b:b + BATCH]
            passages = [train_passages[i] for i in idx]
            if mode == "simcse":
                a, pos = embed(model, tok, passages), embed(model, tok, passages)
            else:
                a, pos = embed(model, tok, [train_queries[i] for i in idx]), embed(model, tok, passages)
            loss = info_nce(a, pos)
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            running += loss.item()
            if step % 10 == 0:
                print(f"   [{mode}] step {step} | loss {running / 10:.4f}")
                with open(f"{WORK}/loss.jsonl", "a") as f:
                    f.write(json.dumps({"mode": mode, "step": step, "loss": running / 10}) + "\n")
                running = 0.0
    model.save_pretrained(f"{WORK}/{mode}")
    tok.save_pretrained(f"{WORK}/{mode}")
    return model.eval(), tok


# ---------- Index FAISS et Recall@K ----------
@torch.no_grad()
def encode_all(model, tok, texts, batch=256):
    return np.concatenate([embed(model, tok, texts[i:i + batch]).cpu().numpy()
                           for i in range(0, len(texts), batch)]).astype("float32")


results, retrievers = [], {}


def evaluate(name, model, tok):
    # IndexFlatIP : recherche exacte par produit scalaire (= cosinus, les vecteurs sont normalises)
    index = faiss.IndexFlatIP(model.config.hidden_size)
    index.add(encode_all(model, tok, corpus))
    _, found = index.search(encode_all(model, tok, eval_queries), 10)
    row = {"model": name, "n_queries": len(eval_queries), "corpus_size": len(corpus)}
    for k in (1, 5, 10):
        row[f"recall@{k}"] = float(np.mean([bool(rel & set(f[:k])) for rel, f in zip(relevant, found)]))
    results.append(row)
    retrievers[name] = (model, tok, index)
    pd.DataFrame(results).to_csv(f"{WORK}/retrieval.csv", index=False)
    print(f"[eval] {name:12s} R@1 {row['recall@1']:.1%} | R@5 {row['recall@5']:.1%} | R@10 {row['recall@10']:.1%}")


evaluate("minilm_brut", AutoModel.from_pretrained(RAW_ID).to(DEVICE).eval(), AutoTokenizer.from_pretrained(RAW_ID))
evaluate("simcse", *train_contrastive("simcse"))
evaluate("supervise", *train_contrastive("supervise"))
evaluate("reference", AutoModel.from_pretrained(REF_ID).to(DEVICE).eval(), AutoTokenizer.from_pretrained(REF_ID))

df = pd.DataFrame(results).set_index("model")
print(df.round(3).to_string())
labels = {"minilm_brut": "MiniLM, untrained", "simcse": "+ SimCSE", "supervise": "+ supervised", "reference": "all-MiniLM-L6-v2"}
ax = df[["recall@1", "recall@5", "recall@10"]].rename(index=labels).plot.bar(figsize=(9, 4.5), rot=0, title="MS MARCO retrieval")
ax.set_xlabel("")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{WORK}/retrieval.png", dpi=150)


# ---------- RAG : notre retriever fournit le contexte a Mistral ----------
def generate(llm, llm_tok, prompt):
    enc = llm_tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=100, do_sample=False, pad_token_id=llm_tok.eos_token_id)
    return llm_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if RUN_RAG:
    model, tok, index = retrievers["supervise"]
    llm_tok = AutoTokenizer.from_pretrained(LLM_ID)
    llm = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map={"": 0})
    examples = []
    for ex in val[:N_RAG]:
        _, found = index.search(encode_all(model, tok, [ex["query"]]), 3)
        context = "\n\n".join(corpus[i] for i in found[0])
        row = {
            "question": ex["query"],
            "reference": ex["answers"],
            "sans_contexte": generate(llm, llm_tok, f"[INST] Answer briefly.\n\nQuestion: {ex['query']} [/INST]"),
            "avec_contexte": generate(llm, llm_tok, "[INST] Answer briefly, using only the context below.\n\n"
                                                    f"Context:\n{context}\n\nQuestion: {ex['query']} [/INST]"),
            "contexte": [corpus[i] for i in found[0]],
        }
        examples.append(row)
        print(json.dumps({k: row[k] for k in ("question", "reference", "sans_contexte", "avec_contexte")},
                         indent=1, ensure_ascii=False))
    with open(f"{WORK}/rag_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)
