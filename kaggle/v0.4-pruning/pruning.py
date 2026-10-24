"""v0.4 - Pruning par magnitude de Mistral 7B : courbe sparsite / qualite (Kaggle T4).

On met a zero une part croissante des poids les plus petits (en valeur absolue) de chaque couche
lineaire des blocs du transformer, et on mesure l'accuracy de sentiment IMDB a chaque palier,
avec la meme methode que les autres etapes (logits de classe, sans generation).

Limites assumees :
  - pas de re-entrainement apres pruning : on mesure la degradation brute ;
  - pruning non structure : les zeros restent stockes, donc ni la memoire ni la vitesse ne changent.
Lancement : kaggle kernels push -p kaggle/v0.4-pruning
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Configuration ----------
MODEL_ID = "unsloth/mistral-7b-instruct-v0.3"      # copie en acces libre du modele officiel, en pleine precision
SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N_EVAL = 1000
SEED = 42
TEXT_MAX_CHARS = 400
WORK = "/kaggle/working/v04_pruning"
PROMPT = ("[INST] Analyze the sentiment of this movie review and respond with exactly one word: "
          "positive or negative.\n\nReview: {text} [/INST]")
LABELS = (("positive", "Positive"), ("negative", "Negative"))      # index 0 = positive, 1 = negative

os.makedirs(WORK, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# IMDB est trie par label sur le Hub : on melange avant de prendre les N premiers
raw = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=SEED).select(range(N_EVAL))
examples = [(PROMPT.format(text=ex["text"][:TEXT_MAX_CHARS]), 1 - ex["label"]) for ex in raw]

base_ids = tok(PROMPT.format(text="example"))["input_ids"]
GROUPS = [sorted({tok(PROMPT.format(text="example") + w)["input_ids"][len(base_ids)] for w in variants})
          for variants in LABELS]
assert not set(GROUPS[0]) & set(GROUPS[1])

# le fp16 remplit presque un T4 a lui seul : on le repartit sur les deux GPU quand il y en a deux
device_map = "balanced" if torch.cuda.device_count() > 1 else {"": 0}
print(f"{torch.cuda.device_count()} GPU, device_map = {device_map}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=device_map).eval()
linears = [m for m in model.model.layers.modules() if isinstance(m, torch.nn.Linear)]
print(f"{len(linears)} couches lineaires a elaguer, {sum(m.weight.numel() for m in linears) / 1e9:.2f} milliards de poids")


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p + z * z / (2 * n)) / d - h, (p + z * z / (2 * n)) / d + h


@torch.no_grad()
def prune(target):
    # par couche : le seuil est la valeur absolue du k-ieme plus petit poids, tout ce qui est dessous passe a zero.
    # Les paliers sont cumulatifs : les zeros deja poses font partie des plus petits.
    for lin in linears:
        w = lin.weight
        k = int(w.numel() * target)
        if k:
            threshold = torch.kthvalue(w.abs().flatten().float(), k).values
            w[w.abs() <= threshold.to(w.dtype)] = 0


@torch.no_grad()
def sparsity():
    return sum((m.weight == 0).sum().item() for m in linears) / sum(m.weight.numel() for m in linears)


@torch.no_grad()
def evaluate():
    correct = positive = 0
    for prompt, c in examples:
        enc = tok(prompt, return_tensors="pt").to(model.device)
        last = model(**enc).logits[0, -1].float()
        scores = [torch.logsumexp(last[g], 0).item() for g in GROUPS]
        positive += scores[0] >= scores[1]
        correct += (scores[1] > scores[0]) == bool(c)
    return correct, positive / N_EVAL


assert sparsity() < 0.001      # controle : le modele d'origine n'a pas de zeros
rows = []
for target in SPARSITIES:
    prune(target)
    real = sparsity()
    assert abs(real - target) < 0.01, (target, real)      # controle : on a bien elague ce qu'on voulait
    correct, frac_positive = evaluate()
    low, high = wilson(correct, N_EVAL)
    rows.append({"target_sparsity": target, "real_sparsity": real, "accuracy": correct / N_EVAL,
                 "ci_low": low, "ci_high": high, "frac_pred_positive": frac_positive, "n_eval": N_EVAL})
    pd.DataFrame(rows).to_csv(f"{WORK}/pruning.csv", index=False)
    print(rows[-1])

df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(df["real_sparsity"] * 100, df["accuracy"],
            yerr=[df["accuracy"] - df["ci_low"], df["ci_high"] - df["accuracy"]], marker="o", capsize=4)
ax.axhline(0.5, color="gray", linestyle="--", label="chance")
ax.set_xlabel("Sparsity (% of weights set to zero)")
ax.set_ylabel("IMDB accuracy (95% CI)")
ax.set_title("Mistral 7B: magnitude pruning, no retraining")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{WORK}/pruning.png", dpi=150)
