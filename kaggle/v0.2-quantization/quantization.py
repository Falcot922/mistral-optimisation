"""v0.1 / v0.2 - Baseline et quantization de Mistral 7B (Kaggle T4).

Meme modele charge en trois precisions, mesure a chaque fois sur :
  qualite : accuracy de sentiment IMDB sans entrainement (meme methode que la v0.5 : logits de classe)
  vitesse : tokens generes par seconde
  memoire : pic de VRAM et taille des poids

AWQ et GGUF ne sont pas mesures : autoawq n'est plus maintenu et ne s'installe plus avec les
versions actuelles de transformers, et GGUF passe par llama.cpp, un autre moteur d'inference.
Lancement : kaggle kernels push -p kaggle/v0.2-quantization
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46.1"], check=True)

import gc
import math
import os
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------- Configuration ----------
MODEL_ID = "unsloth/mistral-7b-instruct-v0.3"      # copie en acces libre du modele officiel, en pleine precision
N_EVAL = 500
SEED = 42
TEXT_MAX_CHARS = 400
NEW_TOKENS = 100
WORK = "/kaggle/working/v02_quantization"
PROMPT = ("[INST] Analyze the sentiment of this movie review and respond with exactly one word: "
          "positive or negative.\n\nReview: {text} [/INST]")
LABELS = (("positive", "Positive"), ("negative", "Negative"))      # index 0 = positive, 1 = negative
GEN_PROMPTS = ["[INST] Explain what a neural network is. [/INST]",
               "[INST] Write a short story about a lighthouse keeper. [/INST]",
               "[INST] Describe how photosynthesis works. [/INST]"]
VARIANTS = {
    "fp16": dict(dtype=torch.float16),
    "int8": dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True)),
    "nf4": dict(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)),
}

os.makedirs(WORK, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# IMDB est trie par label sur le Hub : on melange avant de prendre les N premiers
raw = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=SEED).select(range(N_EVAL))
examples = [(PROMPT.format(text=ex["text"][:TEXT_MAX_CHARS]), 1 - ex["label"]) for ex in raw]
print(f"{N_EVAL} critiques, {sum(c == 0 for _, c in examples) / N_EVAL:.0%} positives")

# premier token de chaque variante d'ecriture, pour chaque classe
base_ids = tok(PROMPT.format(text="example"))["input_ids"]
GROUPS = [sorted({tok(PROMPT.format(text="example") + w)["input_ids"][len(base_ids)] for w in variants})
          for variants in LABELS]
assert not set(GROUPS[0]) & set(GROUPS[1])


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p + z * z / (2 * n)) / d - h, (p + z * z / (2 * n)) / d + h


@torch.no_grad()
def measure(name, load_kwargs):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map={"": 0}, **load_kwargs).eval()
    row = {"variant": name, "load_s": time.time() - t0, "weights_gib": model.get_memory_footprint() / 2**30}

    correct = 0
    for prompt, c in examples:
        last = model(**tok(prompt, return_tensors="pt").to("cuda")).logits[0, -1].float()
        scores = [torch.logsumexp(last[g], 0).item() for g in GROUPS]
        correct += (scores[1] > scores[0]) == bool(c)
    row["accuracy"] = correct / N_EVAL
    row["ci_low"], row["ci_high"] = wilson(correct, N_EVAL)

    def gen(prompt):
        enc = tok(prompt, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t = time.time()
        # min = max : chaque variante genere exactement le meme nombre de tokens
        model.generate(**enc, min_new_tokens=NEW_TOKENS, max_new_tokens=NEW_TOKENS, do_sample=False,
                       pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return time.time() - t

    gen(GEN_PROMPTS[0])                                             # echauffement, non compte
    row["tokens_per_s"] = NEW_TOKENS * len(GEN_PROMPTS) / sum(gen(p) for p in GEN_PROMPTS)
    row["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return row


rows = []
for name, load_kwargs in VARIANTS.items():
    try:
        rows.append(measure(name, load_kwargs))
    except Exception as e:      # une variante qui ne tient pas en memoire ne doit pas faire perdre les autres
        rows.append({"variant": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    gc.collect()
    torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(f"{WORK}/quantization.csv", index=False)
    print(rows[-1])

print(pd.DataFrame(rows).round(3).to_string(index=False))
