# NLA Labeled Probes

---

Can a Natural Language Autoencoder (NLA) verbalize what a probe direction encodes, and can those verbalizations recover the direction? We test this on **refusal**, using Arditi et al.'s single refusal direction as ground truth.

A probe direction is just a vector; we infer its meaning from training data. NLAs (Fraser-Taliente et al. 2026) read out activation contents in language without concept labels, giving a direct answer to "what does this direction encode."

## Overview

See `research-plan.md` for the full plan and `research-log.md` for progress.

### Part 1 — Does the NLA verbalize refusal?

Reproduce Arditi's refusal direction on Llama-3.3-70B; check whether the NLA mentions refusal on probe-positive activations and not on probe-negative ones.

### Part 2 — Can we recover the refusal direction from NLA-derived labels?

Label activations by whether their NLA output mentions refusal (LLM judge), train a new diff-of-means probe on those labels, and compare to Arditi's (cosine similarity, AUROC).

---

## Setup

```bash
pip install -r requirements.txt
modal token new                        # one-time auth
cp .env.example .env                   # fill in HF_TOKEN
```

`HF_TOKEN` needs access to gated `meta-llama/Llama-3.3-70B-Instruct` and the NLA checkpoints `kitft/Llama-3.3-70B-NLA-L53-{av,ar}`.

## Run

### Part 1 — refusal direction + NLA

```bash
python scripts/part1_refusal_nla.py [--n-heldout 100] [--run-name tag]
```

Runs two Modal apps sequentially (never co-resident): 

- `src/target_model.py` extracts Llama-3.3-70B last-token resid_post at layers {40, 53, 63} + Arditi refusal scores
- then `src/nla_modal.py` serves the NLA AV (`kitft/Llama-3.3-70B-NLA-L53-av`) via SGLang and verbalizes the held-out L53 activations.

The script builds the diff-of-means refusal direction (`src/probe.py`) from refusal-score-filtered Arditi train data (splits vendored in `data/refusal/`), reports held-out AUROC per layer (sanity; L53 is committed), and writes:

- `data/results/part1_refusal_nla_<tag>.json` — per-prompt probe score, fires flag, NLA
explanation, and refusal-keyword hits (raw explanations kept for later LLM-judge re-grading).
- `data/probes/part1_refusal_dir_<tag>.json` — the unit direction + threshold per layer.

