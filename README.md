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

*Implementation in progress — run instructions will be added as Part 1 / Part 2 are built.*