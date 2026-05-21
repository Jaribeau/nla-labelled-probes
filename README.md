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
- `data/results/part1_refusal_acts_<tag>.npz` — raw held-out activations (L40/53/63) for Part 2.
- `data/probes/part1_refusal_dir_<tag>.json` — the unit direction + threshold per layer.

Render a report: `python scripts/viz_part1.py [results.json]` (writes a sibling `.html`).

### Part 2 — recover the direction from NLA-derived labels

Offline / no GPU. Reads a Part 1 run's artifacts, labels each activation by whether the NLA
explanation indicates refusal (Claude judge), rebuilds a diff-of-means direction from those
labels, and compares it to Arditi's. Needs `ANTHROPIC_API_KEY` in `.env`.

```bash
python scripts/part2_recover_direction.py [--run-id <tag>] [--pilot 10] [--regrade]
python scripts/viz_part2.py            # render the HTML report
```

Writes `data/results/part2_judge_<tag>.json` (raw judge calls, re-gradable) and
`part2_recover_<tag>.json` (cosine of NLA-label vs Arditi direction, per-layer AUROC,
divergence cases).

### Part 3 — audit: does the NLA catch a planted MCQ-format confound?

Trains a "refusal" probe on format-confounded data (harmful-MCQ vs harmless-free), scores a
de-confounded 2×2 with it and the clean Part 1 probe, then NLA-verbalizes the 2×2 activations
and theme-judges them (format / refusal / harmful_topic / other). The headline: the confounded
probe's firing set should audit as "format", the clean probe's as "refusal". Uses Modal
(extraction + NLA serving) and `ANTHROPIC_API_KEY` (theme judge).

```bash
python scripts/part3_audit_confound.py [--run-name v1] [--reextract] [--reverbalize] [--regrade]
python scripts/viz_part3.py            # render the HTML report
```

The two GPU steps (extraction, verbalization) and the judge are cached per `--run-name`, so
re-analysis is free; pass `--reextract` / `--reverbalize` / `--regrade` to force a step. Writes
`part3_acts_<rn>.npz`, `part3_verbalize_<rn>.json`, `part3_theme_<rn>.json`, and
`part3_audit_<rn>.json` (2×2 fire-rates, confound strength, per-probe theme distributions).

