# NLA Labeled Probes

🚀 [Read the report here](https://jareds.site/nla-probe-audit/report.html)

---

Can a Natural Language Autoencoder (NLA) verbalize what a probe direction encodes, and can those verbalizations recover the direction? We test this on **refusal**, using Arditi et al.'s single refusal direction as ground truth.

A probe direction is just a vector; we infer its meaning from training data. NLAs (Fraser-Taliente et al. 2026) read out activation contents in language without concept labels, giving a direct answer to "what does this direction encode."

## Overview

See `research-plan.md` for the full plan and `research-log.md` for progress.

### Part 1 — Does the NLA verbalize refusal?

Reproduce Arditi's refusal direction on Llama-3.3-70B; check whether the NLA mentions refusal on probe-positive activations and not on probe-negative ones.

### Part 2 — Can we recover the refusal direction from NLA-derived labels?

Label activations by whether their NLA output mentions refusal (LLM judge), train a new diff-of-means probe on those labels, and compare to Arditi's (cosine similarity, AUROC).

### Part 3 —  Can the NLA audit a probe and catch a hidden confound? Building the probe audit pipeline.

Audit *what a probe actually fires on* by reading the NLA verbalizations of its activations. As a test, train a "refusal" probe on format-confounded data (harmful multiple-choice vs harmless free-text) and check whether the audit surfaces "multiple-choice format" — the planted confound — without it being named in advance.

---

## Setup

```bash
pip install -r requirements.txt
modal token new                        # one-time auth
cp .env.example .env                   # fill in HF_TOKEN
```

`HF_TOKEN` needs access to gated `meta-llama/Llama-3.3-70B-Instruct` and the NLA checkpoints `kitft/Llama-3.3-70B-NLA-L53-{av,ar}`.

## Run

Every script is **interactive by default**: run it with no arguments and it prompts for each
parameter, listing prior runs / files to pick from. The flags shown in each script's docstring
still work and skip the prompts (handy for scripting). To see/choose all runs and their
artifacts in one place:

```bash
python scripts/viz_runs.py              # writes runs.html at the repo root and opens it
```

### Part 1 — Does the NLA verbalize refusal?

```bash
python scripts/part1_refusal_nla.py
```

Runs two Modal apps sequentially:

- `src/target_model.py` extracts Llama-3.3-70B last-token resid_post at layers {40, 53, 63} + Arditi refusal scores
- then `src/nla_modal.py` serves the NLA AV (`kitft/Llama-3.3-70B-NLA-L53-av`) via SGLang and verbalizes the held-out L53 activations.

The script builds the diff-of-means refusal direction (`src/probe.py`) from refusal-score-filtered Arditi train data (splits vendored in `data/refusal/`), reports held-out AUROC per layer (sanity; L53 is committed), and writes:

- `data/results/part1_refusal_nla_<tag>.json` — per-prompt probe score, fires flag, NLA
explanation, and refusal-keyword hits (raw explanations kept for later LLM-judge re-grading).
- `data/results/part1_refusal_acts_<tag>.npz` — raw held-out activations (L40/53/63) for Part 2.
- `data/probes/part1_refusal_dir_<tag>.json` — the unit direction + threshold per layer.

Render a report: `python scripts/viz_part1.py` (pick a run; writes a sibling `.html`).

### Part 2 — Can we recover the refusal direction from NLA-derived labels?

Reads Part 1 artifacts, labels each activation by whether the NLA explanation indicates refusal (Claude judge), rebuilds a diff-of-means direction from those labels, and compares it to Arditi's.

```bash
python scripts/part2_recover_direction.py   # prompts: which Part 1 run, pilot, regrade, ...
python scripts/viz_part2.py                  # render the HTML report
```

### ⭐️ Part 3 — The probe audit pipeline

The audit answers "what concepts actually make this probe fire?" 

The pipeline lives in `scripts/audit_pipeline/`. Run them in order to audit  
any probe on any prompt set. The built-in prompt set is the confound demo: a probe trained on  
harmful-MCQ vs harmless-free-text, audited on a de-confounded 2×2 to see whether the planted  
"multiple-choice format" confound surfaces. 

GPU stages use Modal; concept extraction needs`ANTHROPIC_API_KEY`.

**Prerequisite: Build the probe** (or skip and use an existing one):

```bash
python scripts/audit_pipeline/0_build_probe.py
```

- *Input:* built-in confound training set. 
- *Output:* `data/probes/part3_confound_dir_<run>.json`.

**Stage 1 — run prompts + extract activations** (GPU):

```bash
python scripts/audit_pipeline/1_run_prompts_extract_activations.py
```

- *Input:* built-in 2×2 prompt set. 
- *Output:* `part3_acts_<run>.npz` + `part3_prompts_<run>.json`.

**Stage 2 — verbalize activations with the NLA** (GPU):

```bash
python scripts/audit_pipeline/2_verbalize_activations.py
```

- *Input:* `part3_acts_<run>.npz`. 
- *Output:* `part3_verbalize_<run>.json`.

**Stage 3 — probe the activations** (local):

```bash
python scripts/audit_pipeline/3_probe_activations.py
```

- *Input:* `part3_acts_<run>.npz` + selected probe files. 
- *Output:* `part3_probe_<run>.json`
(per-probe projections; plus an optional `confound_validation` block — per-cell fire rates and
confound strength — when the prompt set carries the 2×2 labels).

**Stage 4 — extract concepts from the verbalizations** (needs `ANTHROPIC_API_KEY`):

```bash
python scripts/audit_pipeline/4_extract_concepts_from_nla_verbalizations.py
```

- *Input:* `part3_verbalize_<run>.json`. 
- *Output:* `part3_concepts_<run>[_<grade-tag>].json`
(open-vocab tags per activation). A grade tag keeps a re-grade alongside prior gradings.

**Stage 5 — analyze + visualize probe–concept associations** (local):

```bash
python scripts/audit_pipeline/5_analyze_probe_concept_associations.py
```

- *Input:* `part3_concepts_<run>.json` + `part3_probe_<run>.json` (+ `part3_verbalize` /
`part3_prompts`). 
- *Output:* `part3_association_<run>.json` + `.html` — one **interactive,
self-contained** explorer: a probe toggle switches the chart, and clicking a concept bar traces
the activations tagged with it end-to-end (prompt → NLA readout → probe firing → concepts).

**Inspect the raw concept vocabulary** (complements stage 4) before any `min-count` filtering —
useful for spotting synonym fragmentation ("multiple choice" vs "multiple-choice format"):

```bash
python scripts/inspect_concepts.py   # pick a concept set; renders an HTML tag-count table
```

