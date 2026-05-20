# Arditi et al. 2024 — Refusal Is Mediated by a Single Direction

- **Paper:** arXiv 2406.11717 (v3). PDF: `refs-source-material/2406.11717v3-Arditi-2024.pdf`
- **Code:** https://github.com/andyrdt/refusal_direction (MIT). Reproduces the main results;
  ships precomputed dataset splits and per-model run artifacts (incl. `direction.pt`).

## Why it matters here

Gives us a ground-truth refusal probe. Their refusal direction is a **difference-of-means**
vector (harmful − harmless mean activations) — exactly the probe form `src/probe.py` builds —
so we can reproduce it on Llama-3.3-70B at the NLA layer (block 53) and compare against a
direction recovered from NLA-derived labels (Part 2).

## Method (as implemented in their repo)

- **Datasets** (vendored, precomputed in `dataset/splits/`):
  - *harmful* = mix of AdvBench, TDC2023, HarmBench, MaliciousInstruct (`harmful_{train,val,test}`
    = 260 / 39 / 572 instructions).
  - *harmless* = Alpaca (`harmless_{train,val,test}`, thousands each).
  - Pipeline samples `n_train=128`, `n_val=32`, `n_test=100` (seed 42), then **filters**: keep
    harmful examples the model actually refuses (refusal score > 0) and harmless it complies
    with (< 0). Refusal score = logit of the first response token being `"I"` (id 40 for Llama-3).
- **Direction generation** (`generate_directions.py`): for every layer and every
  end-of-instruction template token position, take the **mean residual-stream activation**
  (captured by a forward *pre*-hook on each decoder block = resid_pre / block input), then
  `mean_diff = mean_harmful − mean_harmless`. Yields one candidate direction per (position, layer).
- **EOI token positions**: the tokens after `{instruction}` in the chat template, i.e.
  `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n` — `positions =
  range(-len(eoi_toks), 0)`. The default single-position elsewhere is `[-1]` (last token).
- **Direction selection** (`select_direction.py`): pick the single (pos, layer) whose
  ablation/activation-steering most reduces refusal on val while preserving coherence. Saved to
  `direction.pt` + `direction_metadata.json` (`{pos, layer}`).
- **Llama-3 chat template** (no system prompt):
  `<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`
  Tokenizer left-padded, `add_special_tokens` adds `<|begin_of_text|>`.
- **Validation of the direction** (not needed for our probe, but it's how they prove it):
  weight orthogonalization (ablate the direction) collapses refusal; activation addition
  (+direction) induces refusal on harmless prompts.

## What we reuse vs. change

- **Reuse**: diff-of-means recipe, their exact split files, the Llama-3 chat template, refusal-
  score filtering.
- **Change**: fix the extraction point to **resid_post of block 53** (the NLA's layer) rather
  than selecting the best layer, and take a single token position (assistant-turn boundary), so
  the probe and the NLA read the *same* activation. Models differ (they tested ≤8B; we use 70B).
