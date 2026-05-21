# Plan: Can an NLA Verbalize a Probe Direction? — Refusal as a Test Case

## Context

Probes read out concepts from model activations, but a probe direction is just a vector — we
infer what it means from the data it was trained on. Natural Language Autoencoders (NLAs,
Fraser-Taliente et al. 2026) verbalize activation contents in language, offering a direct
read of what a direction encodes. Refusal is a good test case: Arditi et al. 2024 showed it
is mediated by a single, reliably recoverable direction, giving a strong ground-truth probe
to compare against.

## Research Question

> 1. Does the NLA verbalize refusal on activations where a refusal probe fires?
> 2. Can the refusal direction be recovered from NLA-derived labels alone?

## Hypothesis

NLA explanations on probe-positive (refusal) activations mention refusal themes and  
don't on probe-negative ones. If so, labels derived purely from NLA output should recover a  
direction close to Arditi's (high cosine similarity, comparable AUROC).

## Approach

1. **Use NLA to verbalize activations on harmful/harmless prompts**
2. **Evaluate NLA verbalizations with LLM judge** ("does this explanation mention refusal?" labeling of NLA outputs.)
3. **Train NLA Labelled Probe** (using using refused/permitted NLA outputs.)
4. **Train standard refusal direction probe** (using harmful/benign prompt dataset labels.)
5. Compare to standard probe

## Infrastructure

- Llama-3.3-70B-Instruct (target) and the NLA AV both run on Modal.com on-demand GPUs.
- Activations extracted on the Modal worker at layer 53; probe applied to returned activations.

## Work Plan

### ✅ Part 1 — Does the NLA verbalize refusal?

1. Reproduce Arditi's refusal direction (diff-of-means, refused vs. complied) on Llama-3.3-70B.
2. Run a held-out prompt set; for each, record the probe score and the NLA explanation at
  layer 53.
3. Check NLA outputs mention refusal/safety on probe-positive activations and not on
  probe-negative ones.

Deliverable: per-prompt probe score + NLA explanation; refusal-mention rate by probe polarity.

#### Implementation

Arditi's recipe (datasets, chat template, refusal-score filter, diff-of-means) is documented in
`docs/refs/refusal-direction-arditi-2024.md`; only our deviations and new code are listed here.

- **Data** — vendor Arditi's precomputed splits into `data/refusal/`; loader `src/data_refusal.py`.
Sample `n_train=128`/`n_val=32` (seed 42), then apply Arditi's refusal-score filter.
- **Extraction** — `src/target_model.py` (Modal, Llama-3.3-70B, adapt from `src/modal_app.py`):
capture **resid_post** at the **last token** (assistant-turn boundary) for layers **{40, 53, 63}**
in one forward pass. One vector per instruction per layer. *(Deviation from Arditi: fixed
extraction point instead of his per-(pos, layer) selection, so probe and NLA read the same
activation.)*
- **Probe** — `refusal_dir = mean(harmful) − mean(harmless)` per layer via `src/probe.py`
(`build_probe`); score held-out activations by projection (`project`). The {40, 53, 63} sweep
is a health check / partial replication — report AUROC per layer; the best-separating layer
should land where Arditi's selection does (~60–80% depth; L53 ≈ 66%). **L53 is committed** for
the NLA comparison; 40/63 are sanity only.
- **NLA verbalization** — `src/nla.py` (`NLAClient`) + `src/nla_modal.py` (Modal app serving
`kitft/Llama-3.3-70B-NLA-L53-av` via SGLang). Inject each held-out L53 activation, get the
`<explanation>`; reads `injection_scale`/template from the checkpoint's `nla_meta.yaml`. Target
extraction and AV serving run **sequentially** (two 70B models).
- **Grading** — keyword/regex pass for refusal/safety terms; persist every raw explanation +
probe score per prompt to `data/results/part1_refusal_nla_<runid>.json` for later LLM-judge re-grading.
- **Orchestrator** — `scripts/part1_refusal_nla.py`.

Open choices: held-out set = balanced `harmful_test` + `harmless_test` sample; probe-positive/negative
split by projection sign (or train-projection midpoint threshold).

---

### ✅ Part 2 — Can we recover the refusal direction from NLA-derived labels?

1. On a fresh prompt set, label each activation by whether its NLA output mentions refusal
  (LLM judge).
2. Train a new diff-of-means probe on those NLA-derived labels.
3. Compare to Arditi's probe: cosine similarity of directions, AUROC agreement on a held-out
  test set.

Deliverable: cosine similarity + AUROC agreement between NLA-label probe and Arditi probe.

## Validation logic

Part 1 establishes the NLA verbalizes the concept a known-good probe fires on. Part 2 tests
the stronger claim that NLA labels alone suffice to recover the direction. A positive Part 2
rests on Part 1 holding.

## Caveats and risks

- **NLA confabulation** — NLA output may not reflect activation contents; Part 1 is the check.
- **Layer coupling** — probe and NLA must use the same layer (53) for the comparison to mean
anything.
- **Judge reliability** — binary refusal-mention judgments need a piloted, fixed prompt.

## Key references

- Arditi et al. 2024 — "Refusal in LLMs Is Mediated by a Single Direction" (arXiv 2406.11717).
- Fraser-Taliente et al. 2026 — "Natural Language Autoencoders" (transformer-circuits.pub/2026/nla).
NLA method + checkpoints (`kitft/Llama-3.3-70B-NLA-L53-{av,ar}`); inference recipe at
github.com/kitft/nla-inference.
- Rimsky et al. 2024 — Contrastive Activation Addition. Origin of the diff-of-means approach.

