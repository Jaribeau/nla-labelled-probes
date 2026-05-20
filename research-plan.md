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

- **Refusal direction**: Arditi-style difference-of-means over refused vs. complied
activations (harmful vs. harmless instructions) on Llama-3.3-70B.
- **NLA**: self-hosted `kitft/Llama-3.3-70B-NLA-L53-av` (no hosted API); verbalize activations
at the AV's layer (resid_post, block 53).
- **LLM judge**: binary "does this explanation mention refusal?" labeling of NLA outputs.

## Infrastructure

- Llama-3.3-70B-Instruct (target) and the NLA AV both run on Modal.com on-demand GPUs.
- Activations extracted on the Modal worker at layer 53; probe applied to returned activations.

## Work Plan

### Part 1 — Does the NLA verbalize refusal?

1. Reproduce Arditi's refusal direction (diff-of-means, refused vs. complied) on Llama-3.3-70B.
2. Run a held-out prompt set; for each, record the probe score and the NLA explanation at
  layer 53.
3. Check NLA outputs mention refusal/safety on probe-positive activations and not on
  probe-negative ones.

Deliverable: per-prompt probe score + NLA explanation; refusal-mention rate by probe polarity.

---

### Part 2 — Can we recover the refusal direction from NLA-derived labels?

1. On a fresh prompt set, label each activation by whether its NLA output mentions refusal
  (LLM judge, binary).
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

