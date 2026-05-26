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
> 3. Can an NLA + LLM-judge *audit* detect when a probe tracks a **confound** rather than its
>   named concept — without building a counterfactual dataset?

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

---

### ✅ Part 3 — Audit demo: NLA catches a planted MCQ-format confound

A probe learns whatever separates its labels, which may be a confound. NLA gives a label-free readout of what a probe actually fires on. 

Part 3 plants an MQC format confound with known ground truth and aims to show that the NLA audit catches the confound.

**Implementation**

1. **Confounded data** — render Arditi instructions in two formats (same content): an MCQ
  wrapper (`Question: <instr>\n\nChoices:\n(A) …\n(B) …\nAnswer:`) vs free-form prose.
2. **Confounded probe** — train a diff-of-means "refusal" probe on harmful-MCQ (1) vs
  harmless-free (0), so format ≡ label. Expect high in-distribution accuracy.
3. **De-confounded 2×2** ({harmful,harmless} × {MCQ,free}) — the expensive standard check:
  fire-rate per cell. A concept probe fires on both harmful cells; a format probe fires on both
   MCQ cells. The off-diagonal reveals confounding and quantifies how much it "bit".
4. **NLA audit (the cheap catch)** — verbalize the activations the confounded probe fires on; an
  LLM theme-judge classifies each explanation: structural/format vs refusal/safety vs
   harmful-topic vs other. A confounded probe → "format" prominent.
5. **Control** — same audit on the clean Part 1 refusal probe → expect "refusal". The
  confounded-vs-clean theme distributions side by side are the headline.

Deliverable: confounded probe looks fine in-distribution but the 2×2 + NLA audit both flag
format; the audit reaches the 2×2's verdict without needing the 2×2. The clean probe audits as
"refusal" — the audit discriminates good from confounded probes.

Bite-risk: refusal/harm signal is strong (Part 1 AUROC 1.0), so format may not fully dominate.
Mitigation: measure confound strength via the 2×2 fire-rate gap and treat it as a dial; a
*partially* confounded probe should still show a *mixed* audit theme distribution, which still  
demonstrates audit sensitivity. Report the confounding strength.

New code: `src/data_confound.py` (MCQ/free formatting + confounded train + 2×2 splits);
`src/judge.py` += `judge_theme` (multi-class, piloted prompt, same robust/cached pattern);
`scripts/part3_audit_confound.py`; `scripts/viz_part3.py` (theme bars confounded-vs-clean, 2×2
table). Reuse `src/target_model.py`, `src/probe.py`, `src/nla_modal.py`.

#### ✅ Part 3b — From fixed themes to concept enrichment (the general audit readout)

The v1 run worked but exposed two limits of a fixed theme taxonomy: it requires *knowing* the
confound in advance, and its classes are baseline-contaminated (the NLA's "Q&A format" tic made
even benign prose 92% "format"). Fix both with an **open-vocabulary, self-baselining** readout.

- `**judge_concepts*`* (`src/judge.py`) — per NLA explanation, emit a short list (≤4) of concise
open-vocab concept tags (lowercase noun phrases). Reuses the `_batch` runner; robust + cached.
v1 normalizes by lowercase/strip; add a codebook-merge pass later if synonym noise is high.
- **Enrichment** — for a probe, split its activations into top-third vs bottom-third by
projection, then per concept `enrichment = freq(concept | high) − freq(concept | low)`.
Baseline concepts (the Q&A tic) appear in both → cancel to ~0 → drop out. Top-enriched concepts
*are* "what the probe detects". No pre-specified taxonomy → works for **discovery** (Part 4).
- **Visualization** — a ranked diverging horizontal **enrichment bar chart**, two panels
(confounded vs clean). The confound concept ("multiple-choice / format") surfaces near the top
for the confounded probe and is ~0 for the clean one — the audit naming the confound unprompted.

Offline / no GPU: reuses the cached `part3_acts_<rn>.npz` + `part3_verbalize_<rn>.json`; one
cached API pass for concept extraction. New code: `judge_concepts`; `scripts/part3_enrichment.py`
(general: probe dir + acts + explanations → per-concept enrichment table); `scripts/viz_enrichment.py`.

---

### Part 4 (exploratory) — Test NLA audit performance on harder confounds

Try to find where the audit stops separating confound from concept.  
Plant each confound (ground truth known) and reuse Part 3 machinery (build → train → verbalize
acts → **concept-enrichment readout**, Part 3b). Enrichment is the discovery tool here: the
confound isn't named in advance, so we look for whichever concept is most enriched in the
probe's high-projection activations.

Confound levels (increasing in expected difficulty):

- **Level 1 — format (i.e. MQC)**
- **Level 2 — stylistic/affective**: confound the concept with formal register / politeness /  
sentiment. Does the audit say "formal register" vs the concept?
- **Level 3 — topical**: confound the positive class with a topic cluster. Topic vs concept?
- **Level 4 — correlated concept**: eval-awareness vs "artificiality/test-ness"; deception vs hedging / uncertainty / role-play (highest safety value; where there is no behavioral ground truth, so an NLA audit would be helpful).

Deliverable: per-confound audit verdict + a **difficulty-ladder summary** (audit catches levels 1–? reliably, degrades at level ?). 

Limitations: the audit can only catch confounds the NLA verbalizes *and describes differently* from the concept; level-4 entanglement is where NLA language overlaps. Specific level-4 targets TBD.

## Validation logic

Part 1 establishes the NLA verbalizes the concept a known-good probe fires on. Part 2 tests
the stronger claim that NLA labels alone suffice to recover the direction. A positive Part 2
rests on Part 1 holding.

Part 3 shows the audit catches a confound with known ground truth (planted MCQ format),
validated against the de-confounded 2×2. Part 4 maps how far up the confound-difficulty ladder
the audit reaches. Together they establish NLA + judge as a cheap, general probe auditor — the
contribution that survives refusal being an easy case.

## Caveats and risks

- **NLA confabulation** — NLA output may not reflect activation contents; Part 1 is the check.
- **Layer coupling** — probe and NLA must use the same layer (53) for the comparison to mean
anything.
- **Judge reliability** — binary refusal-mention judgments need a piloted, fixed prompt.
- **Confound must bite** — a planted confound only tests the audit if it actually shifts the
probe; strong intrinsic concept signal (refusal) may resist it. Measure and report strength.
- **Audit reach is bounded by NLA expressiveness** — the audit can only surface confounds the
NLA verbalizes and distinguishes in language from the concept. Mapping that boundary is a
Part 4 goal, not just a limitation.

## Key references

- Arditi et al. 2024 — "Refusal in LLMs Is Mediated by a Single Direction" (arXiv 2406.11717).
- Fraser-Taliente et al. 2026 — "Natural Language Autoencoders" (transformer-circuits.pub/2026/nla).
NLA method + checkpoints (`kitft/Llama-3.3-70B-NLA-L53-{av,ar}`); inference recipe at
github.com/kitft/nla-inference.
- Rimsky et al. 2024 — Contrastive Activation Addition. Origin of the diff-of-means approach.

