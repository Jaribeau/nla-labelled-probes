# Plan: Verbalizing the Format Confound — NLAs as a Diagnostic for Eval-Awareness Probes

## Context

Safety evals may be critical to catch warning shots, or trigger development pauses. However, models increasingly display awareness that they are being evaluated, and as a result, behaving differently. This eval-awareness has the potential to quickly render alignment evals useless.

Devbunova 2026 showed that some eval-awareness probes (Nguyen et al. 2025) primarily track benchmark-canonical format rather than the evaluation-context concept. Her results showed probe accuracy degrades when format is controlled. She doesn't say *what* the format features are — only that the probe is keying on something other than the concept.

Natural Language Autoencoders (Fraser-Taliente et al. 2026) provide a way to read out activation contents in language. Applying NLAs to activations the probe fires on gives a direct verbal answer to "what is the probe actually detecting" — replacing Devbunova's indirect accuracy-degradation argument with explicit language about what features show up in the probe's positive activations.

---

## Research Question

> What features does the eval-awareness probe actually detect, expressed in natural language? 
>
> Specifically: on prompts where the probe fires despite the absence of an evaluation context (Devbunova's eval-format/deploy-context cell), what do NLA explanations of those activations say?

## Hypothesis

On the eval-format/deploy-context cell, NLA explanations will mention surface structural themes — multiple-choice formatting, "Question:" prefixes, formal register, structured prompt scaffolding — rather than evaluation-context themes. This directly verbalizes what Devbunova showed indirectly.

If NLA explanations on that cell instead mention evaluation-context themes, then either (a) the probe is detecting more than format, or (b) the NLA is confabulating around the probe's behavior rather than reading the activation. The 2nd phase, on toy-reward validation, distinguishes these.

---

## Approach

Three public artifacts will be used to run this experiment:

- **Devbunova's 2×2 dataset** (HuggingFace).
- **Nguyen et al.'s released probe weights** on Llama-3.3-70B-Instruct
- **NLA for Llama-3.3-70B-Instruct** (Neuronpedia)

The toy reward environment (Nitishinskaya & Schoen 2026) serves as a validation substrate where ground truth is known by construction (hint-present vs hint-absent).

## Infrastructure

- Target model (Llama-3.3-70B) hosted on NDIF, accessible via NNsight API 
- Probe applied locally
- NLA (Llama-3.3-70B-NLA-L53-av) hosted on Neuronpedia accessed via API

**Workflow**

1. Send prompts to NDIF using NNSight, get activations back
2. Apply probe locally
3. Send activations to Neuronpedia, get NLA explanations back



---

## Work Plan

### Phase 1 — Setup Tooling & Replicate Debunova

- Run Llama-3.3-70B-Instruct over Devbunova's 2×2 dataset, applying Nguyen et al.'s released Simple Contrastive probe to the residual stream activations. Reproduce her format-confound finding on this model.
- Note: Changing Llama-3.1-8B (Devbunova's target) to Llama-3.3-70B to match Neuronpedia's NLA. The expectation is that the format-confound replicates qualitatively. If it, the plan needs revisiting.

Deliverable: probe scores per 2×2 cell on Llama-3.3-70B + reproduced format-confound finding.



### Phase 2 — Validate NLA-as-characterization on toy reward

The core methodological claim is that NLA outputs on probe-firing activations reflect what the probe is detecting. Validate this on a substrate where the answer is known by construction.

Build the toy reward setup: matched prompts where only hint-presence differs. Train a probe on hint-present vs hint-absent (this is a second probe, trained separately from the eval-awareness one — its only job is to be a known-good probe with known target features). Apply NLA to activations the probe fires on. Verify NLA outputs mention reward/scoring on hint-present and don't on hint-absent.

This is the methodological sanity check. If NLAs can't verbalize the obvious thing on a controlled substrate, the rest of the plan collapses.

Deliverable: per-cell NLA theme distribution on toy reward + go/no-go decision.



### Phase 3 — Verbalize the Devbunova confound (core)

Apply the validated diagnostic to the Phase 1 probe on Devbunova's 2×2.

For each of the four 2×2 cells, sample activations where the probe fires above its classification threshold. Run NLA on those activations at the assistant turn boundary. Categorize NLA outputs by theme: structural-format mentions vs evaluation-context mentions vs other.

The contrast that matters most is **eval-format + deploy-context** — prompts where the probe fires but the evaluation-context is absent. NLA explanations on this cell give the direct readout of what the probe latched onto.

Secondary contrast: **deploy-format + eval-context** — prompts where the probe should fire if it's tracking the concept but doesn't (if Devbunova is right). NLA explanations on activations from this cell, even sub-threshold ones, may show whether eval-context information is present in the activation at all even though the probe missed it.

Deliverable: per-cell NLA theme distribution on Devbunova's 2×2 + writeup.



### Phase 4 (optional) — Behavioral correlation on toy reward

If Phase 3 produces clean results and there's time, extend the toy reward analysis:

- Does probe activation predict gaming behavior (model outputs an even number when reward favors even)?
- Does NLA-measured reward awareness predict gaming?
- Where do they disagree?

This connects the diagnostic to behavioral consequence rather than just internal representation.

### Future work — NLA-labeled probes

If NLAs can reliably verbalize what a probe is detecting, the next step is to explore using NLA outputs as the supervision signal for a new probe — replacing the format-correlated label "what class is this prompt" with the activation-grounded label "what does the NLA say is in this activation." This is working towards "NLA as probe factory".

## Validation logic

Phase 1 establishes that Devbunova's effect reproduces on the model we have NLAs for. Phase 2 validates that NLAs can verbalize what a probe is detecting in a controlled setting. Phase 3 applies the validated diagnostic to the unknown case. The chain of supporting results means a positive Phase 3 finding rests on validated methodology, not on the assumption that NLAs do what we think.

A null result is also informative. If Phase 3 NLA explanations on the eval-format/deploy-context cell mention evaluation context rather than format features, that pushes back on Devbunova — the probe may be detecting more than she claimed.

## Caveats and risks

- **Model substitution within family.** Devbunova used Llama-3.1-8B-Instruct; we use Llama-3.3-70B-Instruct (Nguyen's and Neuronpedia's target). Staying within the Llama family removes the family-substitution variable, but scale and instruction-tuning generation still differ. Phase 1 either reproduces the format-confound qualitatively (proceed) or doesn't (rethink).
- **NLA confabulation.** NLA outputs reflect what the AR can reconstruct as language, which isn't the same as what's in the activation. Phase 2 validates the diagnostic specifically against this concern by using a substrate where ground truth is known.
- **Token-position choice.** Hold fixed and document. The assistant turn boundary is the default given prior NLA work; Devbunova uses the last token of the prompt for her probe.
- **Layer alignment.** NLAs are trained at one layer. The Devbunova probe should be trained at that same layer for the NLA readout to be meaningful.
- **NLA verbosity and grader interpretation.** NLA outputs need categorization by theme, which means an LLM judge in the loop. Use the NLA paper's appendix grader prompts as a starting point; pilot the categorization before scaling.

## Key references

- Devbunova 2026 — "Is Evaluation Awareness Just Format Sensitivity?" (arXiv 2603.19426). Public 2×2 dataset and code.
- Nguyen et al. 2025 — "Probing and Steering Evaluation Awareness of Language Models" (arXiv 2507.01786). Standard contrastive probe methodology.
- Fraser-Taliente et al. 2026 — "Natural Language Autoencoders" ([transformer-circuits.pub/2026/nla](http://transformer-circuits.pub/2026/nla)). Method + public NLAs via Neuronpedia.
- Nitishinskaya & Schoen 2026 — "A Toy Environment for Exploring Reasoning About Reward" (LessWrong). Validation substrate with controllable hint-presence.
- Abdelnabi & Salem 2025 — "Linear Control of Test Awareness" (arXiv 2505.14617). Alternative probe methodology using CoT-derived labels; relevant comparison point.
- Rimsky et al. 2024 — Contrastive Activation Addition. Origin of difference-of-means probe approach.
- Arya, Rajamanoharan & Nanda 2026 — "Current Activation Oracles Are Hard to Use" (LessWrong, MATS 9.0). Empirical evaluation of AOs on safety tasks; key negative result (AUC ~0.55 on sycophancy vs. 0.77 for dense probe); text inversion confound.
- Essifi 2026 — "Where We Are on Evaluation Awareness" (LessWrong, Apr 2026). Survey of eval-awareness measurement methods and their limitations; frames AOs as promising but unreliable.
- Karvonen 2026 — "Realistic Evaluations Will Not Prevent Evaluation Awareness" (LessWrong). Argues realism engineering has structural limits.
- Anthropic Alignment 2026 — "Measuring and Improving Coding Audit Realism". Defines realism win rate metric; stakes heuristic in LLM judges.
- LessWrong community — "A Framework for Eval Awareness". Taxonomy for situating feature-decomposition work.
- LessWrong community — "Call for Science of Eval Awareness: Research Directions". Open problems; positions this project's contribution.

