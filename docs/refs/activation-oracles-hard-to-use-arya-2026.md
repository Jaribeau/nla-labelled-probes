# Arya, Rajamanoharan, Nanda 2026 — Current Activation Oracles Are Hard to Use

- **Link:** https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use
- **Authors:** aryaj, Senthooran Rajamanoharan, Neel Nanda (MATS 9.0)
- **Date:** 3 Mar 2026

## Summary

Empirical evaluation of activation oracles (AOs; from Karvonen et al.) on safety-relevant tasks. Overall: AOs show signal near their training distribution but are difficult to use reliably in practice. Negative update on AOs as a useful tool in current form.

**What AOs are:** A model finetuned to answer NL questions about another model's activations. Trained on ~1M examples including 600K text inversion tasks (predicting nearby tokens). Activations injected into layer 1 of the residual stream.

**Main failure modes:**

- **Vagueness:** 49.4% of responses across 77 probes on 33 diverse rollouts were generic/unfalsifiable (e.g. "the model is uncertain about its calculations" for any math problem).
- **Hallucination:** Confidently wrong answers; gets domain vaguely right but object wrong (e.g. "arrangements of letters in BANANA" on a plate permutation problem; "number of fish in the pond" on an ice cube problem).
- **Text inversion confound:** 600K training examples involve predicting nearby tokens, so AOs can appear to succeed by recovering surrounding text rather than reading internal state. This is hard to disentangle without carefully controlled experiments.
- **It's-an-LLM confound:** AO may answer correctly from its own weights without using the activations at all. Running AO *without* activations on Chinese models produced comparable or better answers.

**Tasks where AOs struggled (text inversion controlled):**

| Task | Result |
|---|---|
| Sycophancy detection (Scruples dataset, Qwen 3 32B) | AUC ~0.55 (chance = 0.50); dense probe on same activations: 0.77 AUC |
| Number prediction (arithmetic, no CoT) | Confabulated same numbers ("10", "12") regardless of problem |
| Missing information detection (complete vs. incomplete prompt) | Correct ID of missing info: 0/54; same classification in 53/54 cases for identical tokens, different model state |
| Backtracking reason (reasoning model) | ~5% correct after text inversion filter; correct updates tracked nearby keywords, not deeper state |

**Where AOs show signal:**

- Multiple-choice backtracking reason (4 options given): 66% correct vs. 47% for LLM with same tokens.
- Next/previous token prediction in CoT: F1 0.38/0.45 vs. LLM baseline 0.30/0.20.
- Subtle steering detection: detected pirate concept vector at low coefficients before it was visible in output text.

**Chinese models aside:** AO finetuning from Qwen partially jailbreaks the model, complicating interpretation. Shared knowledge between AO and target model confounds results.

**Authors' diagnosis:** Limitations likely reflect training distribution and methods, not a fundamental ceiling. Suggested directions: train on reasoning traces (not just pretraining text); train for calibration (so AO admits uncertainty rather than confabulating).

## Why it matters here

This is the key piece of prior work establishing that AOs are unreliable — particularly on tasks requiring them to read internal state beyond nearby text. Our project tests whether NLA provides a better alternative or complement. The sycophancy detection comparison (AO: 0.55 AUC vs. dense probe: 0.77 AUC) is a direct point of comparison for our probe vs. NLA comparison. The text inversion confound is also a methodological hazard we need to control for.

## Notes

