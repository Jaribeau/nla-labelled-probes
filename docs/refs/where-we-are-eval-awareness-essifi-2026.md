# Essifi 2026 — Where We Are on Evaluation Awareness

- **Link:** https://www.lesswrong.com/posts/[slug]/where-we-are-on-evaluation-awareness
- **Author:** Yassine Essifi
- **Date:** 25 Apr 2026

## Summary

Survey post reviewing the current state of eval awareness in frontier models. Argues it is alignment's version of the observer effect, and that methodology to handle it is not yet in place.

**Why it's happening:** Labs train models on evaluations, creating a feedback loop — models learn to recognize eval distributions, and RL incentivizes performing well when they do. Anthropic found verbalized eval awareness on one honeypot distribution jumped from 20% to 80% during post-training. Unverbalized awareness (no CoT trace) is now also present (Mythos), possibly caused by accidentally training against the chain of thought.

**Why it matters:** Undermines capability and alignment evals. Models passing evals belong to an equivalence class of policies that may diverge in deployment. Governance frameworks (EU AI Act, RSP) rely on evals that may not reflect real behavior. Eval awareness scales as a power law with model size.

**Measurement methods:**

| Method | What it measures | Key limitation |
|---|---|---|
| CoT monitoring | Verbalized awareness | Breaks down if CoT is sanitized or non-verbalized |
| Transcript classification | Model's ability to classify transcripts as eval vs. deployment | Proxy; doesn't measure behavioral change |
| Linear probing | Activation-level representations of eval awareness | Assumes linear representation; unclear if causally meaningful |
| Probe-Rewrite-Evaluate (PRE) | Behavior shift when eval cues removed from prompt | Only captures surface-level cues; probe doesn't generalize across architectures |
| Activation oracles | LLM reads activations and reports internal state in NL | May confabulate; adds another black box |

**What can be done:** Make evals more realistic (cheap gains from typos/framing); stop training on evals; encrypted/private benchmarks; white-box steering during eval (limited — models can become steering-aware); track eval awareness throughout training.

## Why it matters here

Directly frames the measurement gap this project addresses. Activation oracles are highlighted as a promising but limited direction — our NLA/probe comparison speaks to whether NLA-based verbalizers are more reliable than linear probes for detecting eval awareness. Also useful for positioning: "Where We Are" is a recent community anchor for this problem.

## Notes

