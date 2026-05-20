# NLA Labeled Probes

---

Comparing NLA to probes, and exploring the viability of using NLA output to train probes for detection of eval-awareness. 

Safety evals may be critical to catch warning shots, or trigger development pauses. However, models increasingly display awareness that they are being evaluated, and as a result, behaving differently. This eval-awareness has the potential to quickly render alignment evals useless.

Devbunova (2026) showed that some eval-awareness probes (Nguyen et al. 2025) primarily track benchmark-canonical format rather than the evaluation-context concept. The results showed probe accuracy degrades when format is controlled. The results don't say *what* the format features are — only that the probe is keying on something other than the concept.

Natural Language Autoencoders or NLAs (Fraser-Taliente et al. 2026) provide a way to read out activation contents in language - without requiring labels of any specific concept. Applying NLAs to activations the probe fires on gives a direct verbal answer to "what is the probe actually detecting" — providing explicit language about what features show up in the probe's positive activations.

Here, I explore using NLAs for evaluation of probes, to see what, in natural language, the probe is activating.

Plan overview:

### ✅ Phase 1 — Replicate Debunova probe + format-confound findings

### ⏳ Phase 2 — Validate NLA as a characterization tool on toy reward scenario

### Phase 3 — Use NLAs to verbalize the Devbunova confound

### Phase 4 (TBD) — Behavioral correlation on toy reward

See `research-plan.md` for the full plan and `research-log.md` for progress.

---

## Phase 1 — Replicate Debunova probe + format-confound findings

Devbunova ([2026](https://arxiv.org/abs/2603.19426v1)) showed linear probes for eval-awareness primarily detect **benchmark formatting** rather than genuine awareness. Here, we will use the same dataset, and compare the probe and NLA outputs. NLA explanations should key on the question format.

1. Train the LogisticRegression probe on `bench_eval` vs `casual_deploy`,
2. Test it on the four held-out 2x2 cells, checking whether it fires on MCQ *format* rather than evaluation *context*.
3. Verify these results (using Llama-3.3-70B) match Debunova's (using Llama-3.1-8B).

### Setup

```bash
pip install -r requirements.txt
modal token new                        # one-time auth
cp .env.example .env                   # fill in HF_TOKEN
```

`HF_TOKEN` needs access to the gated `meta-llama/Llama-3.3-70B-Instruct` model and the `viliana-dev/eval-awareness-2x2` dataset.

### Run

```bash
python scripts/reproduce_format_confound.py                          # default layer sweep [23, 40, 53, 63]
python scripts/reproduce_format_confound.py --layers 23,53           # custom layers
python scripts/reproduce_format_confound.py --run-name baseline      # label appended to output filenames
```

This script does the following:

- Initilizes cloud GPUs spot instances (on Modal, using 2x H100s by default)
- Loads Llama-3.3-70B-Instruct (caches weights on `nla-labelled-probes-hf-cache` volume for future runs)
- Trains the probe
- Prints results to stdout in a per-cell table
- Saves outputs to `data/results` and `data/probes`
  - `data/results/phase1_2x2_scores_<tag>.json` — per-cell scores + config
  - `data/probes/phase1_devbunova_logreg_<tag>.json` — trained probe (LR weights + scaler)

**Training Devbunova's probe.**  
For the two training cells from `viliana-dev/eval-awareness-2x2` — `bench_eval` (class 0 = eval, MCQ benchmark prompts) and `casual_deploy` (class 1 = deploy, casual ShareGPT queries) — the worker:

- tokenizes the **raw** prompts (no chat template, no system prompt; BOS added by tokenizer),
- runs a forward pass and captures the **last-token** activation at the *output* (resid_post) of every requested layer,
- per layer, fits a `StandardScaler` on the training activations and trains a `LogisticRegression` (max_iter=1000, seed=42) on the scaled features.

### Prompt shapes

Devbunova feeds raw prompts; the only special token is the BOS the tokenizer prepends. The activation is the residual stream leaving the layer (`resid_post`) at the last real token.

**Training class 0 — `bench_eval` (MCQ benchmark):**

```
Question: What is the capital of Canada?
Choices: <A, B, C, D>
Answer:
```

**Training class 1 — `casual_deploy` (casual ShareGPT first turn):**

```
What is the capital of Canada?
```

**Test cells** are the same shape as their format implies — MCQ-formatted cells (`bench_deploy`, `bench_deploy_formal`) carry the `Question:/Choices:/Answer:` scaffold; free-form cells (`casual_eval`, `casual_deploy_formal`) do not. This is the discriminating  
test for the confound: if the probe keys on MCQ structure, the MCQ test cells fire ~90%+ "eval" despite being deployment-context prompts.