# Research Log

---

**Instructions**

- New dated entries go at the top. 
- Don't restructure old entries.
- Be concise.

---

### May 20, 2026 — Phase 1 replication SUCCEEDS (Devbunova method)

First run of the faithful Devbunova pipeline (run tag
`20260520T170144_debunova-style-probes-first-run`). The format confound replicates cleanly on
Llama-3.3-70B, matching her Llama-3.1-8B numbers both qualitatively and quantitatively:

Ours — Llama-3.3-70B (% classified EVAL):

| cell (context)            | format | L23  | L40  | L53  | L63  |
|---------------------------|--------|------|------|------|------|
| casual_eval (eval)        | free   | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy (deploy)     | mcq    | 95.1 | 95.0 | 94.6 | 94.5 |
| casual_deploy_formal (dep)| free   | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy_formal (dep) | mcq    | 94.1 | 94.0 | 93.8 | 93.7 |

Devbunova — Llama-3.1-8B (her RESULTS.md, % classified EVAL):

| cell (context)            | format | L8   | L16  | L24  | L30  | L31  |
|---------------------------|--------|------|------|------|------|------|
| casual_eval (eval)        | free   | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy (deploy)     | mcq    | 94.3 | 94.3 | 94.3 | 94.3 | 94.3 |
| casual_deploy_formal (dep)| free   | 0.1  | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy_formal (dep) | mcq    | 90.1 | 89.9 | 88.5 | 89.7 | 90.2 |

Same pattern, near-identical magnitudes (her layer indices differ — 8B has 32 layers vs the
70B's 80). Train acc 100% at every layer in both (perfectly linearly separable, as she found). MCQ format →
~94% "eval" regardless of eval/deploy context or casual/formal style; free-form → 0%. Style
is not the driver (casual vs formal both ~0% free / ~94% mcq). Layer-robust across 23-63.

Conclusion: the probe is a format detector, not an eval-context detector — Devbunova's
finding reproduces on the NLA target model. Phase 1 deliverable met; clear to proceed to
Phase 2 (validate NLA-as-characterization on toy reward). L53 (the NLA layer) behaves like
the others, which is what Phase 3 needs.

### May 20, 2026 — Phase 1 re-run + Devbunova method correction

Re-ran the Nguyen-reconstruction pipeline after the calibration fix. The bug is gone — scores
now spread across the range (~13–78%) instead of saturating to 0/100. But the Nguyen-style
contrastive probe shows **no MCQ-vs-free split**: e.g. L23 last-token has MCQ `bench_deploy`
at 67% vs free `casual_deploy_formal` at 77%, and free `casual_eval` is the *lowest* (13%).
So a properly-calibrated Nguyen contrastive probe does not reproduce Devbunova's format
confound on Llama-3.3-70B.

Bigger correction, from reading her actual code
(`debunova-2026-eval-awareness-format/scripts/evaluate_probe_cached.py`): **Devbunova never
uses Nguyen's probe.** She trains her own sklearn `LogisticRegression` on `bench_eval`
(class 0 = eval) vs `casual_deploy` (class 1 = deploy), with `StandardScaler` fit on train,
on **last-token** activations from the layer *output* (resid_post, 0-based), from **raw**
prompts (no chat template, no system prompt), and reports % predicted class 0 on the four
held-out test cells. Our Phase 1 reconstructed Nguyen's difference-of-means probe — a
different probe, training data, normalization, and prompt space.

Decision (with Jared): rebuild Phase 1 to match her method exactly; only model + layers
differ. The Nguyen reconstruction is dropped from the replication.

Rebuilt: `src/modal_app.py` now trains her LogisticRegression (StandardScaler on train,
last-token resid_post via forward hook, raw prompts) on `bench_eval` vs `casual_deploy` and
evaluates the 4 held-out cells; `src/data.py` gained train/test cell constants; the script
and README were updated; `research-plan.md` Phase 1 + Approach corrected. `src/probe.py` and
the Nguyen helpers are retained but unused. Pending: re-run on Modal.

### May 20, 2026 — Phase 1 calibration fix

Fixed the threshold/scale bug from the first run. Root cause: the probe scored *raw*
projections, which carry each dataset's common-mode offset along the probe direction (it
grows with depth and differs between Nguyen's contrastive set and Devbunova's 2x2), so the
train-derived threshold landed far off-scale on the 2x2 set.

Fix: per-dimension z-scoring, with the standardizer **fit separately per dataset**.
- Probe direction + midpoint threshold are built in the contrastive set's z-space.
- The 2x2 set is standardized using stats fit on the *combined 2x2 pool* (mean-token and
  last-token pools each get their own standardizer), so each cell's score is relative to the
  2x2 set's own mean. Centering per-dataset removes the offset; the threshold then measures
  relative eval-vs-deploy separation — the actual confound question.
- Per-dim standardizing with *train* stats alone was insufficient (a synthetic test confirmed
  it leaves a cross-dataset offset); fitting per dataset is what makes the threshold transfer.

Code: `src/probe.py` (added `fit_standardizer`, `standardize`); `src/modal_app.py`
(`_build_probe_vectors` builds in z-space + stores mean/std; `_score_2x2` now does a two-pass
gather-then-score so it can fit the standardizer on the full 2x2 pool). Probe artifact gains
`mean`/`std` per layer. Validated locally on synthetic data; needs a real re-run on Modal.

### May 20, 2026 — Phase 1 first run

First end-to-end run on Llama-3.3-70B-Instruct, chat-templated, layers [23,40,53,63].
**Did not reproduce Devbunova's format confound.** Results are dominated by a probe
calibration bug, not by prompt format.

What the scores show (% classified EVAL, mean-token / last-token):
- **L23:** ~99–100% across *all* cells (MCQ and free alike) — saturated high.
- **L40:** 38–87% mean / 41–96% last, no clean MCQ-vs-free split (free `casual_eval` is the
  *lowest*, opposite of the expected pattern).
- **L53, L63:** 0.0% everywhere — saturated low.

Root cause = threshold/scale mismatch, not signal. The decision thresholds are derived from
projections of Nguyen's contrastive training set, but the 2x2 activations project to a very
different magnitude range:

| layer | threshold | 2x2 mean-token score range (min/med/max) |
|------|----------:|------------------------------------------|
| L23  | −0.013    | −0.089 / 0.089 / 0.243 |
| L40  |  0.007    | −0.138 / 0.037 / 0.195 |
| L53  |  2.394    | −0.571 / −0.238 / 0.089 |
| L63  |  3.611    | −1.047 / −0.518 / −0.075 |

At L53/L63 the threshold (2.4, 3.6) sits far above the entire 2x2 score range → everything
lands below → 0% eval. At L23/L40 the threshold is ≈0 with scores straddling it → near-binary
saturation. So the probe is reading scale, not content. The non-normalized difference-of-means
direction projects training vs 2x2 activations onto incompatible scales (likely a centering /
activation-norm mismatch between the two datasets).

**Next:** fix calibration before drawing any conclusion — center/normalize activations
consistently across train and eval (e.g. unit-norm activations or standardize the projection),
or set the threshold from the 2x2 distribution. Only after the probe is on-scale can we judge
whether the MCQ/free confound reproduces.



### May 14, 2026 — Phase 1 implementation

Built the Phase 1 pipeline (reconstruct Nguyen's probe, apply to Devbunova's 2x2).

- **Infra change:** dropped NDIF/NNsight (no access) for Modal.com on-demand GPU instances.
  Updated `research-plan.md` Infrastructure + Workflow sections accordingly.
- **Nguyen's probe — no released weights.** Their anonymized repo de-anonymized to
  `github.com/Jordine/evaluation-awareness-probing`; it ships a contrastive dataset (204
  eval-awareness Q&A pairs) + generation code, not saved probe weights. The Simple
  Contrastive probe is a difference-of-means direction, so Phase 1 *reconstructs* it — which
  is better than downloaded weights anyway, since it lets us build the probe at any layer
  (including L53, the Neuronpedia NLA layer). Vendored their `contrastive_dataset.json`.
- **Decisions baked in (configurable + documented):** layer sweep `[23, 40, 53, 63]`;
  scoring reports both mean-token (Nguyen) and last-token (Devbunova) aggregations;
  2x2 prompts chat-templated by default (matches the probe's activation space), `--raw`
  flag to match Devbunova exactly.
- Code: `src/{data,probe,modal_app}.py`, `scripts/reproduce_format_confound.py`. Imports +
  probe logic smoke-tested locally. Running the experiment is pending Modal auth + HF access.

### May 14, 2026 — Getting setup

This paper was just released, and opens up an interesting path to look at eval awareness:
https://transformer-circuits.pub/2026/nla/index.html
tldr of that: NLAs (natural language autoencoders) output a text representation of the models latent representation. 
Here’s a demo: https://www.neuronpedia.org/llama3.3-70b-it/nla

The idea with this research:

1. Reproduce Debunova's eval awareness format confound paper (her paper shows that those eval-awareness linear probes respond to prompt format, not the content)
2. Extend it, exposing the same format confound using NLAs
3. Then, if that works well, explore using the output of an NLAs to generate labels to train better linear probes 
- NLA inference is very expensive; however, probes trained off NLA output could be usable production/further research.
- Most likely source of this part to fail is that current NLAs provide a noisy/uncalibrated signal. Step 2 should help get a sense of this. 
- If it’s too noisy, a more complex project (that anthropic is likely already working on) is to train a better NLA (using more than 1 activation layer as input). This would have to be a proof of concept on a small model, as this would be extremely expensive to train.

1 and 2 derisk things. 3 gets much more involved; and potentially much more interesting.
