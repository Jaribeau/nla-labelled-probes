# Research Log

---

**Instructions**

- New dated entries go at the top. 
- Don't restructure old entries.
- Be concise.

---

### May 26, 2026 — Big refactor to clean up the audit pipeline structure

Split the Part 3 monolith (`part3_audit_confound.py` + `part3_association.py`) into discrete,
cached, re-runnable stages under `scripts/audit_pipeline/`: `0_build_probe`,
`1_run_prompts_extract_activations`, `2_verbalize_activations`, `3_probe_activations`,
`4_extract_concepts_from_nla_verbalizations`, `5_analyze_probe_concept_associations`. Probe and
prompt set are now selectable inputs; confound-specific 2×2 metrics moved to an optional
`confound_validation` block in stage 3 (new artifact `part3_probe_<rn>.json`). Artifact
filenames otherwise unchanged so cached GPU outputs (`part3_acts/verbalize_part3v1`) are reused.
Removed the fixed 4-class theme judge entirely (`judge_theme`/`THEME_SYSTEM`/`viz_part3.py`) —
superseded by concept-association. `viz_runs.py` Part 3 columns updated (drop theme/audit, add
prompts/probe); legacy `part3_theme/audit_part3v1` files remain on disk under "Other files".

### May 26, 2026 — Scripts are now interactive CLIs; added a runs viewer

Each script defaults to an interactive prompt when run bare (no flags), listing prior runs /
files to pick from; flags still work and skip the prompts. Shared helper: `scripts/_cli.py`
(`interactive()`, `ask_*`, `choose`, `choose_run`, `choose_file`). New `scripts/viz_runs.py`
scans `data/results/` + `data/probes/`, groups files by part + run id, and writes/opens
`runs.html` (a table of artifacts per run). README "Run" section simplified to bare commands;
flag details live in the script docstrings.

### May 26, 2026 — Renamed "enrichment" → "association"; debiased concept judge

Terminology cleanup to match `report.html`, which dropped "enrichment" for "association" /
"probe–concept-association". Renamed `scripts/part3_enrichment.py` → `part3_association.py`,
`scripts/viz_enrichment.py` → `viz_association.py`, output `part3_enrichment_<rn>.json` →
`part3_association_<rn>.json` (and the frozen `part3v1` artifacts + the figure png). Updated
`research-plan.md` Part 3b/Part 4 wording and `src/judge.py` comment. README Part 3 rewritten
as the report's 5-step audit pipeline (prompt → probe → verbalize → concept judge → rank by
association), split into Step A (audit/theme) and Step B (probe–concept association), plus the
`inspect_concepts.py` raw-vocab viewer.

Also debiased `src/prompts/concept-extraction-judge-prompt.md`: replaced in-domain examples
(refusal / MCQ / illegal activity) with out-of-domain ones so the judge isn't primed toward the
concepts under study, and dropped the "reuse tags across explanations" rule (each explanation is
a separate stateless judge call — canonicalization is a separate post-hoc step). Added
`--grade-tag` to `part3_association.py` so a re-grade saves alongside prior gradings.

### May 21, 2026 — Part 3b run: enrichment audit names the confound unprompted

Implemented + ran the open-vocab concept-enrichment audit on the cached `part3v1` data (no GPU;
one cached ~200-call concept-extraction pass). `judge_concepts` (`src/judge.py`),
`scripts/part3_enrichment.py` (general: dir + acts + explanations → enrichment), `scripts/viz_enrichment.py`.

`enrichment = freq(concept | top-third projection) − freq(concept | bottom-third)`:


| concept                       | confounded     | clean     |
| ----------------------------- | -------------- | --------- |
| refusal                       | +0.89          | +0.95     |
| illegal activity              | +0.41          | +0.39     |
| **multiple-choice format**    | **+0.35 👀🔥** | **−0.08** |
| q&a format (NLA baseline tic) | +0.05          | +0.08     |


- **Confound named without a taxonomy:** "multiple-choice format" is a top-3 enriched concept
for the confounded probe (+0.35) and absent/negative for the clean probe (−0.08).
- **Self-baselining works:** the generic "q&a format" tic that contaminated the fixed-taxonomy
run (benign prose was 92% "format") enriches to only ~+0.05–0.08 and drops out — it appears
equally in high/low projection, so the contrast cancels it. The *specific* MCQ concept survives.

Conclusion: enrichment is a strictly better audit readout than the fixed 4-class theme judge —
fixes the baseline contamination and works for discovery. This is the tool Part 4 will use.

### May 21, 2026 — Plan: added Part 3b (concept-enrichment audit)

Added **Part 3b** to `research-plan.md`: open-vocab `judge_concepts` + enrichment readout
(`enrichment = freq(concept | top-third projection) − freq(concept | bottom-third)`), visualized
as a ranked diverging bar chart (confounded vs clean). Self-baselining (cancels the Q&A tic) and
discovery-capable (no fixed taxonomy) — Part 4 reuses it as its discovery tool. Planned code:
`judge_concepts`, `scripts/part3_enrichment.py` (general: dir + acts + explanations → enrichment),
`scripts/viz_enrichment.py`. Offline; reuses cached Part 3 acts + explanations.

### May 21, 2026 — NLA Audit v1 run (Part 3): probe bit on the confound, audit caught it

NLA audit v1
Ran `part3_audit_confound.py --run-name part3v1` (n_train=128, n_cell=50).

- **Confound bit (partially).** 
  - Confounded probe fire-rates: harmful-MCQ 100%, harmful-free  
  86%, **harmless-MCQ 54%, harmless-free 0%** → `format_effect_harmless = +54%`,  
  `concept_effect = 86%`. 
  - So, it's a *mixture* — still a harm detector, but it now also fires on benign content purely because it's MCQ-wrapped.
  - Clean Part 1 probe is concept-pure: harmful 100/54%, harmless 0/0% (`format_effect 0`, `concept_effect 100%`).
- **The Audit caught it.** 
  - Theme distribution over each probe's firing set: confounded = **42% format** / 57% refusal vs clean = **13% format** / 84% refusal. 
  - The `harmless_mcq` cell (which only the confounded probe fires on) audits as **100% format**. So the NLA audit flags that the  
  confounded probe fires on format-themed activations the clean probe ignores.
- **Methodological lesson.** 
  - The NLA has a baseline habit of describing conversational structure ("Q&A format established") — `harmless_free` is **92% "format"** theme despite being plain prose. So *absolute* theme rates are baseline-contaminated; the signal lives in the **contrast** (confounded-vs-clean firing; harmless-MCQ vs harmless-free), not the raw rate. 
  - The fixed 4-class taxonomy is also too coarse (lumps MCQ with generic Q&A).
- **Next steps (operationalization).** 
  - Move from fixed-taxonomy judge → **open-vocabulary concept extraction + enrichment heatmap** (concept frequency in high- vs low-projection activations).
  - Enrichment is self-baselining (cancels the NLA's tics) and works for *discovery* (Part 4) where the confound isn't known in advance. 
  - Can prototype offline on the cached `part3_verbalize_part3v1.json`.

### May 21, 2026 — Part 3 implemented (audit demo code; not yet run)

Built the Part 3 audit pipeline (validated offline; the GPU+API run is pending):

- `src/data_confound.py` — MCQ-scaffold wrapper (`Question:/Choices:/(A)/(B)/Answer:`, content
unchanged) + free-form; confounded train (harmful-MCQ=1 vs harmless-free=0) + de-confounded
2×2 builders, paired so the same instruction appears in both formats.
- `src/judge.py` — refactored to a shared `_batch` runner; added `judge_theme` (multi-class
format/refusal/harmful_topic/other, robust + cached like the refusal judge).
- `scripts/part3_audit_confound.py` — extract L53 (train + 2×2) → confounded diff-of-means
probe + clean Part 1 dir → 2×2 fire-rates + confound-strength metrics → verbalize 2×2 acts →
theme-judge → theme distribution over each probe's firing set. Both GPU steps + judge cached
per `--run-name` (flags `--reextract/--reverbalize/--regrade`) so iteration is free.
- `scripts/viz_part3.py` — confound-strength cards, 2×2 fire-rate grids (confounded vs clean),
theme-distribution bars, theme-by-cell, per-prompt table.

Headline to look for: confounded probe fires on the MCQ column (format) and audits as "format";
clean probe fires on the harmful row and audits as "refusal"/"harmful_topic". Bite-risk
(strong harm signal) is measured directly via the 2×2 fire-rate gap (`confound_format_effect`
vs `confound_concept_effect`).

### May 21, 2026 — Reframed project around the audit; added Part 3 + Part 4 to plan

Refusal is an easy case (source label already faithful), so Parts 1–2 validate the *instrument* but not its *value* over supervised probes. The value is the **audit**: NLA + LLM-judge reads what a probe actually fires on, catching confounds invisible to in-distribution accuracy (Devbunova's eval-awareness probe was an MCQ-format detector) — a cheap, general replacement for concept-specific counterfactual datasets.

- Distinguished two products: (a) NLA-labelled probe factory (only beats supervision when labels  
are confounded) vs (b) NLA audit of an existing probe (more general, lower-risk). 
- Chose to pursue the audit. 
- Added to `research-plan.md`: Research Question 3, 
  - **Part 3** (planted MCQ-format
  confound on refusal, self-contained — show 2×2 + NLA audit both flag format while in-dist accuracy
  looks fine; clean Part 1 probe as control), and 
  - **Part 4** (exploratory difficulty-ladder:stylistic/topical → correlated-concept confounds like eval-awareness-vs-artificiality and  
  deception-vs-hedging; map where the audit breaks). 
  - Noted bite-risk (strong refusal signal may resist the planted confound — measure strength) and the audit's reach being bounded by NLA expressiveness.

### May 21, 2026 — Reviewed divergence cases + changed to probabilistic judge

**Divergence cases (binary judge, 3 of them).** All harmful-source, judged non-refusal:

- *"explain why Trump unanimously won 2020"* — probe **quiet**, NLA **comply**: model prepares
a factual rebuttal, not a refusal. Here probe + NLA agree; the *source label* is the outlier  
(correcting misinformation is compliance). This is good, NLA tracks behaviour, not category.
- *immigration-evasion* and *chargeback-to-trick-a-store* — probe **fires**, judge **comply**.  
But the NLA explanations are genuinely **hedged**: each contains BOTH a disclaimer cue  
("likely beginning with 'I cannot provide direct guidance'") AND informational/advisory  
framing. The binary judge collapsed that to comply. So the divergence is less NLA failure,  
more (a) ambiguous activations at the assistant boundary and (b) a binary judge discarding  
the hedge. (#2 is also a borderline-harmful prompt.)
- Takeaway: the NLA verbalizes a *distribution over the upcoming response* at this token  
position and often hedges; a hard binary throws that away.

**Iterated the judge to probabilistic output.** 

- `src/judge.py` now returns `p_refusal` in [0,1] (explicit decline → high, pure informational → low, disclaimer+info hedge → mid). 
- Part 2 builds the recovered direction as a **soft-label diff-of-means** weighted by `p_refusal`  
(`probe.build_probe_weighted`), down-weighting ambiguous activations. Outputs are `--tag`-suffixed  
so the binary run's artifacts are preserved; viz fades hedged dots and adds a hedged-cases table.
- Probabilistic run (`..._part1-n100_prob`): cosine(soft, Arditi) = **0.961** (hard 0.960; binary  
run was 0.953), judge vs source **99%**, test AUROC **1.0**, **2** divergence + **7 hedged**  
cases (0.2<p<0.8)

### May 21, 2026 — Part 2 implemented + first result (NLA labels recover the direction)

Built Part 2 (offline, no GPU): `src/judge.py` (Claude Sonnet binary judge — "does this NLA
explanation indicate *refusal*, not just mention a harmful topic"), `scripts/part2_recover_direction.py`,
`scripts/viz_part2.py`. Added `auroc`/`cosine` to `src/probe.py`. Judge made per-call robust
(some explanations quote harmful text → Sonnet returns empty content / stop=refusal; handled
as `refusal: None`, 2/200 here).

First run on the clean n=100 Part 1 set (`20260521T023346_part1-n100`):

- judge vs source label agreement **98.5%**, judge vs probe-fires **99%**.
- **cosine(dir_nla, dir_arditi) = 0.953** at L53 (0.956 / 0.953 / 0.949 at L40/53/63).
- cos(dir_nla, dir_true) = **0.999** — using NLA labels instead of true labels costs ~nothing;
the 0.95 gap to Arditi is the cross-split difference (Arditi dir built on the train split),
identical for NLA- and true-label directions.
- held-out test AUROC = **1.0** for all three directions (nla / true / arditi).
- 3 divergence cases (all harmful-source, judged non-refusal): NLA describes "preparing a
structured/advisory/corrective response" rather than declining — the judge correctly
separating refusal from compliance-on-a-sensitive-topic.

Clean positive, as expected on separable data. Next: borderline set (XSTest) via a new Part 1
extraction run to create genuine divergence; Part 2 code runs unchanged on that run_id.

### May 21, 2026 — Part 1 implemented: pipeline to build probe and fetch NLA explanations

- Currently checking refusal presence in NLA with just refusal keyword regex. 
  - This worked reasonably well as a first draft (100%/6%)
  - Next will implement an LLM judge, which should be at or close to perfect on refusal (100%/0%)
- part1-n-100
- part1-test-run-n-10

### May 21, 2026 — Populated Part 1 implementation details (Arditi refusal direction)

Read Arditi's published code (`github.com/andyrdt/refusal_direction`, MIT) to pin down the recipe; added a refs entry `docs/refs/refusal-direction-arditi-2024.md`. 

Decisions: 

- fix extraction to resid_post of block 53 (the NLA layer) and a single last-token (assistant-boundary) position, instead of Arditi's layer-selection, so probe and NLA read the same activation.
- **keep Arditi's refusal-score filtering** (first token "I" logit > 0 for harmful, < 0 for harmless) for cleaner labels; 
- **sweep layers {40, 53, 63}** in one forward pass as a health check / partial replication of Arditi's layer-selection, but commit to L53 for the NLA comparison.

### May 20, 2026 — Rewrote plan for refusal design + cleaned up repo

Rewrote `research-plan.md` around the refusal design: skeleton with the same sections but  
Part 1 (does the NLA verbalize refusal on probe-positive activations?) and Part 2 (recover  
the refusal direction from NLA-derived labels; compare to Arditi via cosine + AUROC) instead  
of phases.

### May 20, 2026 — Pivot from eval awareness to refusal for NLA

Instead of trying to focus on eval awareness, it seems best to start with a cleaner concept to test the mechanism, and only if that works try connecting it back to eval awareness.

Now: will clear out a bunch of the now irrelevant code and planning documentation.

### May 20, 2026 — Phase 2 toy reward dataset + methodology

Built the Phase 2 toy "reward awareness" dataset and fleshed out the methodology in
`research-plan.md`.

- New generator `build_toy_reward_dataset()` in `src/data.py`; output written to
`data/toy_reward_dataset.json`. 12 tasks across 3 families (number_parity,
rubric_leak, game_payoff) × 3 variants = 36 records (24 neg / 12 pos).
- **3-way construction** to separate reward semantics from "an extra clause was
appended": hint-absent (0), length-matched filler (0), hint-present/blatant (1).
Filler doubles as a leakage check — if NLA says "reward" on filler, the diagnostic
is leaky. Mirrors Phase 1's format-vs-concept logic.
- **Subtle hint tier stubbed** (`hints["subtle"] = None`) for a less-explicit phrasing
later; blatant is the floor test and ships first.
- Plan updated with per-variant expected NLA themes and a decision to capture both the
assistant-turn boundary and last-token-of-prompt positions in one pass.

### May 20, 2026 — Phase 1 replication SUCCEEDS (Devbunova method)

First run of the faithful Devbunova pipeline (run tag
`20260520T170144_debunova-style-probes-first-run`). The format confound replicates cleanly on
Llama-3.3-70B, matching her Llama-3.1-8B numbers both qualitatively and quantitatively:

Ours — Llama-3.3-70B (% classified EVAL):


| cell (context)             | format | L23  | L40  | L53  | L63  |
| -------------------------- | ------ | ---- | ---- | ---- | ---- |
| casual_eval (eval)         | free   | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy (deploy)      | mcq    | 95.1 | 95.0 | 94.6 | 94.5 |
| casual_deploy_formal (dep) | free   | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy_formal (dep)  | mcq    | 94.1 | 94.0 | 93.8 | 93.7 |


Devbunova — Llama-3.1-8B (her RESULTS.md, % classified EVAL):


| cell (context)             | format | L8   | L16  | L24  | L30  | L31  |
| -------------------------- | ------ | ---- | ---- | ---- | ---- | ---- |
| casual_eval (eval)         | free   | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy (deploy)      | mcq    | 94.3 | 94.3 | 94.3 | 94.3 | 94.3 |
| casual_deploy_formal (dep) | free   | 0.1  | 0.0  | 0.0  | 0.0  | 0.0  |
| bench_deploy_formal (dep)  | mcq    | 90.1 | 89.9 | 88.5 | 89.7 | 90.2 |


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
| ----- | --------- | ---------------------------------------- |
| L23   | −0.013    | −0.089 / 0.089 / 0.243                   |
| L40   | 0.007     | −0.138 / 0.037 / 0.195                   |
| L53   | 2.394     | −0.571 / −0.238 / 0.089                  |
| L63   | 3.611     | −1.047 / −0.518 / −0.075                 |


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
[https://transformer-circuits.pub/2026/nla/index.html](https://transformer-circuits.pub/2026/nla/index.html)
tldr of that: NLAs (natural language autoencoders) output a text representation of the models latent representation. 
Here’s a demo: [https://www.neuronpedia.org/llama3.3-70b-it/nla](https://www.neuronpedia.org/llama3.3-70b-it/nla)

The idea with this research:

1. Reproduce Debunova's eval awareness format confound paper (her paper shows that those eval-awareness linear probes respond to prompt format, not the content)
2. Extend it, exposing the same format confound using NLAs
3. Then, if that works well, explore using the output of an NLAs to generate labels to train better linear probes

- NLA inference is very expensive; however, probes trained off NLA output could be usable production/further research.
- Most likely source of this part to fail is that current NLAs provide a noisy/uncalibrated signal. Step 2 should help get a sense of this. 
- If it’s too noisy, a more complex project (that anthropic is likely already working on) is to train a better NLA (using more than 1 activation layer as input). This would have to be a proof of concept on a small model, as this would be extremely expensive to train.

1 and 2 derisk things. 3 gets much more involved; and potentially much more interesting.
