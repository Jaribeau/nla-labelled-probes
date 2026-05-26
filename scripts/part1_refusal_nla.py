"""Part 1 — Does the NLA verbalize refusal?

Pipeline:
  1. Sample Arditi train/val + a balanced held-out test set (src/data_refusal.py).
  2. Extract last-token resid_post at layers {40,53,63} + refusal scores (src/target_model.py).
  3. Filter train by refusal score (harmful>0, harmless<0), build per-layer diff-of-means
     direction (src/probe.py), report held-out AUROC per layer (sanity); commit L53.
  4. Verbalize held-out L53 activations with the NLA AV (src/nla_modal.py).
  5. Keyword-grade explanations; save everything for later LLM-judge re-grading.

Run:  python scripts/part1_refusal_nla.py [--n-heldout 100] [--run-name tag]
"""
import argparse
import datetime
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src import probe as P
from src.data_refusal import sample_heldout, sample_train_val

LAYERS = [40, 53, 63]
NLA_LAYER = 53
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "probes")

REFUSAL_RE = re.compile(
    r"\b(refus\w*|declin\w*|cannot|can't|won't|will not|unsafe|harmful|harm|safety|"
    r"illegal|unethical|inappropriate|dangerous|sorry|apolog\w*|policy|comply|"
    r"compliance|disallow\w*|reject\w*)\b",
    re.IGNORECASE,
)


def auroc(scores, labels):
    """Rank-based AUROC: P(score | label=1 > score | label=0). No sklearn dependency."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Tie-averaged ranks (1-based), then Mann-Whitney U / (|pos|*|neg|).
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = (csum - counts + 1 + csum) / 2.0  # average rank per unique value
    ranks = avg_rank[inv]
    return (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def extract_all(harmful_train, harmless_train, heldout):
    from src.target_model import TargetModel, app

    with app.run():
        tm = TargetModel()
        h = tm.extract.remote(harmful_train, LAYERS)
        l = tm.extract.remote(harmless_train, LAYERS)
        ho = tm.extract.remote(heldout, LAYERS)
    return h, l, ho


def verbalize(vectors, max_new_tokens=256):
    from src.nla_modal import NLAVerbalizer, app

    with app.run():
        return NLAVerbalizer().verbalize.remote(vectors, max_new_tokens=max_new_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heldout", type=int, default=100, help="per class in held-out set")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="NLA explanation length cap")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        existing = _cli.list_runs("part1_refusal_nla_*.json", r"part1_refusal_nla_(.+)\.json")
        if existing:
            print("Existing Part 1 runs: " + ", ".join(r for r, _ in existing[:8]))
        args.run_name = _cli.ask_str("Run-name suffix (blank = timestamp only)", "") or None
        args.n_heldout = _cli.ask_int("Held-out examples per class", args.n_heldout)
        args.max_new_tokens = _cli.ask_int("NLA explanation length cap", args.max_new_tokens)

    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.run_name:
        run_id += f"_{args.run_name}"

    tv = sample_train_val()
    heldout, labels = sample_heldout(n_each=args.n_heldout)
    labels = np.array(labels)

    print(f"Extracting: {len(tv['harmful_train'])} harmful_train, "
          f"{len(tv['harmless_train'])} harmless_train, {len(heldout)} held-out...")
    h_res, l_res, ho_res = extract_all(tv["harmful_train"], tv["harmless_train"], heldout)

    # Refusal-score filter (Arditi): keep harmful that refuse (>0), harmless that comply (<0).
    h_keep = h_res["refusal_scores"] > 0
    l_keep = l_res["refusal_scores"] < 0
    print(f"Filter kept {int(h_keep.sum())}/{len(h_keep)} harmful, "
          f"{int(l_keep.sum())}/{len(l_keep)} harmless train examples.")

    probes, layer_report = {}, {}
    ho_scores = {}
    for L in LAYERS:
        sl = str(L)
        h_acts = h_res["acts"][sl][h_keep].astype(np.float64)
        l_acts = l_res["acts"][sl][l_keep].astype(np.float64)
        direction = P.build_probe(h_acts, l_acts)  # mean(harmful) - mean(harmless)
        threshold = P.midpoint_threshold(h_acts, l_acts, direction)
        scores = P.project(ho_res["acts"][sl].astype(np.float64), direction)
        ho_scores[L] = scores
        layer_report[L] = {"auroc": auroc(scores, labels), "threshold": float(threshold)}
        probes[L] = {"direction_unit": P.unit(direction).tolist(), "threshold": float(threshold)}
        print(f"  L{L}: held-out AUROC={layer_report[L]['auroc']:.3f}")

    # Commit L53 for the NLA comparison.
    scores53 = ho_scores[NLA_LAYER]
    thr53 = layer_report[NLA_LAYER]["threshold"]
    fires = scores53 > thr53
    print(f"L{NLA_LAYER}: {int(fires.sum())}/{len(fires)} held-out prompts fire the probe.")

    print("Verbalizing held-out L53 activations via NLA AV...")
    l53_vectors = ho_res["acts"][str(NLA_LAYER)].astype(np.float32)
    explanations = verbalize([v for v in l53_vectors], max_new_tokens=args.max_new_tokens)

    examples = []
    for i, instr in enumerate(heldout):
        hits = sorted(set(m.group(0).lower() for m in REFUSAL_RE.finditer(explanations[i])))
        examples.append({
            "instruction": instr,
            "label": int(labels[i]),  # 1 harmful, 0 harmless
            "score_l53": float(scores53[i]),
            "fires": bool(fires[i]),
            "explanation": explanations[i],
            "refusal_keyword_hits": hits,
            "scores_by_layer": {str(L): float(ho_scores[L][i]) for L in LAYERS},
        })

    # Refusal-mention rate by probe polarity (the Part 1 deliverable).
    def mention_rate(mask):
        m = np.array([bool(e["refusal_keyword_hits"]) for e in examples])[mask]
        return float(m.mean()) if len(m) else float("nan")

    summary = {
        "refusal_mention_rate_probe_positive": mention_rate(fires),
        "refusal_mention_rate_probe_negative": mention_rate(~fires),
        "layer_auroc": {str(L): layer_report[L]["auroc"] for L in LAYERS},
    }
    print("Summary:", json.dumps(summary, indent=2))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PROBES_DIR, exist_ok=True)
    result = {
        "run_id": run_id,
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "nla_checkpoint": "kitft/Llama-3.3-70B-NLA-L53-av",
        "layers": LAYERS,
        "nla_layer": NLA_LAYER,
        "config": {"n_heldout_per_class": args.n_heldout, "extraction": "last-token resid_post"},
        "layer_report": {str(L): layer_report[L] for L in LAYERS},
        "summary": summary,
        "examples": examples,
    }
    res_path = os.path.join(RESULTS_DIR, f"part1_refusal_nla_{run_id}.json")
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    probe_path = os.path.join(PROBES_DIR, f"part1_refusal_dir_{run_id}.json")
    with open(probe_path, "w") as f:
        json.dump({"run_id": run_id, "layers": LAYERS, "probes": {str(L): probes[L] for L in LAYERS}}, f)

    # Persist raw held-out activations (all layers) so Part 2 can re-label via an LLM judge
    # and train the NLA-label diff-of-means probe with NO re-extraction. Order matches
    # `examples` in the results JSON. float16 to keep it small (~10MB at n=100).
    acts_path = os.path.join(RESULTS_DIR, f"part1_refusal_acts_{run_id}.npz")
    np.savez_compressed(
        acts_path,
        labels=labels.astype(np.int8),
        score_l53=scores53.astype(np.float32),
        fires=fires.astype(bool),
        **{f"L{L}": ho_res["acts"][str(L)].astype(np.float16) for L in LAYERS},
    )
    print(f"Saved {res_path}\nSaved {probe_path}\nSaved {acts_path}")


if __name__ == "__main__":
    main()
