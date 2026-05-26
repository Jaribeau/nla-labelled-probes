"""Part 2 — Recover the refusal direction from NLA-derived labels.

Offline / no GPU. Reads a Part 1 run's artifacts (NLA explanations + held-out activations +
Arditi direction), labels each activation by whether the NLA explanation indicates refusal
(LLM judge, src/judge.py), builds a diff-of-means direction from those labels, and compares
it to Arditi's direction: cosine of directions + AUROC agreement on a held-out test split.

Run:
  python scripts/part2_recover_direction.py [--run-id <tag>] [--pilot 10] [--regrade]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src import probe as P
from src.judge import MODEL, judge_refusal

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "probes")
LAYERS = [40, 53, 63]
NLA_LAYER = 53


def resolve_run_id(run_id):
    files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "part1_refusal_nla_*.json"))]
    if run_id:
        files = [f for f in files if run_id in f]
    if not files:
        sys.exit(f"No part1_refusal_nla_*.json matching run_id={run_id!r}")
    latest = max(files, key=os.path.getmtime)
    # part1_refusal_nla_<runid>.json -> <runid>
    return os.path.basename(latest)[len("part1_refusal_nla_"):-len(".json")]


def load_run(run_id):
    res = json.load(open(os.path.join(RESULTS_DIR, f"part1_refusal_nla_{run_id}.json")))
    acts = np.load(os.path.join(RESULTS_DIR, f"part1_refusal_acts_{run_id}.npz"))
    dirs = json.load(open(os.path.join(PROBES_DIR, f"part1_refusal_dir_{run_id}.json")))
    return res, acts, dirs


def get_or_run_judge(run_id, explanations, regrade, suffix):
    path = os.path.join(RESULTS_DIR, f"part2_judge_{run_id}{suffix}.json")
    if os.path.exists(path) and not regrade:
        cached = json.load(open(path))
        if len(cached["judgments"]) == len(explanations) and "p_refusal" in cached["judgments"][0]:
            print(f"Using cached judge output {path}")
            return cached["judgments"]
        print("Cached judge output stale/old-format; regrading.")
    print(f"Judging {len(explanations)} explanations with {MODEL}...")
    judgments = judge_refusal(explanations)
    json.dump({"run_id": run_id, "model": MODEL, "judgments": judgments}, open(path, "w"), indent=2)
    print(f"Saved {path}")
    return judgments


def stratified_split(label, test_frac, seed):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in (0, 1):
        idx = np.where(label == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * (1 - test_frac)))
        train += list(idx[:k])
        test += list(idx[k:])
    return np.array(sorted(train)), np.array(sorted(test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="Part 1 run tag (default: latest)")
    ap.add_argument("--pilot", type=int, default=0, help="judge only first N and print; no training")
    ap.add_argument("--regrade", action="store_true", help="ignore cached judge output")
    ap.add_argument("--tag", default="", help="suffix for output files (preserves prior runs)")
    ap.add_argument("--test-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        args.run_id = _cli.choose_run("part1_refusal_nla_*.json",
                                      r"part1_refusal_nla_(.+)\.json", "Part 1 run")
        args.pilot = _cli.ask_int("Pilot: judge only first N (0 = full run + training)", args.pilot)
        args.regrade = _cli.ask_bool("Re-grade (ignore cached judge output)?", args.regrade)
        args.tag = _cli.ask_str("Output tag suffix (blank = none)", args.tag) or ""
        args.test_frac = _cli.ask_float("Test fraction", args.test_frac)
        args.seed = _cli.ask_int("Random seed", args.seed)
    suffix = f"_{args.tag}" if args.tag else ""

    run_id = resolve_run_id(args.run_id)
    print(f"Run: {run_id}")
    res, acts, dirs = load_run(run_id)
    ex = res["examples"]
    explanations = [e["explanation"] for e in ex]
    true_label = acts["labels"].astype(int)
    fires = acts["fires"].astype(bool)

    # --- pilot: judge a handful, eyeball against keyword grader, exit ---
    if args.pilot:
        js = judge_refusal(explanations[: args.pilot])
        for i, j in enumerate(js):
            kw = bool(ex[i]["refusal_keyword_hits"])
            print(f"[{i}] true={true_label[i]} fires={int(fires[i])} keyword={int(kw)} "
                  f"p_refusal={j['p_refusal']}  ({j['reason']})")
        return

    judgments = get_or_run_judge(run_id, explanations, args.regrade, suffix)
    p_raw = np.array([j["p_refusal"] for j in judgments], dtype=object)
    valid = np.array([v is not None for v in p_raw])
    if not valid.all():
        print(f"WARNING: {int((~valid).sum())} explanations unparseable by judge; excluded.")
    idx_valid = np.where(valid)[0]
    p = p_raw[valid].astype(float)              # soft P(refusal) per valid prompt
    nla_label = (p >= 0.5).astype(int)          # hard label for split/diagnostics

    train, test = stratified_split(nla_label, args.test_frac, args.seed)
    train_g, test_g = idx_valid[train], idx_valid[test]  # back to original indices

    report = {}
    directions = {}
    for L in LAYERS:
        A = acts[f"L{L}"].astype(np.float64)
        dir_arditi = np.array(dirs["probes"][str(L)]["direction_unit"])
        # Primary: soft-label diff-of-means weighted by p_refusal (uses the probabilistic signal).
        dir_nla = P.build_probe_weighted(A[train_g], p[train])
        # Secondary: hard 0.5-threshold direction, for comparison.
        nl = nla_label[train]
        dir_nla_hard = P.build_probe(A[train_g][nl == 1], A[train_g][nl == 0])
        dir_true = P.build_probe(A[train_g][true_label[train_g] == 1], A[train_g][true_label[train_g] == 0])
        report[L] = {
            "cosine_nla_arditi": P.cosine(dir_nla, dir_arditi),          # soft
            "cosine_nlahard_arditi": P.cosine(dir_nla_hard, dir_arditi),  # hard 0.5
            "cosine_true_arditi": P.cosine(dir_true, dir_arditi),
            "cosine_nla_true": P.cosine(dir_nla, dir_true),
            "auroc_test_nla": P.auroc(P.project(A[test_g], dir_nla), true_label[test_g]),
            "auroc_test_true": P.auroc(P.project(A[test_g], dir_true), true_label[test_g]),
            "auroc_test_arditi": P.auroc(P.project(A[test_g], dir_arditi), true_label[test_g]),
        }
        directions[L] = {"dir_nla_unit": P.unit(dir_nla).tolist()}
        if L == NLA_LAYER:
            proj_a = P.project(A[test_g], dir_arditi)
            proj_n = P.project(A[test_g], dir_nla)
            scatter = [
                {"proj_arditi": float(proj_a[i]), "proj_nla": float(proj_n[i]),
                 "true_label": int(true_label[g]), "nla_label": int(nla_label[test[i]]),
                 "p_refusal": float(p[test[i]])}
                for i, g in enumerate(test_g)
            ]

    # diagnostics on valid subset (hard label)
    tl = true_label[idx_valid]
    fr = fires[idx_valid].astype(int)
    p_g = {int(g): float(p[k]) for k, g in enumerate(idx_valid)}  # original idx -> p
    diverge = [
        {"instruction": ex[g]["instruction"], "true_label": int(true_label[g]),
         "fires": bool(fires[g]), "nla_label": int(p_g[g] >= 0.5), "p_refusal": p_g[g],
         "reason": judgments[g]["reason"], "explanation": ex[g]["explanation"]}
        for g in idx_valid
        if int(p_g[g] >= 0.5) != int(true_label[g]) or int(p_g[g] >= 0.5) != int(fires[g])
    ]
    hedged = [
        {"instruction": ex[g]["instruction"], "true_label": int(true_label[g]),
         "fires": bool(fires[g]), "p_refusal": p_g[g], "reason": judgments[g]["reason"],
         "explanation": ex[g]["explanation"]}
        for g in idx_valid if 0.2 < p_g[g] < 0.8
    ]
    summary = {
        "n_total": len(ex),
        "n_valid": int(valid.sum()),
        "n_train": len(train_g),
        "n_test": len(test_g),
        "nla_label_pos_rate": float(nla_label.mean()),
        "judge_vs_true_agreement": float((nla_label == tl).mean()),
        "judge_vs_fires_agreement": float((nla_label == fr).mean()),
        "n_divergence_cases": len(diverge),
        "n_hedged_cases": len(hedged),
        "L53_cosine_nla_arditi": report[NLA_LAYER]["cosine_nla_arditi"],
        "L53_cosine_nlahard_arditi": report[NLA_LAYER]["cosine_nlahard_arditi"],
        "L53_auroc_test_nla": report[NLA_LAYER]["auroc_test_nla"],
        "L53_auroc_test_arditi": report[NLA_LAYER]["auroc_test_arditi"],
    }
    print("Summary:", json.dumps(summary, indent=2))
    print(f"L53 cosine(dir_nla_soft, dir_arditi) = {report[NLA_LAYER]['cosine_nla_arditi']:.4f}")

    out = {
        "run_id": run_id,
        "judge_model": MODEL,
        "nla_layer": NLA_LAYER,
        "config": {"test_frac": args.test_frac, "seed": args.seed, "direction": "soft-weighted by p_refusal"},
        "summary": summary,
        "layer_report": {str(L): report[L] for L in LAYERS},
        "directions": {str(L): directions[L] for L in LAYERS},
        "scatter_l53": scatter,
        "p_refusal": [float(v) if v is not None else None for v in p_raw],
        "divergence_cases": diverge,
        "hedged_cases": hedged,
    }
    path = os.path.join(RESULTS_DIR, f"part2_recover_{run_id}{suffix}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
