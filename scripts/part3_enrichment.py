"""Concept-association audit: what concept does a probe actually fire on?

General, offline readout (no GPU). Given a probe direction, the activations it scores, and the
NLA explanations of those activations, it:
  1. extracts open-vocabulary concept tags from each explanation (LLM, cached);
  2. splits activations into top-third vs bottom-third by the probe's projection;
  3. per concept, association = freq(concept | high) − freq(concept | low).

Baseline concepts (the NLA's "Q&A format" tic) appear in both → cancel to ~0 → drop out, so the
top-associated concepts are "what the probe detects" — no pre-specified taxonomy (works for
discovery). Runs for the confounded probe and the clean Part 1 probe on the same explanations.

Reads the cached Part 3 artifacts (`part3_acts_<rn>.npz`, `part3_verbalize_<rn>.json`).
Run:  python scripts/part3_enrichment.py [--run-name part3v1] [--top-frac 0.33] [--min-count 4] [--regrade]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src import probe as P
from src.judge import MODEL, judge_concepts

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "probes")
NLA_LAYER = 53


def resolve_clean_dir(clean_run_id):
    files = glob.glob(os.path.join(PROBES_DIR, "part1_refusal_dir_*.json"))
    if clean_run_id:
        files = [f for f in files if clean_run_id in f]
    if not files:
        sys.exit("No part1_refusal_dir_*.json found")
    f = max(files, key=os.path.getmtime)
    return np.array(json.load(open(f))["probes"][str(NLA_LAYER)]["direction_unit"]), os.path.basename(f)


def get_concepts(run_name, explanations, regrade):
    path = os.path.join(RESULTS_DIR, f"part3_concepts_{run_name}.json")
    if os.path.exists(path) and not regrade:
        cached = json.load(open(path))
        if len(cached["concepts"]) == len(explanations):
            print(f"Using cached concepts {path}")
            return cached["concepts"]
    print(f"Extracting concepts from {len(explanations)} explanations with {MODEL}...")
    judg = judge_concepts(explanations)
    concepts = [j["concepts"] for j in judg]
    json.dump({"run_name": run_name, "model": MODEL, "concepts": concepts}, open(path, "w"), indent=2)
    print(f"Saved {path}")
    return concepts


def association(concepts, proj, top_frac, min_count):
    """Rank concepts by how much more they appear when the probe fires strongly vs weakly.

    For each concept tag, association = freq(concept | high projection) − freq(concept | low
    projection), where "high"/"low" are the top/bottom `top_frac` of activations ranked by the
    probe's projection score. A concept equally common in both tails cancels to ~0 and drops out
    (this is what removes the NLA's baseline tics, e.g. "Q&A format"); a concept the probe truly
    keys on is frequent in the high tail and rare in the low tail → large positive association.

    Args:
        concepts: list (one entry per activation, aligned to `proj` by index) of concept-tag
            lists; an entry may be None (unparseable) and is skipped.
        proj: 1-D array of probe projection scores, one per activation, same index order as
            `concepts`.
        top_frac: fraction of activations taken from each end (e.g. 0.33 → top/bottom third).
        min_count: drop concepts seen in fewer than this many activations overall (noise floor).

    Returns:
        (table, k) where `k` is the per-tail size (max(1, round(n * top_frac))) and `table` is a
        list of dicts sorted by descending association, each:
            {concept, freq_high, freq_low, association (= freq_high − freq_low), count}
        with freq_high/freq_low computed as (tail occurrences) / k.

    Note: a concept appearing in both tails contributes to both counts; ties in `proj` are broken
    by argsort order, and the high/low tails are disjoint by construction.
    """
    n = len(proj)
    k = max(1, int(round(n * top_frac)))
    order = np.argsort(proj)
    low_idx, high_idx = set(order[:k].tolist()), set(order[-k:].tolist())
    # presence sets per concept
    rows = {}
    for i, tags in enumerate(concepts):
        for t in (tags or []):
            r = rows.setdefault(t, {"high": 0, "low": 0, "count": 0})
            r["count"] += 1
            if i in high_idx:
                r["high"] += 1
            elif i in low_idx:
                r["low"] += 1
    out = []
    for t, r in rows.items():
        if r["count"] < min_count:
            continue
        fh, fl = r["high"] / k, r["low"] / k
        out.append({"concept": t, "freq_high": fh, "freq_low": fl,
                    "association": fh - fl, "count": r["count"]})
    out.sort(key=lambda x: -x["association"])
    return out, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="part3v1")
    ap.add_argument("--clean-run-id", default=None)
    ap.add_argument("--top-frac", type=float, default=0.33)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--regrade", action="store_true", help="re-extract concepts (ignore cache)")
    args = ap.parse_args()
    rn = args.run_name

    z = np.load(os.path.join(RESULTS_DIR, f"part3_acts_{rn}.npz"))
    A = z["A2x2"].astype(np.float64)
    explanations = json.load(open(os.path.join(RESULTS_DIR, f"part3_verbalize_{rn}.json")))["explanations"]
    assert len(explanations) == len(A), "explanations / activations length mismatch"

    dir_confound = P.build_probe(z["pos"].astype(np.float64), z["neg"].astype(np.float64))
    dir_clean, clean_src = resolve_clean_dir(args.clean_run_id)
    print(f"Clean probe from {clean_src}")

    concepts = get_concepts(rn, explanations, args.regrade)
    n_bad = sum(c is None for c in concepts)
    if n_bad:
        print(f"WARNING: {n_bad} explanations had no parseable concepts.")

    probes = {"confounded": dir_confound, "clean": dir_clean}
    results = {}
    for name, d in probes.items():
        proj = P.project(A, d)
        table, k = association(concepts, proj, args.top_frac, args.min_count)
        results[name] = table
        print(f"\n=== {name} probe — top associated concepts (high vs low third, k={k}) ===")
        for r in table[:8]:
            print(f"  {r['association']:+.2f}  {r['concept']}  "
                  f"(high {r['freq_high']:.2f} / low {r['freq_low']:.2f}, n={r['count']})")

    out = {
        "run_name": rn, "model": MODEL, "clean_probe_src": clean_src,
        "config": {"top_frac": args.top_frac, "min_count": args.min_count, "n": len(A)},
        "association": results,
    }
    out_path = os.path.join(RESULTS_DIR, f"part3_enrichment_{rn}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
