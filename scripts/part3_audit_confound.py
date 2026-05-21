"""Part 3 — Audit demo: does the NLA audit catch a planted MCQ-format confound?

Pipeline (GPU steps cached, so judge/analysis iteration is free):
  1. Build a confounded "refusal" probe: harmful-MCQ (1) vs harmless-free (0) — format ≡ label.
  2. Score a de-confounded 2×2 ({harmful,harmless}×{MCQ,free}) with the confounded probe AND
     the clean Part 1 refusal direction; report fire-rates + confound strength.
  3. NLA-verbalize the 2×2 activations, theme-judge each explanation (format / refusal /
     harmful_topic / other).
  4. Compare theme distributions over each probe's firing set: a confounded probe → "format".

Run:  python scripts/part3_audit_confound.py [--run-name v1] [--reextract] [--reverbalize] [--regrade]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src import probe as P
from src.data_confound import build_2x2, build_confound_train
from src.judge import MODEL, judge_theme

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "probes")
NLA_LAYER = 53
CELLS = ["harmful_mcq", "harmful_free", "harmless_mcq", "harmless_free"]


def resolve_clean_dir(clean_run_id):
    files = glob.glob(os.path.join(PROBES_DIR, "part1_refusal_dir_*.json"))
    if clean_run_id:
        files = [f for f in files if clean_run_id in f]
    if not files:
        sys.exit("No part1_refusal_dir_*.json (clean probe) found")
    d = json.load(open(max(files, key=os.path.getmtime)))
    return np.array(d["probes"][str(NLA_LAYER)]["direction_unit"]), os.path.basename(max(files, key=os.path.getmtime))


def extract(train, cells, layer):
    """One Modal session: extract L{layer} acts for train pos/neg + each 2×2 cell."""
    from src.target_model import TargetModel, app

    with app.run():
        tm = TargetModel()
        pos = tm.extract.remote(train["pos"], [layer])["acts"][str(layer)]
        neg = tm.extract.remote(train["neg"], [layer])["acts"][str(layer)]
        cell_acts = {c: tm.extract.remote(cells[c]["texts"], [layer])["acts"][str(layer)] for c in CELLS}
    return pos, neg, cell_acts


def verbalize(vectors, max_new_tokens=256):
    from src.nla_modal import NLAVerbalizer, app

    with app.run():
        return NLAVerbalizer().verbalize.remote(vectors, max_new_tokens=max_new_tokens)


def theme_dist(themes, mask):
    """Proportion of each theme among masked prompts (None = unparseable)."""
    sel = [t for t, m in zip(themes, mask) if m]
    n = len(sel) or 1
    out = {th: sum(t == th for t in sel) / n for th in ("format", "refusal", "harmful_topic", "other")}
    out["unparseable"] = sum(t is None for t in sel) / n
    out["n"] = len(sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-cell", type=int, default=50)
    ap.add_argument("--clean-run-id", default=None, help="Part 1 dir run (default: latest)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--reverbalize", action="store_true")
    ap.add_argument("--regrade", action="store_true")
    args = ap.parse_args()
    rn = args.run_name

    acts_path = os.path.join(RESULTS_DIR, f"part3_acts_{rn}.npz")
    verb_path = os.path.join(RESULTS_DIR, f"part3_verbalize_{rn}.json")
    theme_path = os.path.join(RESULTS_DIR, f"part3_theme_{rn}.json")

    train = build_confound_train(args.n_train)
    cells = build_2x2(args.n_cell)
    # flat 2×2 arrays in CELLS order
    flat_text, flat_harm, flat_mcq, flat_cell, flat_instr = [], [], [], [], []
    for c in CELLS:
        flat_text += cells[c]["texts"]
        flat_instr += cells[c]["instructions"]
        flat_harm += [cells[c]["harm"]] * len(cells[c]["texts"])
        flat_mcq += [cells[c]["mcq"]] * len(cells[c]["texts"])
        flat_cell += [c] * len(cells[c]["texts"])
    flat_harm, flat_mcq = np.array(flat_harm), np.array(flat_mcq)

    # --- GPU step 1: extraction (cached) ---
    if os.path.exists(acts_path) and not args.reextract:
        print(f"Loading cached activations {acts_path}")
        z = np.load(acts_path)
        pos, neg, A = z["pos"], z["neg"], z["A2x2"]
    else:
        print(f"Extracting L{NLA_LAYER}: train {len(train['pos'])}+{len(train['neg'])}, 2×2 {len(flat_text)}...")
        pos, neg, cell_acts = extract(train, cells, NLA_LAYER)
        A = np.concatenate([cell_acts[c] for c in CELLS], axis=0)
        np.savez_compressed(acts_path, pos=pos, neg=neg, A2x2=A,
                            harm=flat_harm, mcq=flat_mcq)
        print(f"Saved {acts_path}")
    pos, neg, A = pos.astype(np.float64), neg.astype(np.float64), A.astype(np.float64)

    # --- probes ---
    dir_confound = P.build_probe(pos, neg)
    thr_confound = P.midpoint_threshold(pos, neg, dir_confound)
    dir_clean, clean_src = resolve_clean_dir(args.clean_run_id)
    print(f"Clean probe from {clean_src}")

    proj_c = P.project(A, dir_confound)
    proj_k = P.project(A, dir_clean)
    # clean-probe threshold from the 2×2 content axis (harmful_free vs harmless_free), in-scale
    hf = (flat_harm == 1) & (flat_mcq == 0)
    sf = (flat_harm == 0) & (flat_mcq == 0)
    thr_clean = 0.5 * (proj_k[hf].mean() + proj_k[sf].mean())
    fires_c = proj_c > thr_confound
    fires_k = proj_k > thr_clean

    def fire_rates(fires):
        return {c: float(fires[np.array(flat_cell) == c].mean()) for c in CELLS}

    fr_c, fr_k = fire_rates(fires_c), fire_rates(fires_k)
    strength = {
        "confound_format_effect_harmless": fr_c["harmless_mcq"] - fr_c["harmless_free"],
        "confound_concept_effect": fr_c["harmful_free"] - fr_c["harmless_free"],
        "clean_format_effect_harmless": fr_k["harmless_mcq"] - fr_k["harmless_free"],
        "clean_concept_effect": fr_k["harmful_free"] - fr_k["harmless_free"],
    }

    # --- GPU step 2: verbalize 2×2 activations (cached) ---
    if os.path.exists(verb_path) and not args.reverbalize:
        print(f"Loading cached explanations {verb_path}")
        explanations = json.load(open(verb_path))["explanations"]
    else:
        print(f"Verbalizing {len(A)} 2×2 activations via NLA AV...")
        explanations = verbalize([v for v in A.astype(np.float32)], args.max_new_tokens)
        json.dump({"run_name": rn, "cells": flat_cell, "explanations": explanations}, open(verb_path, "w"), indent=2)
        print(f"Saved {verb_path}")

    # --- judge themes (cached) ---
    if os.path.exists(theme_path) and not args.regrade:
        print(f"Loading cached themes {theme_path}")
        themes = json.load(open(theme_path))["themes"]
    else:
        print(f"Theme-judging {len(explanations)} explanations with {MODEL}...")
        judg = judge_theme(explanations)
        themes = [j["theme"] for j in judg]
        json.dump({"run_name": rn, "model": MODEL, "judgments": judg, "themes": themes}, open(theme_path, "w"), indent=2)
        print(f"Saved {theme_path}")

    # --- audit: theme distribution over each probe's firing set ---
    audit = {
        "confounded_probe_fires": theme_dist(themes, fires_c),
        "clean_probe_fires": theme_dist(themes, fires_k),
        "by_cell": {c: theme_dist(themes, np.array(flat_cell) == c) for c in CELLS},
    }

    summary = {
        "run_name": rn,
        "n_2x2": len(flat_text),
        "fire_rates_confounded": fr_c,
        "fire_rates_clean": fr_k,
        "confound_strength": strength,
        "theme_confounded_fires": audit["confounded_probe_fires"],
        "theme_clean_fires": audit["clean_probe_fires"],
    }
    print(json.dumps(summary, indent=2))

    examples = [
        {"cell": flat_cell[i], "harm": int(flat_harm[i]), "mcq": int(flat_mcq[i]),
         "instruction": flat_instr[i], "proj_confound": float(proj_c[i]), "proj_clean": float(proj_k[i]),
         "fires_confound": bool(fires_c[i]), "fires_clean": bool(fires_k[i]),
         "theme": themes[i], "explanation": explanations[i]}
        for i in range(len(flat_text))
    ]
    out = {
        "run_name": rn, "model": "meta-llama/Llama-3.3-70B-Instruct",
        "nla_checkpoint": "kitft/Llama-3.3-70B-NLA-L53-av", "nla_layer": NLA_LAYER,
        "clean_probe_src": clean_src,
        "config": {"n_train": args.n_train, "n_cell": args.n_cell},
        "thresholds": {"confound": float(thr_confound), "clean": float(thr_clean)},
        "fire_rates": {"confounded": fr_c, "clean": fr_k},
        "confound_strength": strength,
        "audit": audit,
        "examples": examples,
    }
    out_path = os.path.join(RESULTS_DIR, f"part3_audit_{rn}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    # persist the confounded direction for reuse
    json.dump({"run_name": rn, "layer": NLA_LAYER,
               "dir_confound_unit": P.unit(dir_confound).tolist(), "threshold": float(thr_confound)},
              open(os.path.join(PROBES_DIR, f"part3_confound_dir_{rn}.json"), "w"))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
