"""Audit pipeline — stage 2: verbalize the activations with the NLA.

Feeds each extracted activation to the natural-language autoencoder
(kitft/Llama-3.3-70B-NLA-L53-av) on Modal (GPU) to produce a short text description of what
the model was representing at that token position.

Run:    python scripts/audit_pipeline/2_verbalize_activations.py [--run-name v1] [--max-new-tokens 256]
Input:  data/results/part3_acts_<run>.npz
Output: data/results/part3_verbalize_<run>.json  (cells + one NLA explanation per activation)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/   -> _cli
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root  -> src

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "results")


def cell_name(harm, mcq):
    return f"{'harmful' if harm else 'harmless'}_{'mcq' if mcq else 'free'}"


def verbalize(vectors, max_new_tokens):
    from src.nla_modal import NLAVerbalizer, app

    with app.run():
        return NLAVerbalizer().verbalize.remote(vectors, max_new_tokens=max_new_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="NLA explanation length cap")
    ap.add_argument("--force", action="store_true", help="re-verbalize even if cached")
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        args.run_name = _cli.choose_run("part3_acts_*.npz", r"part3_acts_(.+)\.npz", "Part 3 run")
        args.max_new_tokens = _cli.ask_int("NLA explanation length cap", args.max_new_tokens)

    rn = args.run_name
    verb_path = os.path.join(RESULTS_DIR, f"part3_verbalize_{rn}.json")
    if os.path.exists(verb_path) and not args.force:
        if not (_cli.interactive() and _cli.ask_bool(f"{os.path.basename(verb_path)} exists — re-verbalize?", False)):
            print(f"Cached {verb_path} exists; nothing to do (use --force to re-verbalize).")
            return

    z = np.load(os.path.join(RESULTS_DIR, f"part3_acts_{rn}.npz"))
    A = z["A2x2"].astype(np.float32)
    cells = [cell_name(h, m) for h, m in zip(z["harm"], z["mcq"])]

    print(f"Verbalizing {len(A)} activations via NLA AV...")
    explanations = verbalize([v for v in A], args.max_new_tokens)
    json.dump({"run_name": rn, "cells": cells, "explanations": explanations},
              open(verb_path, "w"), indent=2)
    print(f"Saved {verb_path}")


if __name__ == "__main__":
    main()
