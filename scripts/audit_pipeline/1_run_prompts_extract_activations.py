"""Audit pipeline — stage 1: run prompts through the target model and extract activations.

Builds the audit prompt set and extracts last-token L53 residual activations on Modal (GPU).
Built-in prompt set: the de-confounded 2×2 ({harmful,harmless} × {MCQ,free}) from
src/data_confound.build_2x2 — harmful and harmless instructions are each rendered in both
formats, so the MCQ-vs-free contrast isolates format with content held fixed.

Run:    python scripts/audit_pipeline/1_run_prompts_extract_activations.py [--run-name v1] [--n-cell 50]
Input:  built-in 2×2 prompt set (src/data_confound.build_2x2)
Output: data/results/part3_acts_<run>.npz     (A2x2 activations + harm/mcq label arrays)
        data/results/part3_prompts_<run>.json  (cell/instruction/text per prompt, for reference)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/   -> _cli
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root  -> src

from src.data_confound import build_2x2

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "results")
NLA_LAYER = 53
CELLS = ["harmful_mcq", "harmful_free", "harmless_mcq", "harmless_free"]


def extract(cells, layer):
    """One Modal session: extract L{layer} acts for each 2×2 cell, concatenated in CELLS order."""
    from src.target_model import TargetModel, app

    with app.run():
        tm = TargetModel()
        per_cell = {c: tm.extract.remote(cells[c]["texts"], [layer])["acts"][str(layer)] for c in CELLS}
    return np.concatenate([per_cell[c] for c in CELLS], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--n-cell", type=int, default=50, help="examples per 2×2 cell")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        args.run_name = _cli.choose_run("part3_acts_*.npz", r"part3_acts_(.+)\.npz",
                                        "Part 3 run", allow_new=True)
        args.n_cell = _cli.ask_int("Examples per 2×2 cell", args.n_cell)

    rn = args.run_name
    acts_path = os.path.join(RESULTS_DIR, f"part3_acts_{rn}.npz")
    if os.path.exists(acts_path) and not args.force:
        if not (_cli.interactive() and _cli.ask_bool(f"{os.path.basename(acts_path)} exists — re-extract?", False)):
            print(f"Cached {acts_path} exists; nothing to do (use --force to re-extract).")
            return

    cells = build_2x2(args.n_cell)
    flat_cell, flat_instr, flat_text, flat_harm, flat_mcq = [], [], [], [], []
    for c in CELLS:
        flat_text += cells[c]["texts"]
        flat_instr += cells[c]["instructions"]
        flat_cell += [c] * len(cells[c]["texts"])
        flat_harm += [cells[c]["harm"]] * len(cells[c]["texts"])
        flat_mcq += [cells[c]["mcq"]] * len(cells[c]["texts"])

    print(f"Extracting L{NLA_LAYER} for {len(flat_text)} prompts ({args.n_cell}/cell)...")
    A = extract(cells, NLA_LAYER)
    np.savez_compressed(acts_path, A2x2=A, harm=np.array(flat_harm), mcq=np.array(flat_mcq))
    print(f"Saved {acts_path}")

    prompts_path = os.path.join(RESULTS_DIR, f"part3_prompts_{rn}.json")
    json.dump({"run_name": rn,
               "prompts": [{"cell": flat_cell[i], "instruction": flat_instr[i], "text": flat_text[i]}
                           for i in range(len(flat_text))]},
              open(prompts_path, "w"), indent=2)
    print(f"Saved {prompts_path}")


if __name__ == "__main__":
    main()
