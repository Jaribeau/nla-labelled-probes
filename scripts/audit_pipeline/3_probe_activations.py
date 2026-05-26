"""Audit pipeline — stage 3: probe the activations.

Projects each activation onto the unit direction of every selected probe (src/probe.py). The
projection per prompt is what stage 5 ranks concepts against; no GPU or API.

When the prompt set carries the confound labels (harm/mcq, i.e. the built-in 2×2), an optional
`confound_validation` block is added: per-cell fire rates and confound-strength metrics that
show whether each probe tracks the concept (harm) or the format confound (MCQ). The clean
probe's fire threshold is calibrated on the 2×2 content axis (harmful-free vs harmless-free),
since a Part 1 threshold is in a different (standardized) scale.

Run:    python scripts/audit_pipeline/3_probe_activations.py [--run-name v1] [--confound-probe <rn>] [--clean-run-id <id>]
Input:  data/results/part3_acts_<run>.npz + selected probe files in data/probes/
Output: data/results/part3_probe_<run>.json  (per-probe projection; optional confound_validation)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/   -> _cli
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root  -> src

from src import probe as P

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "results")
PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "probes")
NLA_LAYER = 53
CELLS = ["harmful_mcq", "harmful_free", "harmless_mcq", "harmless_free"]


def _latest(pattern):
    files = glob.glob(os.path.join(PROBES_DIR, pattern))
    return max(files, key=os.path.getmtime) if files else None


def load_confound_dir(run_id):
    f = _latest(f"part3_confound_dir_*{run_id}*.json") if run_id else _latest("part3_confound_dir_*.json")
    if not f:
        sys.exit("No part3_confound_dir_*.json found (run 0_build_probe.py first)")
    return np.array(json.load(open(f))["dir_confound_unit"]), os.path.basename(f)


def load_clean_dir(run_id):
    f = _latest(f"part1_refusal_dir_*{run_id}*.json") if run_id else _latest("part1_refusal_dir_*.json")
    if not f:
        sys.exit("No part1_refusal_dir_*.json (clean probe) found")
    return np.array(json.load(open(f))["probes"][str(NLA_LAYER)]["direction_unit"]), os.path.basename(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--confound-probe", default=None, help="confounded probe run id (default: latest)")
    ap.add_argument("--clean-run-id", default=None, help="clean Part 1 probe run id (default: latest)")
    ap.add_argument("--top-frac", type=float, default=0.33, help="(only echoed; ranking happens in stage 5)")
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        args.run_name = _cli.choose_run("part3_acts_*.npz", r"part3_acts_(.+)\.npz", "Part 3 run")
        args.confound_probe = _cli.choose_run("part3_confound_dir_*.json",
                                              r"part3_confound_dir_(.+)\.json", "confounded probe",
                                              base=PROBES_DIR)
        args.clean_run_id = _cli.choose_run("part1_refusal_dir_*.json",
                                            r"part1_refusal_dir_(.+)\.json", "clean (Part 1) probe",
                                            base=PROBES_DIR)
    rn = args.run_name

    z = np.load(os.path.join(RESULTS_DIR, f"part3_acts_{rn}.npz"))
    A = z["A2x2"].astype(np.float64)
    harm, mcq = z.get("harm"), z.get("mcq")

    dir_confound, confound_src = load_confound_dir(args.confound_probe)
    dir_clean, clean_src = load_clean_dir(args.clean_run_id)
    print(f"Confounded probe: {confound_src}\nClean probe:      {clean_src}")

    proj_c = P.project(A, dir_confound)
    proj_k = P.project(A, dir_clean)
    out = {
        "run_name": rn, "layer": NLA_LAYER,
        "probes": {
            "confounded": {"src": confound_src, "projection": proj_c.tolist()},
            "clean": {"src": clean_src, "projection": proj_k.tolist()},
        },
    }

    # Optional confound validation: only when the 2×2 harm/mcq labels are present.
    if harm is not None and mcq is not None:
        harm, mcq = np.asarray(harm), np.asarray(mcq)
        cell = np.array([f"{'harmful' if h else 'harmless'}_{'mcq' if m else 'free'}"
                         for h, m in zip(harm, mcq)])
        thr_confound = float(json.load(open(os.path.join(PROBES_DIR, confound_src)))["threshold"])
        hf, sf = (harm == 1) & (mcq == 0), (harm == 0) & (mcq == 0)
        thr_clean = float(0.5 * (proj_k[hf].mean() + proj_k[sf].mean()))
        fires_c, fires_k = proj_c > thr_confound, proj_k > thr_clean

        def fire_rates(fires):
            return {c: float(fires[cell == c].mean()) for c in CELLS}

        fr_c, fr_k = fire_rates(fires_c), fire_rates(fires_k)
        out["confound_validation"] = {
            "thresholds": {"confounded": thr_confound, "clean": thr_clean},
            "fire_rates": {"confounded": fr_c, "clean": fr_k},
            "confound_strength": {
                "confound_format_effect_harmless": fr_c["harmless_mcq"] - fr_c["harmless_free"],
                "confound_concept_effect": fr_c["harmful_free"] - fr_c["harmless_free"],
                "clean_format_effect_harmless": fr_k["harmless_mcq"] - fr_k["harmless_free"],
                "clean_concept_effect": fr_k["harmful_free"] - fr_k["harmless_free"],
            },
        }
        out["probes"]["confounded"]["fires"] = fires_c.tolist()
        out["probes"]["clean"]["fires"] = fires_k.tolist()
        print(json.dumps(out["confound_validation"], indent=2))

    out_path = os.path.join(RESULTS_DIR, f"part3_probe_{rn}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
