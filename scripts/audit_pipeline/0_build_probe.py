"""Audit pipeline — build (or select) the probe to audit.

Builds the confounded "refusal" probe used in the confound demo: a diff-of-means direction
from a format-confounded training set (harmful-MCQ = positive, harmless-free = negative), so
format is perfectly correlated with the label. Extracts last-token L53 train activations on
Modal (GPU), builds the direction (src/probe.py), and saves it to data/probes/.

To audit an EXISTING probe instead (e.g. the clean Part 1 refusal direction), skip this script
— stage 3 lets you pick any probe file already in data/probes/.

Run:    python scripts/audit_pipeline/0_build_probe.py [--run-name v1] [--n-train 128]
Input:  built-in confound training set (src/data_confound.build_confound_train)
Output: data/probes/part3_confound_dir_<run>.json  (unit direction, threshold, layer)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/   -> _cli
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root  -> src

from src import probe as P
from src.data_confound import build_confound_train

PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "probes")
NLA_LAYER = 53


def extract_train(train, layer):
    """One Modal session: extract L{layer} acts for the confound train pos/neg sets."""
    from src.target_model import TargetModel, app

    with app.run():
        tm = TargetModel()
        pos = tm.extract.remote(train["pos"], [layer])["acts"][str(layer)]
        neg = tm.extract.remote(train["neg"], [layer])["acts"][str(layer)]
    return np.asarray(pos, np.float64), np.asarray(neg, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--n-train", type=int, default=128, help="confound train examples per class")
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        args.run_name = _cli.ask_str("Run name for the confounded probe", args.run_name)
        args.n_train = _cli.ask_int("Confound train examples per class", args.n_train)

    rn = args.run_name
    train = build_confound_train(args.n_train)
    print(f"Extracting L{NLA_LAYER} train acts: {len(train['pos'])} pos + {len(train['neg'])} neg...")
    pos, neg = extract_train(train, NLA_LAYER)

    direction = P.build_probe(pos, neg)
    threshold = P.midpoint_threshold(pos, neg, direction)
    out = os.path.join(PROBES_DIR, f"part3_confound_dir_{rn}.json")
    json.dump({"run_name": rn, "layer": NLA_LAYER,
               "dir_confound_unit": P.unit(direction).tolist(), "threshold": float(threshold)},
              open(out, "w"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
