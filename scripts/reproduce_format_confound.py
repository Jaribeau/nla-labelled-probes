#!/usr/bin/env python3
"""Phase 1: faithfully replicate Devbunova's format-confound finding on Llama-3.3-70B.

Trains Devbunova's LogisticRegression probe on bench_eval vs casual_deploy (last-token
resid_post, raw prompts, StandardScaler) and applies it to the four held-out 2x2 test cells,
reporting per-cell "eval" classification rates. If the probe fires on MCQ *format* rather
than evaluation *context*, her confound replicates. Only the model and layers differ from her
setup.

The 70B model runs on Modal (see src/modal_app.py). Prerequisites:
  - pip install -r requirements.txt
  - modal token new             (one-time Modal auth)
  - a local .env with HF_TOKEN  (gated Llama-3.3-70B + viliana-dev/eval-awareness-2x2 access)

Usage:
  python scripts/reproduce_format_confound.py                  # default layer sweep
  python scripts/reproduce_format_confound.py --layers 23,53   # custom layer sweep
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal
from dotenv import load_dotenv

load_dotenv()

from src.data import DEVBUNOVA_CELLS
from src.modal_app import DEFAULT_LAYERS, Llama, app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--layers", default=",".join(map(str, DEFAULT_LAYERS)),
        help="comma-separated resid_post layer indices (0-based) to probe",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--run-name", default=None,
        help="optional label appended to output filenames after the timestamp, "
             "e.g. --run-name layers-sweep",
    )
    parser.add_argument(
        "--scores-out", default="data/results/phase1_2x2_scores.json",
        help="where to write per-cell scores + config",
    )
    parser.add_argument(
        "--probe-out", default="data/probes/phase1_devbunova_logreg.json",
        help="where to write the trained probe (LR weights + scaler per layer)",
    )
    return parser.parse_args()


def run_on_modal(layers, batch_size, run_id):
    """Run the full Phase 1 pipeline on a Modal GPU instance; return its output dict."""
    with modal.enable_output(), app.run():
        return Llama().run_phase1.remote(layers=layers, batch_size=batch_size, run_id=run_id)


def sanitize(name):
    """Make a run-name filesystem-safe: keep alnum/dash/underscore, collapse the rest to '-'."""
    return "".join(c if (c.isalnum() or c in "-_") else "-" for c in name).strip("-_")


def stamped(path, tag):
    """Insert a run tag before the extension: foo.json -> foo_<tag>.json."""
    path = Path(path)
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def update_latest_symlink(base_path, target):
    """Point <base_stem>_latest<suffix> at the given run-scoped file."""
    base_path, target = Path(base_path), Path(target)
    link = base_path.with_name(f"{base_path.stem}_latest{base_path.suffix}")
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.name)  # relative link within the same directory
    return link


def save_scores(phase1_output, path):
    """Persist the full Phase 1 output: per-cell scores, probe, and config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(phase1_output, f, indent=2)


def save_probe(phase1_output, path):
    """Persist just the trained probe (LR weights + scaler) as its own artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probe_artifact = {
        "model": phase1_output["model"],
        "method": phase1_output["method"],
        "generated_at": phase1_output["generated_at"],
        "layers": phase1_output["probe"],  # {layer: {coef, intercept, scaler_*, train_acc}}
    }
    with open(path, "w") as f:
        json.dump(probe_artifact, f, indent=2)


def print_results_table(phase1_output):
    """Print per-cell EVAL-classification rates, ready to eyeball against Devbunova's pattern."""
    layers = phase1_output["layers"]
    cell_results = phase1_output["cells"]
    probe = phase1_output["probe"]

    print()
    print("=" * 78)
    print(f"PHASE 1 RESULTS  --  {phase1_output['model']}")
    print(f"method={phase1_output['method']}")
    print(f"layers={layers}")
    print("=" * 78)
    print("% of each held-out test cell classified as EVAL (class 0) by Devbunova's probe,")
    print("trained on bench_eval vs casual_deploy. Train accuracy shown per layer.")
    print()

    header = f"{'cell':<22}{'format':<7}{'style':<8}{'context':<9}"
    for layer in layers:
        header += f"{'L' + str(layer):>10}"
    print(header)
    print("-" * len(header))

    for cell_name, (_, _, meta) in DEVBUNOVA_CELLS.items():
        if cell_name not in cell_results:
            continue
        row = f"{cell_name:<22}{meta['format']:<7}{meta['style']:<8}{meta['context']:<9}"
        for layer in layers:
            row += f"{cell_results[cell_name][str(layer)]['pct_eval'] * 100:9.1f}%"
        print(row)

    train_row = f"{'(train acc)':<46}"
    for layer in layers:
        train_row += f"{probe[str(layer)]['train_acc'] * 100:9.1f}%"
    print("-" * len(header))
    print(train_row)

    print()
    print("Devbunova's finding to reproduce: MCQ-format cells classify ~90%+ EVAL while")
    print("free-form cells classify ~0%, regardless of casual/formal style.")


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    # Tag = timestamp, with the optional run-name appended after it.
    run_tag = f"{timestamp}_{sanitize(args.run_name)}" if args.run_name else timestamp
    phase1_output = run_on_modal(layers=layers, batch_size=args.batch_size, run_id=run_tag)
    phase1_output["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    phase1_output["run_id"] = run_tag
    phase1_output["run_name"] = args.run_name

    # Run-scoped paths so previous runs (even "bad" ones) are never overwritten;
    # a _latest symlink always points at the most recent run for convenience.
    scores_path = stamped(args.scores_out, run_tag)
    probe_path = stamped(args.probe_out, run_tag)
    save_scores(phase1_output, scores_path)
    save_probe(phase1_output, probe_path)
    scores_latest = update_latest_symlink(args.scores_out, scores_path)
    probe_latest = update_latest_symlink(args.probe_out, probe_path)
    print_results_table(phase1_output)

    print(f"\nRun tag                  : {run_tag}")
    print(f"Scores + per-cell detail : {scores_path}  (latest -> {scores_latest.name})")
    print(f"Trained probe            : {probe_path}  (latest -> {probe_latest.name})")
    print("Last-token activations   : Modal volume 'phase1-format-confound-out' "
          f"(phase1_pooled_activations_{run_tag}.npz)")


if __name__ == "__main__":
    main()
