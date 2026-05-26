"""Audit pipeline — stage 4: extract concepts from the NLA verbalizations.

Sends each verbalization to the open-vocabulary concept judge (Claude; prompt at
src/prompts/concept-extraction-judge-prompt.md) which returns ≤4 concise concept tags. No
fixed taxonomy — the tags emerge from the data. Cached per run; needs ANTHROPIC_API_KEY.

A grade tag suffixes the output so a re-grade (e.g. under a revised judge prompt) is saved
alongside prior gradings instead of overwriting them.

Run:    python scripts/audit_pipeline/4_extract_concepts_from_nla_verbalizations.py [--run-name v1] [--regrade] [--grade-tag <tag>]
Input:  data/results/part3_verbalize_<run>.json
Output: data/results/part3_concepts_<run>[_<grade-tag>].json  (open-vocab tags per activation)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/   -> _cli
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root  -> src

from src.judge import MODEL, judge_concepts

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--regrade", action="store_true", help="re-extract concepts (ignore cache)")
    ap.add_argument("--grade-tag", default=None,
                    help="suffix for the concepts file (e.g. a prompt version); keeps prior "
                         "gradings instead of overwriting. Input still keyed on --run-name.")
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        args.run_name = _cli.choose_run("part3_verbalize_*.json", r"part3_verbalize_(.+)\.json",
                                        "Part 3 run")
        args.regrade = _cli.ask_bool("Re-extract concepts (ignore cache)?", args.regrade)
        args.grade_tag = _cli.ask_str("Grade tag (blank = overwrite default grading)", "") or None

    rn = args.run_name
    suffix = f"{rn}_{args.grade_tag}" if args.grade_tag else rn
    explanations = json.load(open(os.path.join(RESULTS_DIR, f"part3_verbalize_{rn}.json")))["explanations"]
    out_path = os.path.join(RESULTS_DIR, f"part3_concepts_{suffix}.json")

    cached_ok = (not args.regrade and os.path.exists(out_path)
                 and len(json.load(open(out_path))["concepts"]) == len(explanations))
    if cached_ok:
        print(f"Using cached concepts {out_path} (use --regrade to re-extract).")
    else:
        print(f"Extracting concepts from {len(explanations)} explanations with {MODEL}...")
        concepts = [j["concepts"] for j in judge_concepts(explanations)]
        json.dump({"run_name": rn, "grade_tag": args.grade_tag, "model": MODEL, "concepts": concepts},
                  open(out_path, "w"), indent=2)
        print(f"Saved {out_path}")

    # Render + open the tag-count view of the result.
    import inspect_concepts
    inspect_concepts.report(suffix)


if __name__ == "__main__":
    main()
