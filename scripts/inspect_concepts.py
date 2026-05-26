"""View ALL extracted concept tags before any min_count filtering.

Loads the cached per-explanation tags from data/results/part3_concepts_<run>.json
(written by audit pipeline stage 4) and renders an HTML report listing every distinct
tag with its total count, sorted by frequency. Surfaces the full vocabulary so synonym
fragmentation ("multiple choice" vs "multiple-choice format" vs "mcq") is visible.
Complements stage 4. No GPU, no activations, no re-grading.

Run:  python scripts/inspect_concepts.py [--run-name part3v1]
"""
import argparse
import html
import json
import os
import webbrowser
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")


def report(token, open_browser=True):
    """Render the tag-count HTML for part3_concepts_<token>.json; returns the HTML path.

    `token` is the concepts-file run token incl. any grade tag (e.g. "part3v1" or
    "part3v1_v2"). Used both by this script's CLI and by audit-pipeline stage 4.
    """
    src = os.path.join(RESULTS_DIR, f"part3_concepts_{token}.json")
    concepts = json.load(open(src))["concepts"]
    n_acts = len(concepts)

    counts = Counter()
    for tags in concepts:
        for t in (tags or []):
            counts[t] += 1

    rows = "\n".join(
        f"<tr><td class=c>{c}</td><td>{html.escape(t)}</td></tr>"
        for t, c in counts.most_common()
    )
    page = f"""<!doctype html><meta charset=utf-8>
<title>concepts — {html.escape(token)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:640px}}
 h1{{font-size:1.2rem}} .meta{{color:#666;margin-bottom:1rem}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{text-align:left;padding:.25rem .75rem;border-bottom:1px solid #eee}}
 td.c{{text-align:right;font-variant-numeric:tabular-nums;color:#333;width:4rem}}
 th{{cursor:pointer;user-select:none}}
</style>
<h1>Concept tags — {html.escape(token)}</h1>
<div class=meta>{len(counts)} distinct tags · {n_acts} activations · pre-filter (no min_count)</div>
<table id=t><thead><tr><th onclick=sort(0)>count</th><th onclick=sort(1)>concept</th></tr></thead>
<tbody>{rows}</tbody></table>
<script>
let dir=1;
function sort(col){{
  const tb=document.querySelector('#t tbody');
  const rows=[...tb.rows];
  rows.sort((a,b)=>{{
    let x=a.cells[col].textContent,y=b.cells[col].textContent;
    if(col===0){{x=+x;y=+y;return (x-y)*dir;}}
    return x.localeCompare(y)*dir;
  }});
  dir=-dir; rows.forEach(r=>tb.appendChild(r));
}}
</script>"""

    out = os.path.join(RESULTS_DIR, f"part3_concepts_{token}.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"Wrote {out} ({len(counts)} distinct tags, {n_acts} activations)")
    if open_browser:
        webbrowser.open(f"file://{os.path.realpath(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="part3v1")
    ap.add_argument("--grade-tag", default=None,
                    help="grading suffix matching stage 4 --grade-tag")
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        token = _cli.choose_run("part3_concepts_*.json", r"part3_concepts_(.+)\.json", "concept set")
    else:
        token = f"{args.run_name}_{args.grade_tag}" if args.grade_tag else args.run_name

    report(token)


if __name__ == "__main__":
    main()
