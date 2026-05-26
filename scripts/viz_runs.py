"""Browse all runs and their files in one HTML page.

Scans data/results/ and data/probes/, groups files by part + run id, and renders a
table per part (one row per run, one column per artifact type). Writes runs.html at the
repo root and opens it. No args.

Run:  python scripts/viz_runs.py
"""
import datetime
import glob
import html
import os
import re
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "data", "results")
PROBES_DIR = os.path.join(ROOT, "data", "probes")

# (section title, [(column label, dir, regex capturing run id in group 1)])
PARTS = [
    ("Part 1 — refusal direction + NLA", [
        ("nla json", RESULTS_DIR, r"part1_refusal_nla_(.+)\.json$"),
        ("report", RESULTS_DIR, r"part1_refusal_nla_(.+)\.html$"),
        ("activations", RESULTS_DIR, r"part1_refusal_acts_(.+)\.npz$"),
        ("direction", PROBES_DIR, r"part1_refusal_dir_(.+)\.json$"),
    ]),
    ("Part 2 — recover direction from NLA labels", [
        ("judge", RESULTS_DIR, r"part2_judge_(.+)\.json$"),
        ("recover", RESULTS_DIR, r"part2_recover_(.+)\.json$"),
        ("report", RESULTS_DIR, r"part2_recover_(.+)\.html$"),
    ]),
    ("Part 3 — audit pipeline", [
        ("activations", RESULTS_DIR, r"part3_acts_(.+)\.npz$"),
        ("prompts", RESULTS_DIR, r"part3_prompts_(.+)\.json$"),
        ("verbalize", RESULTS_DIR, r"part3_verbalize_(.+)\.json$"),
        ("confound dir", PROBES_DIR, r"part3_confound_dir_(.+)\.json$"),
        ("probe", RESULTS_DIR, r"part3_probe_(.+)\.json$"),
        ("concepts", RESULTS_DIR, r"part3_concepts_(.+)\.json$"),
        ("concepts report", RESULTS_DIR, r"part3_concepts_(.+)\.html$"),
        ("association", RESULTS_DIR, r"part3_association_(.+)\.json$"),
        ("assoc report", RESULTS_DIR, r"part3_association_(.+)\.html$"),
    ]),
]


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def cell(path):
    if not path:
        return '<td class="empty">·</td>'
    st = os.stat(path)
    href = html.escape(os.path.relpath(path, ROOT))
    name = html.escape(os.path.basename(path))
    when = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
    meta = f"{human_size(st.st_size)} · {when}"
    return (f'<td><a href="{href}" title="{name}">{name}</a>'
            f'<div class="meta">{meta}</div></td>')


def section(title, cols):
    # run_id -> {col_label: path}; also track newest mtime per run for ordering
    runs, mtimes, matched = {}, {}, set()
    for label, d, rx in cols:
        for f in glob.glob(os.path.join(d, "*")):
            m = re.match(rx, os.path.basename(f))
            if m:
                rid = m.group(1)
                runs.setdefault(rid, {})[label] = f
                mtimes[rid] = max(mtimes.get(rid, 0), os.path.getmtime(f))
                matched.add(os.path.realpath(f))
    if not runs:
        return "", matched
    order = sorted(runs, key=lambda r: -mtimes[r])
    head = "".join(f"<th>{html.escape(lbl)}</th>" for lbl, _, _ in cols)
    body = ""
    for rid in order:
        cells = "".join(cell(runs[rid].get(lbl)) for lbl, _, _ in cols)
        body += f'<tr><th class="run">{html.escape(rid)}</th>{cells}</tr>'
    table = (f'<table><thead><tr><th class="run">run</th>{head}</tr></thead>'
             f'<tbody>{body}</tbody></table>')
    return f"<h2>{html.escape(title)}</h2>{table}", matched


def render():
    sections, matched = [], set()
    for title, cols in PARTS:
        s, m = section(title, cols)
        sections.append(s)
        matched |= m
    # anything in the data dirs we didn't slot into a part
    others = []
    for d in (RESULTS_DIR, PROBES_DIR):
        for f in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isfile(f) and os.path.realpath(f) not in matched:
                others.append(f)
    if others:
        rows = "".join(f"<tr>{cell(f)}</tr>" for f in others)
        sections.append(f'<h2>Other files</h2><table><tbody>{rows}</tbody></table>')
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><meta charset="utf-8">
<title>Runs — nla-labelled-probes</title>
<style>
 body{{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:28px auto;max-width:1200px;padding:0 16px;color:#222}}
 h1{{font-size:20px;margin-bottom:2px}} .sub{{color:#777;font-size:12px;margin-bottom:20px}}
 h2{{font-size:15px;margin:28px 0 8px}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:8px}}
 th,td{{border:1px solid #eee;padding:5px 9px;text-align:left;vertical-align:top}}
 thead th{{background:#fafafa;font-weight:600;white-space:nowrap}}
 th.run{{background:#f4f6f8;white-space:nowrap;font-family:ui-monospace,monospace;font-weight:600}}
 td.empty{{color:#ccc;text-align:center}}
 a{{color:#2563eb;text-decoration:none}} a:hover{{text-decoration:underline}}
 .meta{{color:#999;font-size:10.5px;margin-top:2px}}
</style>
<h1>Runs & artifacts</h1>
<div class="sub">generated {when} · scanned <code>data/results/</code> + <code>data/probes/</code></div>
{''.join(sections)}
"""


def main():
    out = os.path.join(ROOT, "runs.html")
    with open(out, "w") as f:
        f.write(render())
    print(f"Wrote {out}")
    webbrowser.open(f"file://{out}")


if __name__ == "__main__":
    main()
