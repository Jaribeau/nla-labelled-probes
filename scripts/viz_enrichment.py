"""Render a concept-association JSON as a self-contained HTML report (no plotting deps).

Ranked diverging bar chart per probe: concepts sorted by association = freq(concept | high
projection) − freq(concept | low projection). Bars right (red) = concepts that make the probe
fire; left (grey) = concepts associated with low projection. The confounded probe should show
its confound concept ("multiple-choice / format") near the top; the clean probe should not.

Run:  python scripts/viz_enrichment.py [part3_enrichment_*.json]   # default: latest
"""
import glob
import html
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
POS, NEG = "#d1495b", "#9aa0a6"


def latest():
    files = glob.glob(os.path.join(RESULTS_DIR, "part3_enrichment_*.json"))
    if not files:
        sys.exit("No part3_enrichment_*.json in data/results/")
    return max(files, key=os.path.getmtime)


def _bar_row(r, mx):
    """One diverging bar row from an association record."""
    e = r["association"]
    w = abs(e) / mx * 46  # % of the half-width
    color = POS if e >= 0 else NEG
    left = 50 if e >= 0 else 50 - w
    return (f'<div class="row" title="high {r["freq_high"]:.2f} / low {r["freq_low"]:.2f}, n={r["count"]}">'
            f'<div class="lbl">{html.escape(r["concept"])}</div>'
            f'<div class="track"><div class="mid"></div>'
            f'<div class="bar" style="left:{left:.1f}%;width:{w:.1f}%;background:{color}"></div></div>'
            f'<div class="val">{e:+.2f}</div></div>')


def panel_ordered(name, idx, order, mx):
    """Diverging bars for a fixed concept `order` (shared across panels); missing concept = 0."""
    zero = {"freq_high": 0.0, "freq_low": 0.0, "count": 0, "association": 0.0}
    bars = "".join(_bar_row({**(idx.get(c, zero)), "concept": c}, mx) for c in order)
    return f'<div class="panel"><h3>{html.escape(name)} probe</h3>{bars}</div>'


def render(path):
    d = json.load(open(path))
    enr = d["association"]
    cfg = d["config"]
    names = [n for n in ("confounded", "clean") if n in enr]

    # Same concepts in the same rows across panels, for direct comparison.
    top_n = 14
    idx = {n: {r["concept"]: r for r in enr[n]} for n in names}
    shared = []
    for n in names:
        shared += [r["concept"] for r in sorted(enr[n], key=lambda r: -abs(r["association"]))[:top_n]]
    shared = list(dict.fromkeys(shared))  # dedupe, preserve first-seen
    primary = names[0]  # order rows by the first probe's association (desc)
    shared.sort(key=lambda c: -(idx[primary].get(c, {}).get("association", 0.0)))
    shared_mx = max((abs(idx[n].get(c, {}).get("association", 0.0)) for n in names for c in shared), default=1.0) or 1.0
    shared_panels = "".join(panel_ordered(name, idx[name], shared, shared_mx) for name in names)
    return f"""<!doctype html><meta charset="utf-8">
<title>Concept association — {html.escape(d['run_name'])}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px;margin:30px auto;padding:0 16px;color:#222}}
 h1{{font-size:20px}} h3{{font-size:15px;margin:0 0 10px}} .sub{{color:#777;font-size:13px}}
 .panels{{display:flex;gap:34px;flex-wrap:wrap;margin-top:18px}} .panel{{flex:1;min-width:380px}}
 .row{{display:grid;grid-template-columns:170px 1fr 46px;align-items:center;gap:8px;margin:3px 0}}
 .lbl{{font-size:12px;text-align:right;color:#333;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
 .track{{position:relative;height:18px;background:#f4f4f6;border-radius:3px}}
 .mid{{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#ccc}}
 .bar{{position:absolute;top:2px;bottom:2px;border-radius:2px}}
 .val{{font-size:12px;color:#555;text-align:left}}
</style>
<h1>Concept association — what does each probe actually fire on?</h1>
<div class="sub">run <code>{html.escape(d['run_name'])}</code> · {html.escape(d['model'])} ·
 clean probe {html.escape(d['clean_probe_src'])} ·
 association = freq(concept | top {cfg['top_frac']:.0%} projection) − freq(concept | bottom {cfg['top_frac']:.0%}) ·
 n={cfg['n']}, min-count {cfg['min_count']}</div>
<p class="sub">Bars right = the probe fires more when this concept is present. Identical concept rows
in both panels (ordered by the {html.escape(names[0])} probe, shared scale), so you can read across
to compare. Baseline NLA tics (e.g. generic "Q&A structure") appear in both high and low projection,
cancel to ~0, and drop out.</p>
<div class="panels">{shared_panels}</div>
"""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest()
    out = path.rsplit(".", 1)[0] + ".html"
    with open(out, "w") as f:
        f.write(render(path))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
