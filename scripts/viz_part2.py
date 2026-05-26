"""Render a Part 2 recovery JSON as a self-contained HTML report (no plotting deps).

Three complementary views of the held-out test set (L53):
  1. Strip: each prompt placed by its projection onto Arditi's direction, shown twice — once
     colored by source (harmful/harmless), once by the NLA judge (refusal/comply). Matching
     color patterns = the NLA labels reproduce the probe; mismatches are the divergence cases.
  2. Scatter: projection onto Arditi's direction vs onto the NLA-label direction (both
     z-scored). Points on the diagonal = the two directions agree.
  3. Confusion: NLA judge label vs source label.
Plus per-layer cosine/AUROC and a full divergence-case table.

Run:  python scripts/viz_part2.py [part2_recover_*.json]   # default: latest
"""
import glob
import html
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
HARMFUL, HARMLESS = "#d1495b", "#3a6ea5"
DIVERGE = "#e09f3e"


def latest():
    files = glob.glob(os.path.join(RESULTS_DIR, "part2_recover_*.json"))
    if not files:
        sys.exit("No part2_recover_*.json in data/results/")
    return max(files, key=os.path.getmtime)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def strip_svg(pts, w=860, h=210, pad=52):
    """Two lanes over the Arditi-projection axis: colored by source, then by NLA judge."""
    if not pts:
        return "<p><i>no test points</i></p>"
    xs = [p["proj_arditi"] for p in pts]
    lo, hi = min(xs), max(xs)
    lo, hi = lo - 0.06 * (hi - lo or 1), hi + 0.06 * (hi - lo or 1)

    def sx(v):
        return pad + (v - lo) / (hi - lo) * (w - 2 * pad)

    lane_true, lane_nla = 70, 150
    boundary = 0.5 * (_mean([p["proj_arditi"] for p in pts if p["nla_label"] == 1])
                      + _mean([p["proj_arditi"] for p in pts if p["nla_label"] == 0]))
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    out.append(f'<line x1="{sx(boundary):.1f}" y1="40" x2="{sx(boundary):.1f}" y2="{h-30}" stroke="{DIVERGE}" stroke-dasharray="5,4"/>')
    out.append(f'<text x="{sx(boundary):.1f}" y="32" font-size="11" text-anchor="middle" fill="#b07d29">NLA decision boundary</text>')
    for lane, key, label in [(lane_true, "true_label", "colored by SOURCE (harmful/harmless)"),
                             (lane_nla, "nla_label", "colored by NLA JUDGE (refusal/comply)")]:
        out.append(f'<line x1="{pad}" y1="{lane}" x2="{w-pad}" y2="{lane}" stroke="#eee"/>')
        out.append(f'<text x="{pad}" y="{lane-12}" font-size="11" fill="#888">{label}</text>')
        jit = {}
        for p in pts:
            k = round(sx(p["proj_arditi"]))
            jit[k] = jit.get(k, 0) + 1
            dy = (jit[k] - 1) * 7 * (1 if jit[k] % 2 else -1) / 2
            diverged = p["true_label"] != p["nla_label"]
            color = HARMFUL if p[key] == 1 else HARMLESS
            ring = f'stroke="{DIVERGE}" stroke-width="2.5"' if diverged else 'stroke="white" stroke-width="0.5"'
            # In the NLA lane, fade dots by judge confidence: hedged (p~0.5) look faint.
            if key == "nla_label" and "p_refusal" in p:
                op = 0.3 + 0.7 * abs(2 * p["p_refusal"] - 1)
            else:
                op = 0.85
            out.append(f'<circle cx="{sx(p["proj_arditi"]):.1f}" cy="{lane+dy:.1f}" r="5" fill="{color}" fill-opacity="{op:.2f}" {ring}/>')
    out.append(f'<text x="{w/2}" y="{h-8}" font-size="12" text-anchor="middle" fill="#666">projection onto Arditi refusal direction →</text>')
    out.append("</svg>")
    return "".join(out)


def scatter_svg(pts, w=420, h=420, pad=52):
    if not pts:
        return ""
    def z(vals):
        m = _mean(vals)
        sd = (_mean([(v - m) ** 2 for v in vals]) ** 0.5) or 1.0
        return [(v - m) / sd for v in vals]
    zx = z([p["proj_arditi"] for p in pts])
    zy = z([p["proj_nla"] for p in pts])
    lim = max(3.0, max(abs(v) for v in zx + zy) * 1.05)

    def sx(v):
        return pad + (v + lim) / (2 * lim) * (w - 2 * pad)

    def sy(v):
        return h - pad - (v + lim) / (2 * lim) * (h - 2 * pad)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    out.append(f'<line x1="{sx(-lim):.1f}" y1="{sy(-lim):.1f}" x2="{sx(lim):.1f}" y2="{sy(lim):.1f}" stroke="#ddd" stroke-dasharray="4,4"/>')
    out.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#ccc"/>')
    out.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#ccc"/>')
    out.append(f'<text x="{w/2}" y="{h-8}" font-size="11" text-anchor="middle" fill="#666">Arditi projection (z)</text>')
    out.append(f'<text x="14" y="{h/2}" font-size="11" text-anchor="middle" fill="#666" transform="rotate(-90 14 {h/2})">NLA-label projection (z)</text>')
    for p, x, y in zip(pts, zx, zy):
        color = HARMFUL if p["true_label"] == 1 else HARMLESS
        diverged = p["true_label"] != p["nla_label"]
        ring = f'stroke="{DIVERGE}" stroke-width="2.5"' if diverged else 'stroke="white" stroke-width="0.5"'
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}" fill-opacity="0.8" {ring}/>')
    out.append("</svg>")
    return "".join(out)


def confusion_html(pts):
    c = {(t, n): 0 for t in (1, 0) for n in (1, 0)}
    for p in pts:
        c[(p["true_label"], p["nla_label"])] += 1
    return f"""<table class="conf">
 <tr><th></th><th colspan="2">NLA judge</th></tr>
 <tr><th></th><th>refusal</th><th>comply</th></tr>
 <tr><th>harmful</th><td class="hit">{c[(1,1)]}</td><td class="miss">{c[(1,0)]}</td></tr>
 <tr><th>harmless</th><td class="miss">{c[(0,1)]}</td><td class="hit">{c[(0,0)]}</td></tr>
</table>"""


def render(path):
    d = json.load(open(path))
    s = d["summary"]
    lr = d["layer_report"]
    layers = sorted(lr, key=int)
    pts = d.get("scatter_l53", [])

    labels = {
        "cosine_nla_arditi": "cosine(NLA soft-label dir, Arditi dir)",
        "cosine_nlahard_arditi": "cosine(NLA hard-label dir, Arditi dir)",
        "cosine_true_arditi": "cosine(true-label dir, Arditi dir)",
        "cosine_nla_true": "cosine(NLA-label dir, true-label dir)",
        "auroc_test_nla": "AUROC test — NLA-label probe",
        "auroc_test_true": "AUROC test — true-label probe",
        "auroc_test_arditi": "AUROC test — Arditi probe",
    }

    def row(metric):
        cells = "".join(f"<td>{lr[L][metric]:.3f}</td>" for L in layers)
        return f'<tr><th>{labels[metric]}</th>{cells}</tr>'

    table = (
        f'<table class="m"><tr><th>metric</th>{"".join(f"<th>L{L}</th>" for L in layers)}</tr>'
        + "".join(row(m) for m in labels) + "</table>"
    )

    def fires_cell(f):
        return ('<span style="color:#222">● fires</span>' if f
                else '<span style="color:#aaa">○ quiet</span>')

    def nla_cell(n):
        return (f'<b style="color:{HARMFUL}">refusal</b>' if n == 1
                else f'<b style="color:{HARMLESS}">comply</b>')

    def case_rows(cases, empty_msg):
        rows = "".join(
            f'<tr><td style="color:{HARMFUL if c["true_label"] else HARMLESS};font-weight:600">'
            f'{"harmful" if c["true_label"] else "harmless"}</td>'
            f'<td style="text-align:center;white-space:nowrap">{fires_cell(c["fires"])}</td>'
            f'<td style="text-align:center;white-space:nowrap">{nla_cell(c["nla_label"]) if "nla_label" in c else "—"}</td>'
            f'<td style="text-align:center">{c.get("p_refusal", float("nan")):.2f}</td>'
            f'<td>{html.escape(c["instruction"])}</td>'
            f'<td style="font-size:12px;color:#555">{html.escape(c.get("reason",""))}</td>'
            f'<td><details><summary>show</summary><div class="expl">{html.escape(c["explanation"])}</div></details></td></tr>'
            for c in cases
        )
        return rows or f'<tr><td colspan="7"><i>{empty_msg}</i></td></tr>'

    div_rows = case_rows(d["divergence_cases"], "none — NLA label agreed with source and probe on every test prompt")
    hedged_rows = case_rows(d.get("hedged_cases", []), "none — every explanation was judged confidently (p<0.2 or p>0.8)")
    case_header = ('<tr><th>source</th><th>probe</th><th>NLA judge</th><th>p(refusal)</th>'
                   '<th>instruction (full)</th><th>judge reason</th><th>NLA explanation</th></tr>')

    return f"""<!doctype html><meta charset="utf-8">
<title>Part 2 — {html.escape(d['run_id'])}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px;margin:30px auto;padding:0 16px;color:#222}}
 h1{{font-size:20px}} h3{{margin-top:30px}} .sub{{color:#777;font-size:13px;margin-bottom:20px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
 .card{{background:#f6f7f9;border-radius:8px;padding:12px 16px;flex:1;min-width:150px}}
 .card .big{{font-size:26px;font-weight:700}}
 table{{border-collapse:collapse;width:100%;margin-top:10px}}
 th,td{{border-bottom:1px solid #eee;padding:7px 9px;vertical-align:top;text-align:left}}
 table.m th:first-child{{text-align:left;color:#555;font-weight:600}} table.m td,table.m th:not(:first-child){{text-align:center}}
 table.full{{table-layout:auto;width:100%}}
 table.full td:nth-child(4){{min-width:280px}}
 .expl{{white-space:pre-wrap;background:#fafafa;border-left:3px solid #ddd;padding:8px;margin-top:6px;font-size:13px}}
 .legend span{{margin-right:18px}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle}}
 .ring{{display:inline-block;width:9px;height:9px;border-radius:50%;border:2.5px solid {DIVERGE};margin-right:5px;vertical-align:middle}}
 .two{{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}}
 table.conf{{width:auto;border:none}} table.conf th{{border:none;color:#888;font-size:12px;text-align:center}}
 table.conf td{{border:1px solid #eee;text-align:center;font-size:18px;font-weight:600;width:64px;height:40px}}
 table.conf td.hit{{background:#eaf3ea}} table.conf td.miss{{background:#fdecea}}
</style>
<h1>Part 2 — Recover the refusal direction from NLA-derived labels</h1>
<div class="sub">run <code>{html.escape(d['run_id'])}</code> · judge {html.escape(d['judge_model'])}
 · {s['n_valid']}/{s['n_total']} judged · train {s['n_train']} / test {s['n_test']}</div>

<div class="cards">
 <div class="card">cosine(NLA dir, Arditi dir)<br><span class="big">{s['L53_cosine_nla_arditi']:.3f}</span><br>L53 — direction recovery</div>
 <div class="card">AUROC on test (L53)<br><span class="big">{s['L53_auroc_test_nla']:.3f}</span><br>NLA-label probe (Arditi {s['L53_auroc_test_arditi']:.3f})</div>
 <div class="card">judge vs source<br><span class="big">{s['judge_vs_true_agreement']*100:.0f}%</span><br>vs probe-fires {s['judge_vs_fires_agreement']*100:.0f}%</div>
 <div class="card">divergence cases<br><span class="big">{s['n_divergence_cases']}</span><br>NLA label ≠ source or probe</div>
</div>

<h3>Do NLA labels reproduce the probe? (held-out test, L53)</h3>
<p class="sub">Same prompts, same x-position (their projection onto Arditi's refusal direction), shown twice.
If the two lanes have the same color pattern, the NLA judge labels reproduce the probe. Orange ring = the prompt where they disagree.</p>
<div class="legend"><span><span class="dot" style="background:{HARMFUL}"></span>harmful / refusal</span>
 <span><span class="dot" style="background:{HARMLESS}"></span>harmless / comply</span>
 <span><span class="ring"></span>divergence (source ≠ NLA)</span></div>
{strip_svg(pts)}

<h3>Two directions agree · NLA-judge confusion</h3>
<div class="two">
 <div style="flex:1;min-width:340px">{scatter_svg(pts)}
  <div class="sub">Each test point projected onto Arditi's direction (x) and the NLA-label direction (y), z-scored.
  Tight diagonal = the directions rank prompts identically.</div></div>
 <div>{confusion_html(pts)}<div class="sub" style="max-width:200px">NLA judge vs source label on the {len(pts)} test prompts. Off-diagonal = divergence.</div></div>
</div>

<h3>Per-layer metrics</h3>
{table}

<h3>Divergence cases <span style="font-weight:400;color:#888;font-size:13px">(NLA label ≠ source or probe)</span></h3>
<table class="full">{case_header}
{div_rows}</table>

<h3>Hedged cases <span style="font-weight:400;color:#888;font-size:13px">(0.2 &lt; p(refusal) &lt; 0.8 — the NLA explanation was ambiguous)</span></h3>
<table class="full">{case_header}
{hedged_rows}</table>
"""


def main():
    import _cli
    if _cli.interactive():
        path = _cli.choose_file("part2_recover_*.json", "Part 2 results to render")
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else latest()
    out = path.rsplit(".", 1)[0] + ".html"
    with open(out, "w") as f:
        f.write(render(path))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
