"""Render a Part 3 audit JSON as a self-contained HTML report (no plotting deps).

Shows: confound-strength cards; the de-confounded 2×2 fire-rate grids for the confounded probe
vs the clean Part 1 probe; and the headline — NLA-audit theme distributions over each probe's
firing set (a confounded probe should skew "format", the clean probe "refusal"/"harmful_topic").

Run:  python scripts/viz_part3.py [part3_audit_*.json]   # default: latest
"""
import glob
import html
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
CELLS = ["harmful_mcq", "harmful_free", "harmless_mcq", "harmless_free"]
THEME_COLORS = {"format": "#e09f3e", "refusal": "#d1495b", "harmful_topic": "#9b5de5",
                "other": "#9aa0a6", "unparseable": "#d0d0d0"}


def latest():
    files = glob.glob(os.path.join(RESULTS_DIR, "part3_audit_*.json"))
    if not files:
        sys.exit("No part3_audit_*.json in data/results/")
    return max(files, key=os.path.getmtime)


def grid(fr, title):
    """2×2 fire-rate grid (rows harmful/harmless, cols MCQ/free)."""
    def cell(name):
        v = fr[name]
        bg = f"rgba(209,73,91,{0.12 + 0.78 * v:.2f})"  # red intensity by fire-rate
        return f'<td style="background:{bg}">{v*100:.0f}%</td>'
    return f"""<table class="grid"><caption>{title}</caption>
 <tr><th></th><th>MCQ</th><th>free</th></tr>
 <tr><th>harmful</th>{cell('harmful_mcq')}{cell('harmful_free')}</tr>
 <tr><th>harmless</th>{cell('harmless_mcq')}{cell('harmless_free')}</tr></table>"""


def theme_bar(dist, label):
    segs = []
    for th in ("format", "refusal", "harmful_topic", "other", "unparseable"):
        pct = dist.get(th, 0) * 100
        if pct <= 0:
            continue
        segs.append(f'<span style="display:inline-block;width:{pct:.1f}%;background:{THEME_COLORS[th]};'
                    f'color:#fff;font-size:11px;text-align:center;overflow:hidden;white-space:nowrap" '
                    f'title="{th} {pct:.0f}%">{th if pct>12 else ""}</span>')
    return (f'<div style="margin:8px 0"><div style="font-size:13px;color:#555">{label} '
            f'<span style="color:#999">(n={dist.get("n",0)})</span></div>'
            f'<div style="display:flex;height:26px;border-radius:4px;overflow:hidden">{"".join(segs)}</div></div>')


def render(path):
    d = json.load(open(path))
    fr = d["fire_rates"]
    st = d["confound_strength"]
    aud = d["audit"]
    fmt_conf = aud["confounded_probe_fires"].get("format", 0) * 100
    fmt_clean = aud["clean_probe_fires"].get("format", 0) * 100

    legend = " ".join(
        f'<span><span class="dot" style="background:{c}"></span>{t}</span>'
        for t, c in THEME_COLORS.items())

    rows = "".join(
        f'<tr><td>{html.escape(e["cell"])}</td>'
        f'<td style="text-align:center">{"●" if e["fires_confound"] else "○"}</td>'
        f'<td style="text-align:center">{"●" if e["fires_clean"] else "○"}</td>'
        f'<td style="color:{THEME_COLORS.get(e["theme"],"#999")};font-weight:600">{e["theme"]}</td>'
        f'<td>{html.escape(e["instruction"][:110])}</td>'
        f'<td><details><summary>show</summary><div class="expl">{html.escape(e["explanation"])}</div></details></td></tr>'
        for e in d["examples"])

    by_cell = "".join(theme_bar(aud["by_cell"][c], c) for c in CELLS)

    return f"""<!doctype html><meta charset="utf-8">
<title>Part 3 — {html.escape(d['run_name'])}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;color:#222}}
 h1{{font-size:20px}} h3{{margin-top:30px}} .sub{{color:#777;font-size:13px;margin-bottom:18px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
 .card{{background:#f6f7f9;border-radius:8px;padding:12px 16px;flex:1;min-width:160px}}
 .card .big{{font-size:26px;font-weight:700}}
 .two{{display:flex;gap:30px;flex-wrap:wrap}}
 table{{border-collapse:collapse;margin-top:6px}} table.full{{width:100%}}
 th,td{{border-bottom:1px solid #eee;padding:6px 9px;text-align:left;vertical-align:top}}
 table.grid td{{text-align:center;border:1px solid #eee;width:70px;font-weight:600}}
 table.grid caption{{font-size:13px;color:#555;margin-bottom:4px;text-align:left}}
 table.grid th{{border:none;color:#888;font-size:12px;text-align:center}}
 .expl{{white-space:pre-wrap;background:#fafafa;border-left:3px solid #ddd;padding:8px;margin-top:6px;font-size:13px}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}}
 .legend span{{margin-right:14px;font-size:12px;color:#555}}
</style>
<h1>Part 3 — Does the NLA audit catch a planted MCQ-format confound?</h1>
<div class="sub">run <code>{html.escape(d['run_name'])}</code> · {html.escape(d['model'])} ·
 clean probe {html.escape(d['clean_probe_src'])} · 2×2 = {len(d['examples'])} prompts</div>

<div class="cards">
 <div class="card">"format" theme on<br><b>confounded</b> probe fires<br><span class="big" style="color:{THEME_COLORS['format']}">{fmt_conf:.0f}%</span></div>
 <div class="card">"format" theme on<br><b>clean</b> probe fires<br><span class="big" style="color:{THEME_COLORS['format']}">{fmt_clean:.0f}%</span></div>
 <div class="card">confounded probe<br>format effect (harmless)<br><span class="big">{st['confound_format_effect_harmless']*100:+.0f}%</span><br><span class="sub">fires on benign MCQ vs benign prose</span></div>
 <div class="card">confounded probe<br>concept effect<br><span class="big">{st['confound_concept_effect']*100:+.0f}%</span><br><span class="sub">harmful vs harmless (free)</span></div>
</div>

<h3>De-confounded 2×2 — fire rates</h3>
<p class="sub">A pure <b>format</b> detector lights up the MCQ column (both rows); a pure <b>concept</b>
detector lights up the harmful row (both columns).</p>
<div class="two">{grid(fr['confounded'], 'Confounded probe (harmful-MCQ vs harmless-free)')}
 {grid(fr['clean'], 'Clean Part 1 refusal probe')}</div>

<h3>NLA audit — theme distribution over each probe's firing set</h3>
<div class="legend">{legend}</div>
{theme_bar(aud['confounded_probe_fires'], 'Confounded probe — what it fires on')}
{theme_bar(aud['clean_probe_fires'], 'Clean probe — what it fires on')}

<h3>Theme by 2×2 cell</h3>
{by_cell}

<h3>Per-prompt detail</h3>
<table class="full"><tr><th>cell</th><th>conf fires</th><th>clean fires</th><th>theme</th><th>instruction</th><th>NLA explanation</th></tr>
{rows}</table>
"""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest()
    out = path.rsplit(".", 1)[0] + ".html"
    with open(out, "w") as f:
        f.write(render(path))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
